from typing import List, Sequence, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from utils import count_tokens # Use our utility

class TokenWindowDualStateMemory(BaseChatMessageHistory):
    """
    Chat history that stores messages in memory, manages a token limit,
    and supports a proposal/commit/discard workflow.
    """
    _messages_committed: List[BaseMessage]
    _messages_proposal: Optional[List[BaseMessage]] = None
    _has_pending_proposal: bool = False # Flag to indicate if proposal differs from committed

    def __init__(self, max_tokens: int = 12000):
        self._messages_committed = []
        self._messages_proposal = None
        self._has_pending_proposal = False
        self.max_tokens = max_tokens

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the current messages (proposal if pending, else committed)."""
        if self._has_pending_proposal and self._messages_proposal is not None:
            return self._messages_proposal
        return self._messages_committed

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        """Set the messages, updating the proposal state."""
        # When messages are set externally (e.g., by RunnableWithMessageHistory),
        # it implies a new proposal based on the input.
        if not self._has_pending_proposal:
             # If no proposal was pending, create one based on committed state
            self._messages_proposal = list(self._messages_committed)
            self._has_pending_proposal = True # Now a proposal exists

        # Update the proposal list
        self._messages_proposal = messages
        self._truncate_messages(self._messages_proposal) # Apply token limit


    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history, managing the proposal state."""
        if not self._has_pending_proposal:
            # Start a new proposal based on the committed state
            self._messages_proposal = list(self._messages_committed)
            self._has_pending_proposal = True

        # Add to the proposal list (ensure it exists)
        if self._messages_proposal is None: # Should not happen if logic above is correct
             self._messages_proposal = []
        self._messages_proposal.append(message)
        self._truncate_messages(self._messages_proposal)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages to the history."""
        if not self._has_pending_proposal:
            self._messages_proposal = list(self._messages_committed)
            self._has_pending_proposal = True

        if self._messages_proposal is None:
            self._messages_proposal = []

        self._messages_proposal.extend(messages)
        self._truncate_messages(self._messages_proposal)


    def _truncate_messages(self, message_list: List[BaseMessage]):
        """Truncates the message list from the beginning to fit max_tokens."""
        if not message_list:
            return

        current_tokens = 0
        # Iterate backwards to keep newest messages
        for i in range(len(message_list) - 1, -1, -1):
            msg = message_list[i]
            # Ensure content is string before counting tokens
            content_str = ""
            if isinstance(msg.content, str):
                content_str = msg.content
            elif isinstance(msg.content, list): # Handle cases like AIMessage chunks
                 content_str = "".join([chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in msg.content])

            msg_tokens = count_tokens(content_str)

            if current_tokens + msg_tokens <= self.max_tokens:
                current_tokens += msg_tokens
            else:
                # Exceeded limit, truncate from this point forward
                # Add 1 because slice end is exclusive
                original_len = len(message_list)
                del message_list[0 : i + 1]
                # print(f"Truncated {original_len - len(message_list)} messages. New count: {len(message_list)}, Tokens: {current_tokens}")
                break # Stop after truncation


    def get_token_count(self, message_list: Optional[List[BaseMessage]] = None) -> int:
        """Calculates the token count for the given message list or current active list."""
        target_list = message_list if message_list is not None else self.messages
        total_tokens = 0
        for msg in target_list:
             content_str = ""
             if isinstance(msg.content, str):
                 content_str = msg.content
             elif isinstance(msg.content, list):
                 content_str = "".join([chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in msg.content])
             total_tokens += count_tokens(content_str)
        return total_tokens

    def prepare_for_response(self):
        """
        Called *before* an LLM call. Backs up the current active messages
        (which become the input to the LLM) to the 'committed' state,
        conceptually freezing the history *before* the new generation starts.
        The actual 'proposal' is the AI's response *after* this point.
        """
        # The messages property already reflects the state *including* the latest user input
        # which is about to be sent. We commit this state.
        self._messages_committed = list(self.messages) # Freeze the state being sent
        self._messages_proposal = None # Clear proposal, new one comes from LLM
        self._has_pending_proposal = False # No *user* proposal pending
        # print(f"Prepared for response. Committed {len(self._messages_committed)} messages.")


    def commit_proposal(self):
        """Commits the current proposal state to the main history."""
        if self._has_pending_proposal and self._messages_proposal is not None:
            self._messages_committed = list(self._messages_proposal)
            self._messages_proposal = None # Clear proposal
            self._has_pending_proposal = False
            # print(f"Committed proposal. Committed {len(self._messages_committed)} messages.")
        # else:
            # print("Commit called, but no pending proposal.")


    def discard_proposal(self):
        """Discards the current proposal, reverting to the last committed state."""
        # Revert the proposal list back to the committed list
        self._messages_proposal = None # Clear proposal content
        self._has_pending_proposal = False # No longer a pending proposal
        # print(f"Discarded proposal. Active messages reverted to committed ({len(self._messages_committed)}).")


    def clear(self) -> None:
        """Clear all messages from the memory."""
        self._messages_committed = []
        self._messages_proposal = None
        self._has_pending_proposal = False
        # print("Memory cleared.")

    # --- Serialization/Deserialization (Optional but good practice) ---
    def to_dict(self) -> dict:
        """Serializes the memory state."""
        return {
            "max_tokens": self.max_tokens,
            "messages_committed": [message_to_dict(m) for m in self._messages_committed],
            "messages_proposal": [message_to_dict(m) for m in self._messages_proposal] if self._messages_proposal else None,
            "has_pending_proposal": self._has_pending_proposal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenWindowDualStateMemory":
        """Deserializes the memory state."""
        memory = cls(max_tokens=data.get("max_tokens", 12000))
        memory._messages_committed = messages_from_dict(data.get("messages_committed", []))
        proposal_data = data.get("messages_proposal")
        if proposal_data is not None:
            memory._messages_proposal = messages_from_dict(proposal_data)
        else:
             memory._messages_proposal = None
        memory._has_pending_proposal = data.get("has_pending_proposal", False)
        return memory