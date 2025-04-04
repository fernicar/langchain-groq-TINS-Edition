import re
import tiktoken
from typing import Tuple, Optional

# --- Constants ---
# Use a common model for token counting, gpt-3.5-turbo is often sufficient
# Adjust if Groq models have significantly different tokenization needs
DEFAULT_TOKEN_MODEL = "gpt-3.5-turbo"
try:
    TOKEN_ENCODING = tiktoken.encoding_for_model(DEFAULT_TOKEN_MODEL)
except KeyError:
    print(f"Warning: Model {DEFAULT_TOKEN_MODEL} not found for tiktoken. Using cl100k_base.")
    TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


# --- Functions ---

def count_tokens(text: str) -> int:
    """Counts tokens using the pre-loaded tiktoken encoder."""
    if not text:
        return 0
    return len(TOKEN_ENCODING.encode(text))

def parse_llm_response(response_text: str) -> Tuple[str, str]:
    """
    Extracts narrative content and thinking content from the LLM response.
    Assumes thinking content is within <think>...</think> tags.

    Args:
        response_text: The raw response string from the LLM.

    Returns:
        A tuple containing (narrative_content, thinking_content).
        Thinking content will be empty if tags are not found.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    thinking_parts = []
    narrative_parts = []
    last_end = 0

    for match in think_pattern.finditer(response_text):
        start, end = match.span()
        # Add narrative part before the tag
        narrative_parts.append(response_text[last_end:start].strip())
        # Extract thinking part
        thinking_parts.append(match.group(1).strip())
        last_end = end

    # Add any remaining narrative part after the last tag
    narrative_parts.append(response_text[last_end:].strip())

    narrative_content = "\n".join(filter(None, narrative_parts)).strip()
    thinking_content = "\n".join(filter(None, thinking_parts)).strip()

    # If no think tags were found, the whole response is narrative
    if not thinking_parts and not narrative_parts:
         narrative_content = response_text.strip()

    # Handle case where only think tags were found (unlikely but possible)
    # Or if the pattern logic resulted in empty narrative but had matches
    if not narrative_content and thinking_parts:
        # Fallback: if we extracted thinking but narrative is empty,
        # maybe the whole thing was meant to be narrative despite tags?
        # Or perhaps the logic above needs refinement for edge cases.
        # For now, let's prioritize narrative if no tags found,
        # otherwise assume the split is correct.
        pass # narrative_content remains empty if only thinking was extracted


    # If no think tags at all, the entire response is narrative
    if not thinking_parts and last_end == 0:
        narrative_content = response_text.strip()


    return narrative_content, thinking_content