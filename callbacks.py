import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# Import Qt Signals if using PySide6/PyQt for decoupling
try:
    from PySide6.QtCore import Signal, QObject
    # Make the callback handler a QObject to emit signals
    class BaseCallbackHandler_QObject(BaseCallbackHandler, QObject):
        # Define signals to emit data to the UI thread safely
        llm_start = Signal(str, object, object) # unique_id, llm_info, prompts
        llm_end = Signal(str, object) # unique_id, response
        llm_error = Signal(str, object, object) # unique_id, error, kwargs
        chain_start = Signal(str, object, object) # unique_id, chain_info, inputs
        chain_end = Signal(str, object) # unique_id, outputs
        chain_error = Signal(str, object, object) # unique_id, error, kwargs
except ImportError:
    print("Warning: PySide6 not found. API Monitor callback will print to console instead of using Qt Signals.")
    QObject = object # Fallback type
    # Define dummy signal class if Qt is not available
    class Signal:
        def __init__(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass
    # Fallback class definition
    class BaseCallbackHandler_QObject(BaseCallbackHandler, QObject):
         # Define signals to emit data to the UI thread safely
        llm_start = Signal(str, object, object) # unique_id, llm_info, prompts
        llm_end = Signal(str, object) # unique_id, response
        llm_error = Signal(str, object, object) # unique_id, error, kwargs
        chain_start = Signal(str, object, object) # unique_id, chain_info, inputs
        chain_end = Signal(str, object) # unique_id, outputs
        chain_error = Signal(str, object, object) # unique_id, error, kwargs


class APIMonitorCallback(BaseCallbackHandler_QObject):
    """Callback handler to log raw API interactions."""

    def __init__(self, *args, **kwargs):
        # Initialize QObject part if PySide6 is available
        if QObject is not object:
             super().__init__(*args, **kwargs) # Calls both BaseCallbackHandler and QObject init

    def _format_time(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _serialize_if_needed(self, data: Any) -> Any:
        """Attempt to serialize complex objects for logging."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        if isinstance(data, (list, dict)):
            # Recursively serialize contents
            try:
                 # Use default=str to handle non-serializable types gracefully
                return json.loads(json.dumps(data, default=str))
            except (TypeError, OverflowError):
                 return str(data) # Fallback to string representation
        if isinstance(data, BaseMessage):
             # Langchain messages might need custom serialization if default fails
             try:
                 from langchain_core.messages import message_to_dict
                 return message_to_dict(data)
             except Exception:
                 return str(data) # Fallback
        # Add more specific type handling if needed
        return str(data) # Default fallback

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        """Log LLM start."""
        log_entry = {
            "timestamp": self._format_time(),
            "type": "LLM Request",
            "run_id": str(run_id),
            "llm_info": self._serialize_if_needed(serialized),
            "prompts": self._serialize_if_needed(prompts),
            "kwargs": self._serialize_if_needed(kwargs)
        }
        log_json = json.dumps(log_entry, indent=2)
        # print(f"--- LLM Start ---\n{log_json}\n--- End LLM Start ---")
        self.llm_start.emit(str(run_id), log_entry, prompts) # Emit signal

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        """Log LLM end."""
         # Serialize LLMResult carefully
        response_data = {
             "generations": [],
             "llm_output": self._serialize_if_needed(response.llm_output),
             "run": self._serialize_if_needed(response.run)
        }
        for gen_list in response.generations:
             serialized_gen_list = []
             for gen in gen_list:
                 # Serialize Generation object
                 serialized_gen_list.append({
                     "text": gen.text,
                     "generation_info": self._serialize_if_needed(gen.generation_info),
                     # Add other fields if needed e.g. message type
                     "type": getattr(gen, 'type', 'Generation') # or gen.__class__.__name__
                 })
             response_data["generations"].append(serialized_gen_list)

        log_entry = {
            "timestamp": self._format_time(),
            "type": "LLM Response",
            "run_id": str(run_id),
            "response": response_data,
            "kwargs": self._serialize_if_needed(kwargs)
        }
        log_json = json.dumps(log_entry, indent=2)
        # print(f"--- LLM End ---\n{log_json}\n--- End LLM End ---")
        self.llm_end.emit(str(run_id), log_entry) # Emit signal

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        """Log LLM error."""
        log_entry = {
            "timestamp": self._format_time(),
            "type": "LLM Error",
            "run_id": str(run_id),
            "error": str(error),
            "error_type": type(error).__name__,
            "kwargs": self._serialize_if_needed(kwargs)
        }
        log_json = json.dumps(log_entry, indent=2)
        # print(f"--- LLM Error ---\n{log_json}\n--- End LLM Error ---")
        self.llm_error.emit(str(run_id), log_entry, kwargs) # Emit signal


    # --- Optional: Chain level monitoring ---
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        log_entry = {
            "timestamp": self._format_time(),
            "type": "Chain Start",
            "run_id": str(run_id),
            "chain_info": self._serialize_if_needed(serialized),
            "inputs": self._serialize_if_needed(inputs),
            "kwargs": self._serialize_if_needed(kwargs)
        }
        self.chain_start.emit(str(run_id), log_entry, inputs)


    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
         log_entry = {
            "timestamp": self._format_time(),
            "type": "Chain End",
            "run_id": str(run_id),
            "outputs": self._serialize_if_needed(outputs),
            "kwargs": self._serialize_if_needed(kwargs)
        }
         self.chain_end.emit(str(run_id), log_entry)


    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any
    ) -> Any:
        log_entry = {
            "timestamp": self._format_time(),
            "type": "Chain Error",
            "run_id": str(run_id),
            "error": str(error),
            "error_type": type(error).__name__,
            "kwargs": self._serialize_if_needed(kwargs)
        }
        self.chain_error.emit(str(run_id), log_entry, kwargs)