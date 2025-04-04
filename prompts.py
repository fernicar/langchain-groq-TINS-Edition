import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_SYSTEM_PROMPT_NAME = "Narrative Writer"
DEFAULT_SYSTEM_PROMPT_CONTENT = """You are a creative writing assistant. Your goal is to collaborate with the user to write a story.
Generate only the requested narrative content based on the user's input and the preceding story text.
Do NOT include any meta-commentary, apologies, questions, or explanations about your process unless specifically asked.
Focus on producing publication-ready prose in the established style and tone.
If you need to think or plan, use <think>...</think> tags. These tags will be hidden from the user.
For example:
<think>The user wants a description of the forest. I should focus on sensory details and maintain the established melancholic tone.</think>
The ancient trees stood like silent sentinels, their gnarled branches reaching for the perpetually overcast sky. A damp chill clung to the air, heavy with the scent of moss and decay.
"""

class SystemPromptManager:
    """Manages loading, saving, and accessing system prompts from a JSON file."""

    def __init__(self, filepath: str = "system_prompts.json"):
        self.filepath = Path(filepath)
        self.prompts_data: Dict = {}
        self.load()

    def _default_structure(self) -> Dict:
        """Returns the default structure for the prompts file."""
        now = datetime.now().isoformat()
        return {
            "active_prompt": DEFAULT_SYSTEM_PROMPT_NAME,
            "prompts": {
                DEFAULT_SYSTEM_PROMPT_NAME: {
                    "content": DEFAULT_SYSTEM_PROMPT_CONTENT,
                    "created_at": now,
                    "last_used": now,
                }
            }
        }

    def load(self):
        """Loads prompts from the JSON file, creating a default if it doesn't exist."""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.prompts_data = json.load(f)
                # Basic validation
                if "active_prompt" not in self.prompts_data or "prompts" not in self.prompts_data:
                    print(f"Warning: '{self.filepath}' has invalid structure. Resetting to default.")
                    self.prompts_data = self._default_structure()
                    self.save()
            else:
                print(f"Prompt file '{self.filepath}' not found. Creating default.")
                self.prompts_data = self._default_structure()
                self.save()
        except json.JSONDecodeError:
            print(f"Error decoding JSON from '{self.filepath}'. Resetting to default.")
            self.prompts_data = self._default_structure()
            self.save()
        except Exception as e:
            print(f"Error loading prompts from '{self.filepath}': {e}")
            # Fallback to in-memory default if loading/saving fails critically
            self.prompts_data = self._default_structure()


    def save(self):
        """Saves the current prompts data to the JSON file."""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.prompts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving prompts to '{self.filepath}': {e}")

    def get_prompt_names(self) -> List[str]:
        """Returns a sorted list of available prompt names."""
        return sorted(self.prompts_data.get("prompts", {}).keys())

    def get_active_prompt_name(self) -> str:
        """Gets the name of the currently active prompt."""
        # Ensure active prompt exists, otherwise fallback
        active_name = self.prompts_data.get("active_prompt", DEFAULT_SYSTEM_PROMPT_NAME)
        if active_name not in self.prompts_data.get("prompts", {}):
            print(f"Warning: Active prompt '{active_name}' not found. Falling back to default.")
            self.set_active_prompt(DEFAULT_SYSTEM_PROMPT_NAME) # Also saves
            return DEFAULT_SYSTEM_PROMPT_NAME
        return active_name

    def set_active_prompt(self, name: str) -> bool:
        """Sets the active prompt by name and saves."""
        if name in self.prompts_data.get("prompts", {}):
            self.prompts_data["active_prompt"] = name
            # Update last used timestamp
            now = datetime.now().isoformat()
            self.prompts_data["prompts"][name]["last_used"] = now
            self.save()
            return True
        else:
            print(f"Error: Cannot set active prompt. Name '{name}' not found.")
            return False

    def get_prompt(self, name: str) -> Optional[Tuple[str, Dict]]:
        """Gets the content and metadata of a prompt by name."""
        prompt_info = self.prompts_data.get("prompts", {}).get(name)
        if prompt_info:
            return prompt_info.get("content"), prompt_info
        return None, None

    def get_active_prompt_content(self) -> str:
        """Gets the content of the currently active prompt."""
        active_name = self.get_active_prompt_name()
        content, _ = self.get_prompt(active_name)
        return content if content is not None else DEFAULT_SYSTEM_PROMPT_CONTENT


    def save_prompt(self, name: str, content: str):
        """Saves or updates a prompt and saves the file."""
        if not name:
            print("Error: Prompt name cannot be empty.")
            return False
        if name not in self.prompts_data.get("prompts", {}):
            # Create new prompt
            now = datetime.now().isoformat()
            self.prompts_data.setdefault("prompts", {})[name] = {
                "content": content,
                "created_at": now,
                "last_used": now,
            }
            print(f"Prompt '{name}' created.")
        else:
            # Update existing prompt
             self.prompts_data["prompts"][name]["content"] = content
             self.prompts_data["prompts"][name]["last_used"] = datetime.now().isoformat()
             print(f"Prompt '{name}' updated.")
        self.save()
        return True


    def delete_prompt(self, name: str) -> bool:
        """Deletes a prompt by name and saves."""
        prompts = self.prompts_data.get("prompts", {})
        if name == DEFAULT_SYSTEM_PROMPT_NAME:
            print("Error: Cannot delete the default prompt.")
            return False
        if name in prompts:
            del prompts[name]
            print(f"Prompt '{name}' deleted.")
            # If the deleted prompt was active, reset to default
            if self.prompts_data.get("active_prompt") == name:
                self.prompts_data["active_prompt"] = DEFAULT_SYSTEM_PROMPT_NAME
                print(f"Active prompt reset to '{DEFAULT_SYSTEM_PROMPT_NAME}'.")
            self.save()
            return True
        else:
            print(f"Error: Prompt '{name}' not found for deletion.")
            return False