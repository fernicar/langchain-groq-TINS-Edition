import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Load .env file first
from dotenv import load_dotenv, set_key, find_dotenv

load_dotenv(find_dotenv(usecwd=True)) # Load .env from current working directory or parent

# Import Qt Components
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QTabWidget, QSplitter, QMenuBar, QToolBar, QFileDialog,
    QMessageBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QSizePolicy, QDialog, QDialogButtonBox, QFormLayout, QStyleFactory
)
from PySide6.QtGui import QAction, QKeySequence, QTextCursor, QColor, QShortcut
from PySide6.QtCore import Qt, Slot, QSize, QSettings # Added QSettings

# Import Langchain & Groq Components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import groq # For listing models and API key check

# Import Custom Modules
from memory import TokenWindowDualStateMemory
from prompts import SystemPromptManager
from callbacks import APIMonitorCallback
from utils import parse_llm_response, count_tokens # Added count_tokens

# --- Constants ---
APP_NAME = "Narrative Collaboration System (TINS Edition)"
APP_VERSION = "1.0"
SETTINGS_ORG = "ScuffedEpoch"
SETTINGS_APP = "TINS_NarrativeCollab"
DEFAULT_WINDOW_SIZE = QSize(1200, 800)
DEFAULT_FONT_SIZE = 11
FALLBACK_GROQ_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
MAX_HISTORY_SIMULATION_CHUNKS = 5 # How many chunks from end of loaded file to simulate history


# --- API Key Dialog ---
class APIKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Groq API Key Required")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        label = QLabel("A Groq API key is required to use this application.\n"
                       "You can obtain one from console.groq.com/keys.\n"
                       "Your key will be stored securely in a local '.env' file.")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter your Groq API key (e.g., gsk_...)")
        layout.addWidget(self.key_input)

        # Add link label (non-clickable in basic QLabel)
        link_label = QLabel("<a href='https://console.groq.com/keys'>Visit Groq Console (console.groq.com/keys)</a>")
        link_label.setOpenExternalLinks(True)
        layout.addWidget(link_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_key(self) -> Optional[str]:
        if self.exec() == QDialog.Accepted:
            return self.key_input.text().strip()
        return None


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP) # For UI state like theme/font

        # --- State Variables ---
        self.groq_api_key: Optional[str] = None
        self.groq_client: Optional[groq.Groq] = None
        self.available_models: List[str] = []
        self.llm: Optional[ChatGroq] = None
        self.chain: Optional[RunnableWithMessageHistory] = None
        self.memory: Optional[TokenWindowDualStateMemory] = None
        self.prompt_manager: Optional[SystemPromptManager] = None
        self.api_monitor_callback: Optional[APIMonitorCallback] = None

        self.canon_validated: List[str] = [] # Committed story parts
        self.current_narrative: str = ""     # Blue text proposal
        self.current_file_path: Optional[str] = None
        self._is_dirty: bool = False # Tracks unsaved blue text proposal

        # --- Initialization ---
        if not self._check_api_key():
             sys.exit(1) # Exit if key setup failed/cancelled

        self.prompt_manager = SystemPromptManager()
        self.api_monitor_callback = APIMonitorCallback()
        self._fetch_groq_models()

        self._init_ui()
        self._init_memory_and_llm_chain() # Now safe to init chain
        self._update_displays()
        self._update_status_bar()
        self._load_settings() # Load theme/font after UI exists


    def _check_api_key(self) -> bool:
        """Checks for Groq API key in env, prompts if missing."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            dialog = APIKeyDialog(self)
            key = dialog.get_key()
            if key:
                # Validate format (optional but recommended)
                if not (key.startswith("gsk_") and len(key) == 56):
                     reply = QMessageBox.warning(
                        self, "Invalid Key Format",
                        f"The entered key does not match the expected format (starts with 'gsk_', 56 characters).\n\n'{key[:10]}...' (length {len(key)})\n\nDo you want to save it anyway?",
                        QMessageBox.Save | QMessageBox.Cancel,
                        QMessageBox.Cancel
                     )
                     if reply == QMessageBox.Cancel:
                         QMessageBox.critical(self, "API Key Error", "API Key not saved. Application cannot continue.")
                         return False

                # Save to .env file
                env_path = find_dotenv(usecwd=True) or ".env" # Find existing or create in CWD
                if set_key(env_path, "GROQ_API_KEY", key):
                    os.environ["GROQ_API_KEY"] = key # Set in current environment
                    self.groq_api_key = key
                    QMessageBox.information(self, "API Key Saved", f"API Key saved successfully to '{env_path}'.")
                else:
                    QMessageBox.critical(self, "API Key Error", f"Failed to save API Key to '{env_path}'. Application cannot continue.")
                    return False
            else:
                QMessageBox.critical(self, "API Key Required", "Groq API Key is required. Application cannot continue.")
                return False

        # Initialize Groq client now that we have a key
        try:
            self.groq_client = groq.Groq(api_key=self.groq_api_key)
        except Exception as e:
             QMessageBox.critical(self, "Groq Client Error", f"Failed to initialize Groq client: {e}")
             return False

        return True

    def _fetch_groq_models(self):
        """Fetches available models from Groq API."""
        if not self.groq_client:
             self.available_models = FALLBACK_GROQ_MODELS
             QMessageBox.warning(self, "Model Fetch Failed", "Groq client not initialized. Using fallback models.")
             return

        try:
            models_response = self.groq_client.models.list()
            # Filter for chat-like models if possible (heuristic: check context window size?)
            # The README doesn't specify filtering, so include all active ones for now.
            self.available_models = sorted([model.id for model in models_response.data if model.active])
            if not self.available_models:
                raise ValueError("No active models returned by Groq API.")
        except Exception as e:
            print(f"Error fetching Groq models: {e}")
            self.available_models = FALLBACK_GROQ_MODELS
            QMessageBox.warning(self, "Model Fetch Failed", f"Could not fetch models from Groq API: {e}\nUsing fallback list.")

    def _init_ui(self):
        """Creates the user interface elements."""
        self.setWindowTitle(f"{APP_NAME} - New Story")
        self.setGeometry(100, 100, DEFAULT_WINDOW_SIZE.width(), DEFAULT_WINDOW_SIZE.height()) # x, y, w, h

        # --- Central Widget & Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # VBox for Splitter + Toolbar

        # --- Menu Bar ---
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        new_action = QAction("&New Story", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_story)
        file_menu.addAction(new_action)

        load_action = QAction("&Load Story...", self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self._load_story)
        file_menu.addAction(load_action)

        save_action = QAction("&Save Story", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_story)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Story &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._save_story_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Main Horizontal Splitter (Top/Bottom Panes) ---
        main_splitter_h = QSplitter(Qt.Vertical) # Renamed from H to V based on diagram flow
        main_layout.addWidget(main_splitter_h, 1) # Make splitter stretch


        # --- Top Pane (Display Area) ---
        top_pane_widget = QWidget()
        top_pane_layout = QHBoxLayout(top_pane_widget)
        top_pane_layout.setContentsMargins(0,0,0,0)
        display_splitter_v = QSplitter(Qt.Horizontal) # Renamed from V to H
        top_pane_layout.addWidget(display_splitter_v)
        main_splitter_h.addWidget(top_pane_widget)

        # --- Left Display (Story) ---
        left_display_widget = QWidget()
        left_display_layout = QVBoxLayout(left_display_widget)
        story_label = QLabel("Story Development (Canon / <font color='#6495ED'>Proposal</font>):") # Use CornflowerBlue for proposal
        left_display_layout.addWidget(story_label)
        self.story_display = QTextEdit()
        self.story_display.setReadOnly(True)
        self.story_display.setPlaceholderText("Your story will appear here. Use the tabs below to generate content.")
        left_display_layout.addWidget(self.story_display)
        display_splitter_v.addWidget(left_display_widget)


        # --- Right Display (Monitor Tabs) ---
        right_display_widget = QWidget()
        right_display_layout = QVBoxLayout(right_display_widget)
        self.monitor_tabs = QTabWidget()
        right_display_layout.addWidget(self.monitor_tabs)
        display_splitter_v.addWidget(right_display_widget)

        # Context Monitor Tab
        context_tab = QWidget()
        context_layout = QVBoxLayout(context_tab)
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setPlaceholderText("Shows the conversation history sent to the LLM (User/Assistant pairs) and token count.")
        context_layout.addWidget(self.context_display)
        commit_button_context = QPushButton("Commit Blue Text Now")
        commit_button_context.clicked.connect(self._handle_commit)
        context_layout.addWidget(commit_button_context)
        self.monitor_tabs.addTab(context_tab, "Context")

        # Thinking Process Tab
        thinking_tab = QWidget()
        thinking_layout = QVBoxLayout(thinking_tab)
        self.thinking_display = QTextEdit()
        self.thinking_display.setReadOnly(True)
        self.thinking_display.setPlaceholderText("Displays internal thoughts extracted from the AI's <think>...</think> tags in its response.")
        thinking_layout.addWidget(self.thinking_display)
        self.monitor_tabs.addTab(thinking_tab, "Thinking")

        # Conversation Log Tab
        conversation_tab = QWidget()
        conversation_layout = QVBoxLayout(conversation_tab)
        self.conversation_log = QTextEdit()
        self.conversation_log.setReadOnly(True)
        self.conversation_log.setPlaceholderText("Shows the full history of user inputs and raw AI responses (including thinking tags).")
        conversation_layout.addWidget(self.conversation_log)
        self.monitor_tabs.addTab(conversation_tab, "Conversation Log")

        # API Monitor Tab
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        self.api_monitor_display = QTextEdit()
        self.api_monitor_display.setReadOnly(True)
        self.api_monitor_display.setLineWrapMode(QTextEdit.NoWrap) # Good for JSON
        self.api_monitor_display.setPlaceholderText("Shows raw requests, responses, and errors for interactions with the Groq API.")
        api_layout.addWidget(self.api_monitor_display)
        clear_api_monitor_button = QPushButton("Clear API Monitor")
        clear_api_monitor_button.clicked.connect(self.api_monitor_display.clear)
        api_layout.addWidget(clear_api_monitor_button)
        self.monitor_tabs.addTab(api_tab, "API Monitor")


        # --- Bottom Pane (Input Area Tabs) ---
        bottom_pane_widget = QWidget()
        bottom_pane_layout = QVBoxLayout(bottom_pane_widget)
        bottom_pane_layout.setContentsMargins(0,5,0,0) # Add some top margin
        self.input_tabs = QTabWidget()
        bottom_pane_layout.addWidget(self.input_tabs)
        main_splitter_h.addWidget(bottom_pane_widget)


        # Edit Blue Tab
        edit_blue_tab = QWidget()
        edit_blue_layout = QVBoxLayout(edit_blue_tab)
        self.edit_input = QTextEdit()
        self.edit_input.setPlaceholderText("Directly edit the current blue text proposal here. Changes are live.\nUse buttons below or 'Save Blue & Continue' tab to proceed.")
        self.edit_input.textChanged.connect(self._handle_edit_blue_text) # Connect live edit
        edit_blue_layout.addWidget(self.edit_input)
        edit_buttons_layout = QHBoxLayout()
        commit_button_edit = QPushButton("Commit Blue Text Now")
        commit_button_edit.clicked.connect(self._handle_commit)
        discard_button_edit = QPushButton("Discard Blue Text Now")
        discard_button_edit.clicked.connect(self._handle_discard)
        edit_buttons_layout.addWidget(commit_button_edit)
        edit_buttons_layout.addWidget(discard_button_edit)
        edit_blue_layout.addLayout(edit_buttons_layout)
        self.input_tabs.addTab(edit_blue_tab, "Edit Blue")

        # Save Blue & Continue Tab
        continue_tab = QWidget()
        continue_layout = QVBoxLayout(continue_tab)
        self.continue_input = QTextEdit()
        self.continue_input.setPlaceholderText("Enter guidance for the *next* section of the story here.\nClicking 'Send' (or Ctrl+Enter) will first commit the current blue text, then generate the continuation based on your input.")
        continue_layout.addWidget(self.continue_input)
        self.input_tabs.addTab(continue_tab, "Save Blue & Continue")

        # Discard Blue & Rewrite Tab
        rewrite_tab = QWidget()
        rewrite_layout = QVBoxLayout(rewrite_tab)
        self.rewrite_input = QTextEdit()
        self.rewrite_input.setPlaceholderText("Enter guidance for *rewriting* the blue text proposal here.\nClicking 'Send' (or Ctrl+Enter) will first discard the current blue text, then generate a new proposal based on your input.")
        rewrite_layout.addWidget(self.rewrite_input)
        self.input_tabs.addTab(rewrite_tab, "Discard Blue & Rewrite")

        # Customize System Prompt Tab
        system_prompt_tab = QWidget()
        system_prompt_layout = QVBoxLayout(system_prompt_tab)
        # Prompt Name Row
        prompt_name_layout = QHBoxLayout()
        prompt_name_label = QLabel("Prompt Name:")
        self.prompt_name_input = QLineEdit()
        self.prompt_name_input.setPlaceholderText("Enter a name for this prompt")
        prompt_name_layout.addWidget(prompt_name_label)
        prompt_name_layout.addWidget(self.prompt_name_input)
        system_prompt_layout.addLayout(prompt_name_layout)
        # Prompt Content
        self.system_input = QTextEdit()
        self.system_input.setPlaceholderText("Enter the system prompt content here.")
        system_prompt_layout.addWidget(self.system_input)
        # Buttons Row
        prompt_buttons_layout = QHBoxLayout()
        self.save_prompt_button = QPushButton("Save Prompt")
        self.save_prompt_button.clicked.connect(self._handle_save_prompt)
        self.delete_prompt_button = QPushButton("Delete Prompt")
        self.delete_prompt_button.clicked.connect(self._handle_delete_prompt)
        prompt_buttons_layout.addWidget(self.save_prompt_button)
        prompt_buttons_layout.addWidget(self.delete_prompt_button)
        system_prompt_layout.addLayout(prompt_buttons_layout)
        self.input_tabs.addTab(system_prompt_tab, "System Prompt")


        # --- Bottom Toolbar ---
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16)) # Smaller icons if used
        self.addToolBar(Qt.BottomToolBarArea, toolbar)

        # Model Selector
        toolbar.addWidget(QLabel(" Model: "))
        self.model_selector = QComboBox()
        self.model_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.model_selector.setMinimumWidth(150)
        self.model_selector.addItems(self.available_models)
        self.model_selector.currentTextChanged.connect(self._update_llm_params)
        toolbar.addWidget(self.model_selector)

        # Temperature
        toolbar.addWidget(QLabel(" Temp: "))
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 2.0)
        self.temp_spinbox.setSingleStep(0.1)
        self.temp_spinbox.setValue(0.7) # Default temp
        self.temp_spinbox.valueChanged.connect(self._update_llm_params)
        toolbar.addWidget(self.temp_spinbox)

        # Max Tokens (Response)
        toolbar.addWidget(QLabel(" Max Tokens (Resp): "))
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setRange(50, 8192) # Reasonable range for response
        self.max_tokens_spinbox.setSingleStep(10)
        self.max_tokens_spinbox.setValue(1024) # Default max tokens
        self.max_tokens_spinbox.valueChanged.connect(self._update_llm_params)
        toolbar.addWidget(self.max_tokens_spinbox)

        # # Max Tokens (Context) - Handled by Memory class, maybe display limit?
        # self.context_token_limit_label = QLabel(f"Ctx Limit: {self.memory.max_tokens if self.memory else 'N/A'}")
        # toolbar.addWidget(self.context_token_limit_label)

        toolbar.addSeparator()

        # XML Tag Input
        toolbar.addWidget(QLabel(" Wrap Input Tag: "))
        self.xml_tag_input = QLineEdit()
        self.xml_tag_input.setPlaceholderText("e.g., <instruction>")
        self.xml_tag_input.setFixedWidth(120)
        toolbar.addWidget(self.xml_tag_input)

        toolbar.addSeparator()

        # Font Size
        toolbar.addWidget(QLabel(" Font Size: "))
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 24)
        self.font_size_spinbox.setValue(DEFAULT_FONT_SIZE)
        self.font_size_spinbox.valueChanged.connect(self._update_font_size)
        toolbar.addWidget(self.font_size_spinbox)

        # Theme Toggle
        self.theme_button = QPushButton("Toggle Theme")
        self.theme_button.setCheckable(True)
        self.theme_button.toggled.connect(self._toggle_theme)
        toolbar.addWidget(self.theme_button)

        toolbar.addSeparator()

        # System Prompt Selector
        toolbar.addWidget(QLabel(" Sys Prompt: "))
        self.system_prompt_selector = QComboBox()
        self.system_prompt_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.system_prompt_selector.setMinimumWidth(150)
        self.system_prompt_selector.currentTextChanged.connect(self._handle_system_prompt_selection_change)
        toolbar.addWidget(self.system_prompt_selector)
        self._update_system_prompt_selector() # Initial population


        # Send Button (Main Action)
        self.send_button = QPushButton("Send")
        self.send_button.setToolTip("Send input based on active tab (Ctrl+Enter)")
        self.send_button.clicked.connect(self._handle_send)
        toolbar.addWidget(self.send_button)

        # Ctrl+Enter Shortcut for Send Button
        send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        send_shortcut.activated.connect(self._handle_send)
        send_shortcut_alt = QShortcut(QKeySequence("Ctrl+Enter"), self) # Common alternative
        send_shortcut_alt.activated.connect(self._handle_send)


        # --- Initial Splitter Sizes ---
        main_splitter_h.setSizes([int(self.height() * 0.65), int(self.height() * 0.35)])
        display_splitter_v.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])

        # --- Connect API Monitor Signals ---
        if self.api_monitor_callback:
            self.api_monitor_callback.llm_start.connect(self._log_api_event)
            self.api_monitor_callback.llm_end.connect(self._log_api_event)
            self.api_monitor_callback.llm_error.connect(self._log_api_event)
            self.api_monitor_callback.chain_start.connect(self._log_api_event)
            self.api_monitor_callback.chain_end.connect(self._log_api_event)
            self.api_monitor_callback.chain_error.connect(self._log_api_event)


    # --- Initialization & Configuration ---

    def _init_memory_and_llm_chain(self):
        """Initializes the memory and Langchain runnable."""
        if not self.prompt_manager or not self.api_monitor_callback:
             QMessageBox.critical(self, "Initialization Error", "Core managers not ready.")
             return

        # TODO: Add context token limit setting later if needed
        self.memory = TokenWindowDualStateMemory(max_tokens=12000) # Default 12k tokens

        # Update the context token limit label if it exists
        # if hasattr(self, 'context_token_limit_label'):
        #     self.context_token_limit_label.setText(f"Ctx Limit: {self.memory.max_tokens}")


        self._update_llm_params() # Creates LLM instance based on UI selectors
        self._rebuild_chain()     # Builds the chain with the LLM and memory


    def _update_llm_params(self):
        """Updates the LLM instance based on UI settings."""
        model_name = self.model_selector.currentText()
        temperature = self.temp_spinbox.value()
        max_tokens = self.max_tokens_spinbox.value()

        if not model_name:
             QMessageBox.warning(self, "LLM Config Error", "No model selected.")
             return

        try:
            self.llm = ChatGroq(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens, # Max tokens for *response* generation
                groq_api_key=self.groq_api_key,
                # streaming=True # Consider adding streaming later
                callbacks=[self.api_monitor_callback] if self.api_monitor_callback else None
            )
            print(f"LLM Instance Updated: Model={model_name}, Temp={temperature}, MaxRespTokens={max_tokens}")
            self._rebuild_chain() # Rebuild chain when LLM changes
        except Exception as e:
            QMessageBox.critical(self, "LLM Initialization Error", f"Failed to create ChatGroq instance: {e}")
            self.llm = None


    def _rebuild_chain(self):
        """Rebuilds the Langchain chain with current LLM, memory, and system prompt."""
        if not self.llm or not self.memory or not self.prompt_manager:
             print("Chain rebuild skipped: LLM, memory, or prompt manager not ready.")
             return

        system_prompt_content = self.prompt_manager.get_active_prompt_content()

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt_content),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        self.chain = RunnableWithMessageHistory(
            prompt_template | self.llm, # Use the updated self.llm
            lambda session_id: self.memory, # Use our custom memory instance
            input_messages_key="input",
            history_messages_key="history",
        )
        print(f"Langchain chain rebuilt with system prompt: {self.prompt_manager.get_active_prompt_name()}")


    # --- Core Action Handlers ---

    @Slot()
    def _handle_send(self):
        """Handles the main 'Send' action based on the active input tab."""
        if not self.chain:
            QMessageBox.critical(self, "Error", "LLM Chain not initialized.")
            return

        current_tab_index = self.input_tabs.currentIndex()
        tab_text = self.input_tabs.tabText(current_tab_index)
        guidance = ""
        action_type = None # 'continue', 'rewrite', 'edit_commit', 'edit_discard'

        if tab_text == "Save Blue & Continue":
            if self._is_dirty: # Commit if there's uncommitted blue text
                self._handle_commit() # Commit first
            guidance = self.continue_input.toPlainText().strip()
            self.continue_input.clear()
            action_type = 'continue'
            # Commit happens implicitly before sending if needed

        elif tab_text == "Discard Blue & Rewrite":
            self._handle_discard() # Discard first
            guidance = self.rewrite_input.toPlainText().strip()
            self.rewrite_input.clear()
            action_type = 'rewrite'

        elif tab_text == "Edit Blue":
            # Send button doesn't trigger generation from this tab
            # User must use "Commit" or "Discard" buttons within the tab,
            # or switch to Continue/Rewrite tabs.
            QMessageBox.information(self, "Info", "Use 'Commit'/'Discard' buttons in this tab, or switch to 'Continue'/'Rewrite' tabs to generate text.")
            return

        elif tab_text == "System Prompt":
             # Send button action for this tab is Save Prompt
             self._handle_save_prompt()
             return # Don't proceed to LLM invocation

        else:
             QMessageBox.warning(self, "Unknown Tab", f"Action not defined for tab: {tab_text}")
             return

        # Wrap guidance if XML tag is provided
        xml_tag = self.xml_tag_input.text().strip()
        if guidance and xml_tag:
             # Basic tag wrapping, ensure closing tag matches
             tag_name = xml_tag.strip('<>').split()[0] # Get tag name like 'instruction'
             guidance = f"<{tag_name}>{guidance}</{tag_name}>"


        print(f"Invoking LLM (Action: {action_type}). Guidance: '{guidance[:50]}...'")
        # Crucially, this call is synchronous and will block the UI!
        # Needs threading (QThread) for production use.
        self._invoke_llm(guidance if guidance else "Continue the story.")


    def _invoke_llm(self, guidance: str):
        """Invokes the Langchain runnable and handles the response."""
        if not self.chain or not self.memory:
            QMessageBox.critical(self, "Error", "Cannot invoke LLM: Chain or memory not ready.")
            return

        # Log user input to conversation log (before potential wrapping)
        raw_guidance = guidance # Keep raw for log
        self._log_conversation("User", raw_guidance)


        # --- IMPORTANT ---
        # This is the blocking call. In a real app, move this to a QThread.
        # Start a QProgressDialog or similar indicator here.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.send_button.setEnabled(False)
        self.statusBar().showMessage("Generating response...")
        try:
            # Prepare memory *before* the call. This freezes the history sent.
            self.memory.prepare_for_response()

            # Use a dummy session ID as we manage history directly
            config = {"configurable": {"session_id": "main_session"}}
            response = self.chain.invoke({"input": guidance}, config=config)

            # Response object structure depends on the chain/LLM
            # For ChatGroq, response.content should contain the text
            ai_response_raw = ""
            if hasattr(response, 'content'):
                 ai_response_raw = response.content
            elif isinstance(response, dict) and 'output' in response:
                 ai_response_raw = response['output']
            elif isinstance(response, str):
                 ai_response_raw = response
            else:
                 raise TypeError(f"Unexpected response type from LLM chain: {type(response)}")


            # Log raw AI response
            self._log_conversation("Assistant (raw)", ai_response_raw)

            # Parse response for narrative and thinking
            narrative, thinking = parse_llm_response(ai_response_raw)

            # Update state and UI
            self.current_narrative = narrative # Set the blue text proposal
            self._is_dirty = True # Mark as having an uncommitted proposal

            self.thinking_display.setPlainText(thinking if thinking else "No <think> tags found in response.")
            self._update_displays() # Update story display with new blue text
            self._update_monitors() # Update context monitor

        except Exception as e:
            error_message = f"LLM invocation failed: {e}"
            print(error_message)
            QMessageBox.critical(self, "LLM Error", error_message)
            # Optionally log to API monitor even if callback failed
            self._log_api_event_raw("LLM Error (Invoke)", {"error": str(e)})
            # Should we attempt to restore memory state? Discard might be safest.
            self._handle_discard() # Revert memory on error


        finally:
            # Stop progress indicator here.
            QApplication.restoreOverrideCursor()
            self.send_button.setEnabled(True)
            self._update_status_bar() # Update status with counts etc.
        # --- END BLOCKING SECTION ---


    @Slot()
    def _handle_commit(self):
        """Commits the current blue text proposal to the canon."""
        if not self.current_narrative:
            # QMessageBox.information(self, "Commit", "Nothing to commit (no blue text proposal).")
            print("Commit skipped: No blue text proposal.")
            return

        if not self.memory:
            QMessageBox.critical(self, "Error", "Memory not initialized.")
            return

        print("Committing blue text...")
        # Append blue text to canon list
        self.canon_validated.append(self.current_narrative)

        # Commit the state in the memory manager
        # This assumes the AI message corresponding to current_narrative
        # was already added to the memory's proposal list by the invoke process.
        # Commit makes the proposal list the new committed list.
        self.memory.commit_proposal()

        # Clear the blue text state variable
        self.current_narrative = ""
        self._is_dirty = False # No longer dirty

        # Update UI
        self._update_displays()
        self._update_monitors()
        self._update_status_bar()


    @Slot()
    def _handle_discard(self):
        """Discards the current blue text proposal."""
        if not self.current_narrative and not self._is_dirty : # Check dirty flag too
            # QMessageBox.information(self, "Discard", "Nothing to discard.")
            print("Discard skipped: No blue text proposal or not dirty.")
            return

        if not self.memory:
            QMessageBox.critical(self, "Error", "Memory not initialized.")
            return

        print("Discarding blue text proposal...")
        # Discard the proposal state in memory (reverts internal list to last committed)
        self.memory.discard_proposal()

        # Clear the blue text state variable
        self.current_narrative = ""
        self._is_dirty = False # No longer dirty

        # Update UI
        # The last AI message from the *committed* history might be restored,
        # but the spec says just clear blue text. Let's stick to that.
        self.edit_input.clear() # Clear edit area as well
        self.thinking_display.clear() # Clear thinking from discarded proposal
        self._update_displays()
        self._update_monitors()
        self._update_status_bar()


    @Slot()
    def _handle_edit_blue_text(self):
        """Called when text in the 'Edit Blue' tab changes."""
        # Update the state variable directly
        self.current_narrative = self.edit_input.toPlainText()
        self._is_dirty = bool(self.current_narrative) # Mark dirty if there's text

        # Update the main story display live
        self._update_story_display()


    # --- File Operations ---

    def _prompt_save_if_dirty(self) -> bool:
        """Asks user to save/discard if there's blue text. Returns True if okay to proceed."""
        if not self._is_dirty:
            return True # Nothing to save

        reply = QMessageBox.question(
            self, "Uncommitted Changes",
            "You have an uncommitted blue text proposal. Do you want to commit it before proceeding?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel
        )

        if reply == QMessageBox.Yes:
            self._handle_commit()
            return True # Committed, ok to proceed
        elif reply == QMessageBox.No:
            self._handle_discard() # Discard the changes
            return True # Discarded, ok to proceed
        else: # Cancel
            return False # User cancelled, do not proceed


    @Slot()
    def _new_story(self):
        """Clears the current story state."""
        if not self._prompt_save_if_dirty():
            return

        self.canon_validated = []
        self.current_narrative = ""
        self.current_file_path = None
        self._is_dirty = False
        if self.memory:
            self.memory.clear()

        self.edit_input.clear()
        self.thinking_display.clear()
        self.conversation_log.clear()
        # Don't clear API monitor unless requested by user

        self._update_displays()
        self._update_monitors()
        self._update_window_title()
        self._update_status_bar()
        print("New story created.")


    @Slot()
    def _load_story(self):
        """Loads story content from a text file."""
        if not self._prompt_save_if_dirty():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Story", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # --- Clear existing state ---
                self._new_story() # Use new story logic to clear things first
                self.current_file_path = file_path

                # --- Process loaded content ---
                # Split into chunks (e.g., by double newline)
                # Filter out empty strings that might result from splitting
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

                if not chunks:
                     print("Loaded file is empty or contains only whitespace.")
                     # State is already cleared by _new_story()
                else:
                     # Per README correction: All loaded content becomes canon.
                     self.canon_validated = list(chunks)
                     self.current_narrative = "" # No initial blue text
                     self._is_dirty = False

                     # Simulate history using the last N chunks
                     if self.memory:
                          self.memory.clear() # Start fresh history for loaded file
                          history_chunks = self.canon_validated[-MAX_HISTORY_SIMULATION_CHUNKS:]
                          simulated_messages = []
                          print(f"Simulating history from last {len(history_chunks)} chunks...")
                          for i, chunk in enumerate(history_chunks):
                               # Add placeholder User message and actual AI message
                               simulated_messages.append(HumanMessage(content=f"✒️✍️📜 (Simulated user prompt for chunk {i+1})"))
                               simulated_messages.append(AIMessage(content=chunk))
                               print(f"  Simulated Pair {i+1}: User='...', AI='{chunk[:50]}...'")

                          # Add messages to memory, letting it handle truncation
                          self.memory.add_messages(simulated_messages)
                          # Important: After simulating, commit this simulated history
                          # so discard doesn't wipe it immediately.
                          self.memory.commit_proposal() # Commit the simulated state


                self._update_displays()
                self._update_monitors()
                self._update_window_title()
                self._update_status_bar()
                print(f"Story loaded from {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load story from {file_path}:\n{e}")


    @Slot()
    def _save_story(self):
        """Saves the current story canon to the current file or prompts for a new one."""
        # Saving *canon* doesn't require prompting for dirty *proposal*
        # Unless we decide saving should implicitly commit? README implies not.

        if not self.current_file_path:
            self._save_story_as() # Use Save As logic if no path exists
        else:
            self._write_canon_to_file(self.current_file_path)


    @Slot()
    def _save_story_as(self):
        """Prompts for a file path and saves the story canon."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Story As", self.current_file_path or "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
             # Ensure .txt extension if not provided
             if not file_path.lower().endswith(".txt"):
                 file_path += ".txt"
             self.current_file_path = file_path
             self._write_canon_to_file(self.current_file_path)


    def _write_canon_to_file(self, file_path: str):
        """Writes the canon_validated list to the specified file."""
        try:
            # Combine canon chunks with double newlines
            content_to_save = "\n\n".join(self.canon_validated)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_to_save)

            print(f"Story canon saved to {file_path}")
            self._update_window_title()
            self.statusBar().showMessage(f"Story saved to {os.path.basename(file_path)}", 5000) # Show message for 5s
            # If we saved, any *committed* state is now persisted.
            # Should we clear the dirty flag if the blue text was empty? No, dirty is only for blue text.

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save story to {file_path}:\n{e}")


    # --- System Prompt Management ---

    def _update_system_prompt_selector(self):
        """Updates the system prompt dropdown and selects the active one."""
        if not self.prompt_manager: return
        self.system_prompt_selector.blockSignals(True) # Avoid triggering handler during update
        self.system_prompt_selector.clear()
        prompt_names = self.prompt_manager.get_prompt_names()
        self.system_prompt_selector.addItems(prompt_names)
        active_prompt = self.prompt_manager.get_active_prompt_name()
        if active_prompt in prompt_names:
            self.system_prompt_selector.setCurrentText(active_prompt)
            self._load_prompt_to_editor(active_prompt) # Load content into editor
        self.system_prompt_selector.blockSignals(False)


    @Slot(str)
    def _handle_system_prompt_selection_change(self, name: str):
        """Handles selection change in the toolbar dropdown."""
        if not self.prompt_manager or not name: return

        print(f"System prompt selection changed to: {name}")
        if self.prompt_manager.set_active_prompt(name):
            self._load_prompt_to_editor(name)
            self._rebuild_chain() # Rebuild chain with new active prompt
        else:
            QMessageBox.warning(self, "Prompt Error", f"Failed to set '{name}' as active prompt.")
            # Revert selector to actual active prompt
            self.system_prompt_selector.blockSignals(True)
            self.system_prompt_selector.setCurrentText(self.prompt_manager.get_active_prompt_name())
            self.system_prompt_selector.blockSignals(False)

    def _load_prompt_to_editor(self, name: str):
        """Loads the content of the named prompt into the editor tab."""
        if not self.prompt_manager: return
        content, _ = self.prompt_manager.get_prompt(name)
        if content is not None:
             self.prompt_name_input.setText(name)
             self.system_input.setPlainText(content)
             # Maybe switch to the tab?
             # self.input_tabs.setCurrentWidget(self.input_tabs.findChild(QWidget, "System Prompt")) # How to get tab by name?
             # Find index by tab text instead
             for i in range(self.input_tabs.count()):
                  if self.input_tabs.tabText(i) == "System Prompt":
                       # self.input_tabs.setCurrentIndex(i) # Don't force tab switch, just load content
                       break


    @Slot()
    def _handle_save_prompt(self):
        """Saves the prompt currently in the editor tab."""
        if not self.prompt_manager: return
        name = self.prompt_name_input.text().strip()
        content = self.system_input.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "Save Prompt", "Please enter a name for the system prompt.")
            return
        if not content:
             QMessageBox.warning(self, "Save Prompt", "System prompt content cannot be empty.")
             return


        if self.prompt_manager.save_prompt(name, content):
            QMessageBox.information(self, "Save Prompt", f"System prompt '{name}' saved successfully.")
            original_active = self.prompt_manager.get_active_prompt_name()
            self._update_system_prompt_selector() # Refresh dropdown
            # If the saved prompt was already active or is newly active, rebuild chain
            new_active = self.prompt_manager.get_active_prompt_name()
            if name == new_active or name == original_active:
                 # Ensure the selector reflects the actual active prompt after potential save/update
                 self.system_prompt_selector.setCurrentText(new_active)
                 self._rebuild_chain()

        else:
            QMessageBox.warning(self, "Save Prompt", f"Failed to save system prompt '{name}'.")


    @Slot()
    def _handle_delete_prompt(self):
        """Deletes the prompt named in the editor tab's name field."""
        if not self.prompt_manager: return
        name_to_delete = self.prompt_name_input.text().strip()

        if not name_to_delete:
            QMessageBox.warning(self, "Delete Prompt", "Please enter the name of the prompt to delete.")
            return

        if name_to_delete == self.prompt_manager.DEFAULT_SYSTEM_PROMPT_NAME:
            QMessageBox.warning(self, "Delete Prompt", "Cannot delete the default system prompt.")
            return


        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the system prompt '{name_to_delete}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            was_active = (self.prompt_manager.get_active_prompt_name() == name_to_delete)
            if self.prompt_manager.delete_prompt(name_to_delete):
                QMessageBox.information(self, "Delete Prompt", f"System prompt '{name_to_delete}' deleted.")
                self._update_system_prompt_selector() # Refresh dropdown
                # Clear editor fields if the deleted prompt was displayed
                if self.prompt_name_input.text() == name_to_delete: # Check if it was the one displayed
                    self.prompt_name_input.clear()
                    self.system_input.clear()
                # If the deleted prompt was active, the manager resets to default,
                # selector is updated, so rebuild chain
                if was_active:
                     self._rebuild_chain()

            else:
                 QMessageBox.warning(self, "Delete Prompt", f"Failed to delete prompt '{name_to_delete}'. It might not exist.")

    # --- UI Updates & Display Logic ---

    def _update_displays(self):
        """Updates the main story display and potentially other related UI."""
        self._update_story_display()
        # Update edit input only if the blue text actually changed programmatically
        # (e.g. after LLM response or discard). Avoids loop with live editing.
        if self.edit_input.toPlainText() != self.current_narrative:
            self.edit_input.setPlainText(self.current_narrative)


    def _update_story_display(self):
        """Updates the main read-only story display with canon and blue text."""
        # Combine canon parts
        canon_html = "<br><br>".join(self.canon_validated).replace("\n", "<br>")

        # Format blue text
        blue_text_html = ""
        if self.current_narrative:
            # Use HTML for blue color - need to escape HTML chars in narrative itself?
            # For basic use, often okay, but robust solution would escape.
            # Let's use a standard blue hex code. CornflowerBlue was used in label.
            blue_text_html = f"<font color='#6495ED'>{self.current_narrative.replace("\n", "<br>")}</font>"


        # Combine and set
        separator = "<br><br>" if self.canon_validated and self.current_narrative else ""
        full_html = f"{canon_html}{separator}{blue_text_html}"

        self.story_display.setHtml(full_html)
        # Scroll to the end to show the latest content
        self.story_display.moveCursor(QTextCursor.End)


    def _update_monitors(self):
        """Updates the monitor tabs (Context, Thinking, Conversation)."""
        # Context Monitor
        if self.memory:
            context_text = []
            messages = self.memory.messages # Get current active messages (proposal or committed)
            token_count = self.memory.get_token_count(messages)
            context_text.append(f"--- Context History (Tokens: {token_count} / {self.memory.max_tokens}) ---")
            for msg in messages:
                role = type(msg).__name__.replace("Message", "") # Human, AI
                content_preview = msg.content[:150].replace("\n", " ") + ("..." if len(msg.content) > 150 else "")
                context_text.append(f"[{role}]: {content_preview}")
            self.context_display.setPlainText("\n".join(context_text))
            self.context_display.moveCursor(QTextCursor.End)
        else:
            self.context_display.setPlainText("Memory not initialized.")

        # Thinking display is updated directly after LLM response
        # Conversation log is updated directly via _log_conversation


    def _log_conversation(self, role: str, text: str):
        """Appends a message to the conversation log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp} {role}]:\n{text}\n{'-'*20}"
        self.conversation_log.append(log_entry)
        self.conversation_log.moveCursor(QTextCursor.End)

    @Slot(str, object) # Add type hint for payload if possible
    @Slot(str, object, object)
    def _log_api_event(self, run_id: str, payload: dict, *args):
        """Logs an event from the API Monitor callback to the UI."""
        # This slot receives signals emitted by the callback handler
        log_json = json.dumps(payload, indent=2)
        self.api_monitor_display.append(f"{'-'*10} {payload.get('type', 'Event')} ({payload.get('timestamp')}) {'-'*10}\n{log_json}\n")
        self.api_monitor_display.moveCursor(QTextCursor.End)


    def _log_api_event_raw(self, event_type: str, data: dict):
         """Logs a simple event directly to the API monitor (fallback)."""
         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
         log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "data": data
         }
         log_json = json.dumps(log_entry, indent=2)
         self.api_monitor_display.append(f"{'-'*10} {event_type} ({timestamp}) {'-'*10}\n{log_json}\n")
         self.api_monitor_display.moveCursor(QTextCursor.End)


    def _update_window_title(self):
        """Updates the main window title based on file path."""
        base_title = APP_NAME
        file_name = os.path.basename(self.current_file_path) if self.current_file_path else "New Story"
        dirty_indicator = "*" if self._is_dirty else ""
        self.setWindowTitle(f"{base_title} - {file_name}{dirty_indicator}")

    def _update_status_bar(self):
        """Updates the status bar message."""
        canon_len = len(self.canon_validated)
        proposal_len = len(self.current_narrative.split()) if self.current_narrative else 0
        memory_tokens = self.memory.get_token_count() if self.memory else 0
        memory_msg_count = len(self.memory.messages) if self.memory else 0

        status = f"Canon Chunks: {canon_len} | Proposal Words: {proposal_len} | History: {memory_msg_count} msgs / {memory_tokens} tokens"
        if self._is_dirty:
            status += " [Uncommitted Proposal]"

        self.statusBar().showMessage(status)


    # --- UI Customization ---

    @Slot(int)
    def _update_font_size(self, size: int):
        """Applies the selected font size to relevant text areas."""
        font = self.font() # Get default app font
        font.setPointSize(size)
        widgets_to_update = [
            self.story_display, self.context_display, self.thinking_display,
            self.conversation_log, self.api_monitor_display, self.edit_input,
            self.continue_input, self.rewrite_input, self.system_input
        ]
        for widget in widgets_to_update:
            widget.setFont(font)
        self.settings.setValue("fontSize", size) # Save setting


    @Slot(bool)
    def _toggle_theme(self, checked: bool):
        """Toggles between light and dark themes."""
        if checked: # Dark theme
             QApplication.setStyle(QStyleFactory.create("Fusion"))
             # Basic dark palette using QPalette
             dark_palette = QApplication.palette() # Start from current Fusion palette
             dark_color = QColor(53, 53, 53)
             disabled_color = QColor(127, 127, 127)
             text_color = QColor(255, 255, 255)
             highlight_color = QColor(42, 130, 218) # Standard highlight blue

             dark_palette.setColor(dark_palette.ColorRole.Window, dark_color)
             dark_palette.setColor(dark_palette.ColorRole.WindowText, text_color)
             dark_palette.setColor(dark_palette.ColorRole.Base, QColor(35, 35, 35)) # Slightly darker base for inputs
             dark_palette.setColor(dark_palette.ColorRole.AlternateBase, dark_color)
             dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, text_color)
             dark_palette.setColor(dark_palette.ColorRole.ToolTipText, dark_color)
             dark_palette.setColor(dark_palette.ColorRole.Text, text_color)
             dark_palette.setColor(dark_palette.ColorRole.Disabled, dark_palette.ColorRole.Text, disabled_color)
             dark_palette.setColor(dark_palette.ColorRole.Button, dark_color)
             dark_palette.setColor(dark_palette.ColorRole.ButtonText, text_color)
             dark_palette.setColor(dark_palette.ColorRole.Disabled, dark_palette.ColorRole.ButtonText, disabled_color)
             dark_palette.setColor(dark_palette.ColorRole.BrightText, Qt.red)
             dark_palette.setColor(dark_palette.ColorRole.Link, QColor(42, 130, 218))
             dark_palette.setColor(dark_palette.ColorRole.Highlight, highlight_color)
             dark_palette.setColor(dark_palette.ColorRole.HighlightedText, Qt.black)
             dark_palette.setColor(dark_palette.ColorRole.Disabled, dark_palette.ColorRole.HighlightedText, disabled_color)

             QApplication.setPalette(dark_palette)
             self.settings.setValue("theme", "dark")
             self.theme_button.setText("Light Theme")
        else: # Light theme (default)
             QApplication.setPalette(QApplication.style().standardPalette()) # Reset to default style palette
             # Or force a specific light style like 'Windows' or 'macOS' if preferred
             # QApplication.setStyle(QStyleFactory.create("Windows"))
             self.settings.setValue("theme", "light")
             self.theme_button.setText("Dark Theme")


    # --- Settings Persistence ---
    def _load_settings(self):
        """Loads UI settings like theme and font size."""
        font_size = self.settings.value("fontSize", DEFAULT_FONT_SIZE, type=int)
        self.font_size_spinbox.setValue(font_size)
        self._update_font_size(font_size) # Apply loaded font size

        theme = self.settings.value("theme", "light", type=str)
        if theme == "dark":
            self.theme_button.setChecked(True) # This will trigger _toggle_theme
        else:
            self.theme_button.setChecked(False)


    def closeEvent(self, event):
        """Handle window close event, prompt to save if needed."""
        if self._prompt_save_if_dirty():
             self.settings.setValue("geometry", self.saveGeometry()) # Save window size/pos
             event.accept() # Proceed with closing
        else:
             event.ignore() # User cancelled, do not close


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(SETTINGS_APP)
    app.setOrganizationName(SETTINGS_ORG)

    # Force Fusion style for more consistent look across platforms initially
    app.setStyle(QStyleFactory.create("Fusion"))

    # Create and show the main window
    window = MainWindow()

    # Restore window geometry
    geometry = window.settings.value("geometry")
    if geometry:
         window.restoreGeometry(geometry)
    else:
         window.resize(DEFAULT_WINDOW_SIZE) # Set default size if no saved geometry

    window.show()

    sys.exit(app.exec())