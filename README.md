# Langchain-Groq Narrative Collaboration App (TINS Edition - Development Process)

**Repository:** [https://github.com/fernicar/langchain-groq-TINS-Edition](https://github.com/fernicar/langchain-groq-TINS-Edition)

This repository documents the creation of the "Narrative Collaboration System," a desktop application built using Python, PySide6, Langchain, and Groq. More importantly, it serves as a case study for developing software using the **"There Is No Source" (TINS)** methodology ([thereisnosource.com](https://thereisnosource.com/)) facilitated by an AI code generation assistant.

The primary goal of this README is to provide insight into the development process for mentor review and to potentially inspire peers interested in AI-assisted development and the TINS paradigm.

## The Initial Challenge: Building from a TINS Specification

The project started with a detailed specification for the application, formatted according to the TINS guidelines. Instead of writing code manually, the core task posed to the AI assistant was:

> "Understand what [TINS](https://github.com/ScuffedEpoch/TINS) is by reading: [`TINS.md`](https://github.com/ScuffedEpoch/TINS/blob/main/README.md), [`specification.md`](https://github.com/ScuffedEpoch/TINS/blob/main/docs/specification.md), [`developer-guide.md`](https://github.com/ScuffedEpoch/TINS/blob/main/docs/developer-guide.md), [`best-practices.md`](https://github.com/ScuffedEpoch/TINS/blob/main/docs/best-practices.md), and the example `todo-app-example.md`
>
> Make the source code of my Desktop App following the [`README.md`](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/tins_edition/README.md)"

*(This initial prompt is captured below for context, with the 7 attached .md files above it)*

![SCREENSHOT_PROMPT](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/images/llm_request.png)

This involved the AI parsing not only the application's specific [`README.md`](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/tins_edition/README.md) (which defined its functionality, UI using Mermaid diagrams, technical architecture, data models, etc.) but also understanding the underlying principles of [TINS](https://github.com/ScuffedEpoch/TINS) itself from the provided documentation files from the repository and one example. The specification was non-trivial, outlining a multi-pane UI, interaction logic, state management using a custom dual-state memory, system prompt handling, and integration with the Groq API via Langchain.

## The AI's Response: Code Generation Based on TINS

The AI assistant processed the comprehensive TINS specification and generated the initial Python codebase. Key aspects of the AI's contribution included:

1.  **Understanding the Specification:** Acknowledging the complexity and the TINS methodology.
2.  **Code Structure:** Generating multiple Python files organized by function:
    *   `main.py`: Main application window, UI logic (PySide6), application state orchestration.
    *   `memory.py`: The custom `TokenWindowDualStateMemory` class for Langchain history management (proposal/commit logic).
    *   `prompts.py`: The `SystemPromptManager` for handling customizable system prompts via JSON.
    *   `callbacks.py`: The `APIMonitorCallback` for logging raw Groq API interactions using Langchain callbacks.
    *   `utils.py`: Helper functions for tasks like LLM response parsing (`<think>` tags) and token counting.
    *   `requirements.txt`: A list of necessary Python dependencies.
3.  **Technology Implementation:** Utilizing the specified stack: PySide6 for the GUI, Langchain for LLM interaction structure, `langchain-groq` for the Groq API connection, `python-dotenv` for API key management.
4.  **Acknowledging Limitations:** Highlighting potential issues, most notably that the generated code performed LLM calls *synchronously*, which would freeze the UI, and recommending threading for a production-ready application.
5.  **Setup Instructions:** Providing initial guidance on setting up the environment (creating `.env` file, installing requirements, running the main script).

*(The key setup instructions provided by the AI are summarized below)*

![INSTRUCTIONS](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/images/llm_instructions.png)

## Bridging the Gap: From Generation to Functionality (User Fixes & Adjustments assisted by VS Code and Augment)

While the AI provided a significant foundation, transforming the generated code into a runnable and repository-ready state required manual intervention and debugging using VS Code. Key steps taken included:

*   **Version Control:** Performed the initial Git commit (`git init`, `git add .`, `git commit -m "initial commit of AI-generated code and basic fixes"`) to establish the project baseline in this repository.
*   **Dependency Resolution:** Identified and fixed missing import statements within the Python files (e.g., ensuring specific `from datetime import datetime` components were explicitly imported where needed) based on errors encountered during initial execution attempts.
*   **UI Refinements:** Renamed UI Tab labels for better clarity and conciseness (e.g., changing `Save Blue & Continue` to `Commit Blue && Continue`). Adjusted other UI element labels as needed during initial testing.
*   **Licensing:** Added an `MIT License` file (`LICENSE`) to define clear usage rights for the code.
*   **Documentation & Visualization:** Prepared this `README.md` file, outlining the process and adding placeholders for relevant screenshots. Captured images of the development process and the application itself (to be added below).
![APP_SCREENSHOT](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/images/app_capture.png)

## TINS in Practice - A Reflection

This project serves as a practical example of the [TINS](https://github.com/ScuffedEpoch/TINS) methodology in action. The detailed [`README.md`](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/tins_edition/README.md) acted as the "source," which the AI interpreted to generate the implementation.

**Key Takeaways:**

*   **Acceleration:** Using a detailed TINS specification allowed the AI to generate a complex application structure rapidly, saving significant initial development time.
*   **Human-in-the-Loop:** Implement Test-Driven Development (TDD) TINS, AI code generation, even from detailed specs, often requires human oversight for debugging, refinement, platform-specific nuances (like GUI event loops vs. blocking calls), and ensuring adherence to best practices.
*   **Specification is Key:** The quality of the generated code is highly dependent on the clarity, detail, and consistency of the TINS `README.md`. Ambiguities in the spec can lead to unexpected or incorrect implementations.
*   **Potential:** The TINS approach, combined with capable AI assistants, shows promise for streamlining software development, especially for well-defined applications. It shifts focus from writing boilerplate code to designing and specifying robustly.

This repository demonstrates one workflow for leveraging TINS and AI. We encourage peers and mentors to review the code and the process, hopefully sparking further interest and experimentation with this development paradigm.

## Running the Application

1.  **Clone:** `git clone https://github.com/fernicar/langchain-groq-TINS-Edition.git`
2.  **Navigate:** `cd langchain-groq-TINS-Edition`
3.  **API Key:** Create a `.env` file in the root directory with your Groq API key:
    ```
    GROQ_API_KEY=gsk_YourActualGroqApiKeyHere
    ```
4.  **Install:** `pip install -r requirements.txt`
5.  **Run:** `python main.py`

*(Refer to the `[INSTRUCTIONS_SCREENSHOT]` section above for potentially more detailed setup steps provided initially by the AI).*

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/fernicar/langchain-groq-TINS-Edition/blob/main/LICENSE) file for details.

## Acknowledgments

*   Special thanks to ScuffedEpoch for the TINS methodology and the initial example.
*   Thanks to the free tier AI assistant for its initial contribution to the project.
*   Gratitude to the Groq team for their API and support.
*   Thanks to the Langchain and PySide6 communities for their respective libraries and documentation.
*   Augment extension for VS Code
*   Tested LLM Gemini2.5pro (free tier beta testing) from Google AI Studio