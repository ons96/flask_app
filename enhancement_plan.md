# Flask App Enhancement Plan (Revised)

**Overall Goal:** Modify `app.py` to improve LLM provider selection logic using scraped performance data, add flexible web search capabilities, and conceptually outline future Blackberry-specific features.

---

## Phase 1: Core `app.py` Modifications (Simplified)

### 1. Refined Provider Selection & Sorting Logic (Using Scraped Data Only)

*   **Goal:** Implement the strict `Cerebras > Groq > g4f` logic and sort g4f providers based *only* on the scraped performance data.
*   **Implementation:**
    *   Refactor the provider selection block (currently lines 365-391 in `flask_app/app.py`).
    *   **Step 1: Try Direct Cerebras:** If `selected_model` is suitable for Cerebras and `CEREBRAS_API_KEY` exists, attempt `ChatCompletion.create` targeting Cerebras *first*.
    *   **Step 2: Try Direct Groq:** If Cerebras wasn't tried or failed, check if `selected_model` is suitable for Groq and `GROQ_API_KEY` exists. Attempt `ChatCompletion.create` targeting Groq next.
    *   **Step 3: Try g4f Providers (Sorted by Scraped Data):** If neither direct attempt succeeded:
        *   Get the list of default g4f providers for the model (`model_provider_info.get(...)`).
        *   Define a function `get_scraped_performance_metric(provider_class)`:
            *   Checks the `PROVIDER_PERFORMANCE_CACHE` (loaded from `provider_performance.csv` at startup) for a matching provider/model entry.
            *   Returns the relevant metric (e.g., `response_time_s`) if found, otherwise returns a default high cost (`float('inf')`).
        *   Sort the default g4f providers using `get_scraped_performance_metric` as the key.
        *   Iterate through the *sorted* g4f providers:
            *   **Internal Priority:** Check if the current provider is Groq or Cerebras. If it is, and the corresponding API key exists, and it wasn't *already tried* directly, attempt it now with the key.
            *   **Standard Attempt:** If not an internal priority case or if the prioritized attempt failed, attempt the provider normally via `ChatCompletion.create`.
            *   Break the loop on the first successful attempt.

### 2. Web Search Integration

*   **Goal:** Add user control (Off/On/Smart) and integrate search results into the LLM prompt when active.
*   **UI Changes (HTML in `index` return string):**
    *   Add radio buttons or a dropdown for `web_search_mode` (Off/On/Smart).
*   **Backend Logic (`index` route POST handler):**
    *   Get `web_search_mode`.
    *   If 'On', call `perform_web_search`.
    *   If 'Smart', trigger search based on keywords (e.g., "latest", "recent", "today", "current news", "what happened", "who is"). *Consider adding LLM uncertainty triggers later if needed.*
    *   If search performed, prepend results to the LLM prompt.
*   **Web Search Tool:** Start with the existing `duckduckgo-search`. Explore alternatives later if performance/quality is insufficient.

### Phase 1 Implementation Flow (Simplified)

```mermaid
graph TD
    A[Start Request Handling] --> B{Get User Input/Model/Search Mode};
    B --> C{Web Search Active?};
    C -- Yes --> D[Perform Web Search];
    C -- No --> E[Prepare Base Prompt/History];
    D --> F[Prepend Search Results to Prompt];
    F --> E;
    E --> G{Try Direct Cerebras?};
    G -- Yes --> H[Attempt Cerebras API];
    G -- No --> I{Try Direct Groq?};
    H --> J{Success?};
    J -- Yes --> Z[Format Response];
    J -- No --> I; // Proceed to Groq check
    I -- Yes --> L[Attempt Groq API];
    I -- No --> M[Get g4f Providers];
    L --> N{Success?};
    N -- Yes --> Z;
    N -- No --> M; // Proceed to g4f providers
    M --> P[Sort Providers by Scraped Perf Data];
    P --> Q[Loop Through Sorted Providers];
    Q --> R{Internal Priority Check (Groq/Cerebras)?};
    R -- Yes --> S[Attempt Prioritized g4f];
    R -- No --> T[Attempt Standard g4f];
    S --> U{Success?};
    T --> U;
    U -- Yes --> Z; // Format Response
    U -- No --> X{More Providers?};
    X -- Yes --> Q;
    X -- No --> Y[Handle All Failed];
    Y --> Z;
    Z --> End[Return Result to User];

    subgraph Performance Data Sources
        PD1[Scraped Data Cache] --> P;
    end
```

---

## Phase 2: Conceptual Planning for Blackberry Features (Separate Projects)

*(This section outlines future work and is not part of the immediate implementation)*

### 1. Webpage Rendering Service

*   **Concept:** Separate service using headless browser (Playwright) to render, simplify, and send basic HTML/image to Blackberry.
*   **Challenges:** Dynamic content, logins, interaction complexity, performance.
*   **Diagram:**
    ```mermaid
    sequenceDiagram
        participant BB as Blackberry Client
        participant RS as Rendering Service
        participant TS as Target Website
        BB->>RS: Request URL
        RS->>TS: Load URL (via Headless Browser)
        TS-->>RS: Rendered Page Content
        RS->>RS: Process/Simplify Content
        RS-->>BB: Simplified HTML/Image
    ```

### 2. Remote Control Service

*   **Concept:** Separate service listening for Blackberry commands, using UI automation (`pyautogui`, `pywinauto`) to interact with tools (VS Code/Terminal) on the laptop, capturing output.
*   **Challenges:** **Security (major concern)**, reliability of UI automation, error handling, focus stealing, locked-screen interaction.
*   **Tool Flexibility:** Can be designed to interface with different tools (Roo, OpenHands, etc.).
*   **Diagram:**
    ```mermaid
    sequenceDiagram
        participant BB as Blackberry Client
        participant CS as Control Service (Laptop)
        participant Tool as Target Tool (VS Code/Terminal)
        BB->>CS: Send Command/Prompt
        CS->>Tool: Automate UI (Input Prompt)
        Tool->>Tool: Process Command
        Tool-->>CS: Capture Output (via UI/Logs)
        CS-->>BB: Send Result/Output
    ```

---