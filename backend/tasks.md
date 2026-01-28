# Tasks: AI-Powered Vision-Based Test Automation

> Generated from plan.md for Spec Kit integration

---

## PHASE 1: PROJECT FOUNDATION

### Task 1.1: Project Structure Setup
- **Status:** pending
- **Priority:** critical
- **Dependencies:** none
- **Files:** Directory structure, `__init__.py` files
- **Estimated Time:** 15 minutes

**Description:**
Create all project directories and Python package initialization files per spec.md section 10.

**Actions:**
- [ ] Create directory structure:
  ```
  TA_AI_Project/
  ├── agents/
  ├── workflows/
  ├── models/
  ├── utils/
  ├── templates/
  ├── Jira_Tickets/
  ├── Reports/
  ├── Videos/
  ├── Generated_Scripts/
  └── Logs/
  ```
- [ ] Create `__init__.py` in: agents/, workflows/, models/, utils/
- [ ] Verify Python package structure is valid

**Acceptance Criteria:**
- ✅ All directories exist
- ✅ Python package structure is valid
- ✅ No import errors when importing empty packages

**References:**
- spec.md section 10
- constitution.md section 1

---

### Task 1.2: Dependencies Configuration
- **Status:** pending
- **Priority:** critical
- **Dependencies:** none
- **Files:** `requirements.txt`
- **Estimated Time:** 10 minutes

**Description:**
Create requirements.txt with all project dependencies and pinned versions.

**Actions:**
- [ ] Create requirements.txt with:
  - langchain==0.1.0
  - langgraph==0.1.0
  - playwright==1.40.0
  - openai==1.0.0
  - pydantic==2.0.0
  - pyyaml==6.0
  - pillow==10.0.0
  - jinja2==3.1.0
  - python-dotenv==1.0.0
- [ ] Test: `pip install -r requirements.txt` succeeds
- [ ] Document Playwright browser installation: `playwright install msedge`

**Acceptance Criteria:**
- ✅ File contains all dependencies with pinned versions
- ✅ pip install succeeds without errors
- ✅ Playwright installation command documented

**References:**
- spec.md section 17
- constitution.md section 15

---

### Task 1.3: State Schema Implementation
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.2
- **Files:** `models/state.py`
- **Estimated Time:** 30 minutes

**Description:**
Implement TestAutomationState TypedDict for LangGraph state management.

**Actions:**
- [ ] Import typing modules: TypedDict, List, Dict, Optional
- [ ] Define TestAutomationState with all fields from spec.md section 5:
  - Configuration: config, ticket_number
  - Jira data: jira_data, module, test_title, description, steps, acceptance_criteria
  - Execution: execution_results, screenshots, times, overall_status, video_path
  - Context: browser_context, current_step
  - History: conversation_history, errors
  - Output: report_path, script_path, report_generation_status
- [ ] Add type hints for all fields
- [ ] Add module docstring explaining state structure

**Acceptance Criteria:**
- ✅ TypedDict matches spec.md section 5 exactly
- ✅ All fields have proper type annotations
- ✅ No type errors when importing
- ✅ Docstrings are clear

**References:**
- spec.md section 5
- constitution.md sections 2, 3

---

### Task 1.4: Configuration Loader Utility
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.2, Task 1.3
- **Files:** `utils/config_loader.py`
- **Estimated Time:** 45 minutes

**Description:**
Implement YAML configuration loader with validation and path resolution.

**Actions:**
- [ ] Import yaml, os, pathlib, typing
- [ ] Implement `load_config(config_path: str) -> Dict`:
  - Read YAML file
  - Validate required fields: base_folder, web_url, browser, login, wait_times, folders, azure_openai, execution
  - Resolve relative folder paths to absolute paths
  - Return validated config dictionary
- [ ] Implement error handling:
  - FileNotFoundError for missing config
  - yaml.YAMLError for invalid YAML
  - KeyError for missing required fields
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Successfully loads valid config YAML
- ✅ Raises clear errors for missing required fields
- ✅ Converts relative paths to absolute paths
- ✅ Returns validated dictionary
- ✅ Handles errors gracefully

**References:**
- spec.md section 3
- constitution.md sections 5, 6

---

### Task 1.5: Logger Utility
- **Status:** pending
- **Priority:** high
- **Dependencies:** Task 1.4
- **Files:** `utils/logger.py`
- **Estimated Time:** 30 minutes

**Description:**
Implement logging utility with file and console handlers, sensitive data masking.

**Actions:**
- [ ] Import logging, datetime, pathlib, re
- [ ] Implement `setup_logger(log_folder: str, ticket_id: str) -> logging.Logger`:
  - Create logger with name "TA_AI_Project"
  - Add file handler: `{log_folder}/{ticket_id}_{timestamp}.log`
  - Add console handler
  - Set format: `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`
  - Set level: INFO for console, DEBUG for file
- [ ] Implement `mask_sensitive_data(message: str) -> str`:
  - Mask patterns: passwords, API keys, credentials
  - Replace with "***MASKED***"
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Creates log files in configured Logs folder
- ✅ Logs to both file and console
- ✅ Proper timestamp formatting
- ✅ Masks sensitive data automatically
- ✅ Supports different log levels

**References:**
- spec.md section 4
- constitution.md sections 10, 13

---

### Task 1.6: Vision Helper Utility
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.2, Task 1.4, Task 1.5
- **Files:** `utils/vision_helper.py`
- **Estimated Time:** 1 hour

**Description:**
Implement Azure OpenAI GPT-4o vision API wrapper with retry logic.

**Actions:**
- [ ] Import openai, base64, json, time, logging
- [ ] Implement `AzureVisionClient` class:
  - `__init__(self, config: Dict, logger: logging.Logger)`
  - Initialize AzureOpenAI client from config
- [ ] Implement `call_vision(self, screenshot: bytes, prompt: str) -> Dict`:
  - Encode screenshot to base64
  - Build messages with image_url format
  - Call GPT-4o with temperature=0.1, max_tokens=500
  - Parse JSON response
  - Return parsed dictionary
- [ ] Implement retry logic for API errors:
  - RateLimitError: sleep 60s, retry
  - APIError with timeout: retry
  - Other errors: log and raise
  - Max 3 retry attempts
- [ ] Add error handling and logging
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Successfully initializes Azure OpenAI client
- ✅ Encodes screenshots to base64 correctly
- ✅ Sends proper message format to GPT-4o
- ✅ Parses JSON responses successfully
- ✅ Handles API errors with retry logic
- ✅ Logs API calls (truncated for large payloads)

**References:**
- spec.md sections 7, 9
- constitution.md sections 7, 11

---

## PHASE 2: AGENT IMPLEMENTATION

### Task 2.1: Jira Parser Agent
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.3, Task 1.5
- **Files:** `agents/jira_parser_agent.py`
- **Estimated Time:** 45 minutes

**Description:**
Implement agent to parse Jira ticket files and extract structured test data.

**Actions:**
- [ ] Import re, pathlib, logging, models.state
- [ ] Implement `jira_parser_agent(state: TestAutomationState) -> TestAutomationState`:
  - Get ticket file path: `{config.folders.jira}/{state.ticket_number}.txt`
  - Read file content
  - Extract ticket ID from title using regex: `\[([A-Z]+-\d+)\]`
  - Extract module from "Component/s:" line
  - Extract title (text after ticket ID)
  - Parse "Steps to Reproduce:" section into numbered list
  - Extract "Acceptance Criteria:" text
  - Build jira_data dictionary
  - Update state with extracted data
  - Return updated state
- [ ] Implement error handling:
  - FileNotFoundError: log and raise
  - Parsing errors: log and use defaults
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Successfully parses RBPLCD-8835.txt and RBPLCD-8862.txt
- ✅ Extracts all required fields correctly
- ✅ Returns proper jira_data structure
- ✅ Handles missing fields gracefully
- ✅ Logs all parsing actions
- ✅ Updates state without modifying other fields

**References:**
- spec.md section 4 (Sub-Agent 1)
- constitution.md section 3

---

### Task 2.2: Vision Executor Agent - Browser Setup & Login
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.6, Task 1.5, Task 1.2
- **Files:** `agents/vision_executor_agent.py` (Part A)
- **Estimated Time:** 1.5 hours

**Description:**
Implement browser initialization and auto-login functions using GPT-4o vision.

**Actions:**
- [ ] Import playwright, logging, time, utils.vision_helper, models.state
- [ ] Implement `initialize_browser(config: Dict, logger: logging.Logger) -> tuple[Browser, BrowserContext, Page]`:
  - Launch Playwright with Edge browser
  - Set viewport to 1920x1080
  - Set record_video options
  - Navigate to config.web_url
  - Return browser, context, page objects
- [ ] Implement `auto_login(page: Page, config: Dict, vision_client: AzureVisionClient, logger: logging.Logger) -> bool`:
  - Take screenshot of login page
  - Build prompt for GPT-4o to find username, password, login button
  - Call vision API
  - Parse coordinates from response
  - Click username field, type username
  - Click password field, type password
  - Click login button
  - Wait after login (config.wait_times.after_login)
  - Take screenshot to verify success
  - Return success boolean
- [ ] Implement error handling for browser timeouts, crashes
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Browser launches successfully with Edge
- ✅ Navigates to correct URL
- ✅ Video recording starts
- ✅ GPT-4o correctly identifies login form elements
- ✅ Login executes successfully
- ✅ Waits appropriately after actions
- ✅ Handles browser timeouts and crashes

**References:**
- spec.md section 4 (Sub-Agent 2, Steps 1-2)
- constitution.md sections 8, 9

---

### Task 2.3: Vision Executor Agent - Step Execution
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 2.2, Task 1.6
- **Files:** `agents/vision_executor_agent.py` (Part B)
- **Estimated Time:** 2 hours

**Description:**
Implement individual test step execution with vision-guided actions.

**Actions:**
- [ ] Implement `execute_test_step(step: Dict, page: Page, vision_client: AzureVisionClient, config: Dict, state: TestAutomationState, logger: logging.Logger) -> Dict`:
  - Take screenshot before action
  - Build context-aware prompt:
    - Current page/module
    - Previous action
    - Current task (step.text)
    - Request JSON format with coordinates, action_type, value, confidence
  - Call vision API
  - Parse JSON response
  - Execute action based on action_type:
    - "click": page.mouse.click(x, y)
    - "type": click + page.keyboard.type(value)
    - "dropdown": click + select option
  - Wait after action (config.wait_times)
  - Take screenshot after action
  - Verify action succeeded (basic comparison)
  - Return execution result dictionary with:
    - step_num, step_text, status, coordinates, confidence, execution_time, screenshots, retries
- [ ] Add error handling for action failures
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Builds proper context-aware prompts
- ✅ GPT-4o returns structured JSON with coordinates
- ✅ Executes click actions at correct coordinates
- ✅ Executes type actions with proper text input
- ✅ Handles dropdown selections
- ✅ Captures before/after screenshots
- ✅ Verifies action success
- ✅ Returns complete execution result

**References:**
- spec.md section 4 (Sub-Agent 2, Step 3)
- constitution.md sections 7, 8

---

### Task 2.4: Vision Executor Agent - Retry Logic
- **Status:** pending
- **Priority:** high
- **Dependencies:** Task 2.3
- **Files:** `agents/vision_executor_agent.py` (Part C)
- **Estimated Time:** 1 hour

**Description:**
Implement 3-attempt retry strategy with prompt refinement.

**Actions:**
- [ ] Implement `execute_with_retry(step: Dict, page: Page, vision_client: AzureVisionClient, config: Dict, state: TestAutomationState, logger: logging.Logger) -> Dict`:
  - Attempt 1: Call execute_test_step with standard prompt
  - Check confidence >= 0.85 and success
  - If failed, Attempt 2:
    - Enhance prompt with more specific instructions
    - Crop screenshot to relevant region
    - Retry execution
  - If still failed, Attempt 3:
    - Use most detailed prompt
    - Add specific constraints
    - Retry execution
  - Log all retry attempts with reasons
  - After max retries, mark FAILED and return result
  - Return final result with retry_count
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Retries up to 3 times for low confidence (<0.85)
- ✅ Refines prompts on each retry
- ✅ Crops screenshots for attempt 2+
- ✅ Logs all retry attempts
- ✅ Returns final result with retry count
- ✅ Continues to next step even if current fails

**References:**
- spec.md section 8
- constitution.md section 9

---

### Task 2.5: Vision Executor Agent - Main Orchestration
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 2.4, Task 2.3, Task 2.2
- **Files:** `agents/vision_executor_agent.py` (Part D)
- **Estimated Time:** 1.5 hours

**Description:**
Implement main vision executor agent function orchestrating entire test execution.

**Actions:**
- [ ] Implement `vision_executor_agent(state: TestAutomationState) -> TestAutomationState`:
  - Initialize vision client from state.config
  - Initialize browser
  - Execute auto_login
  - Record execution_start_time
  - Initialize execution_results list
  - For each step in state.jira_data.steps:
    - Call execute_with_retry
    - Append result to execution_results
    - Save screenshots
    - Update current_step in state
  - Record execution_end_time
  - Calculate total_execution_time
  - Determine overall_status (PASSED if all passed, else FAILED)
  - Save video recording
  - Update state with all execution data
  - Close browser
  - Return updated state
- [ ] Implement browser crash handling:
  - Catch playwright errors
  - Restart browser
  - Resume from last successful step
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Executes all test steps in sequence
- ✅ Tracks execution time accurately
- ✅ Determines correct overall status
- ✅ Saves video recording to Videos folder
- ✅ Updates state with complete execution data
- ✅ Handles browser crashes gracefully
- ✅ Closes browser properly on completion or error

**References:**
- spec.md section 4 (Sub-Agent 2)
- constitution.md sections 3, 5, 8

---

### Task 2.6: Report Generator Agent - HTML Report
- **Status:** pending
- **Priority:** high
- **Dependencies:** Task 1.3, Task 1.5, Task 1.2
- **Files:** `agents/report_generator_agent.py` (Part A)
- **Estimated Time:** 1.5 hours

**Description:**
Implement HTML report generation using Jinja2 templates.

**Actions:**
- [ ] Import jinja2, base64, pathlib, datetime, logging, models.state
- [ ] Implement `generate_html_report(state: TestAutomationState, template_path: str, logger: logging.Logger) -> str`:
  - Load Jinja2 template from template_path
  - Prepare context dictionary:
    - ticket_id, module, test_title, description, acceptance_criteria
    - overall_status, execution_start_time, execution_end_time, total_execution_time
    - execution_results with embedded screenshots (base64)
    - video_path, script_path
  - Render template with context
  - Generate filename: `{ticket_id}_report_{YYYYMMDD_HHMMSS}.html`
  - Save to config.folders.reports
  - Return report file path
- [ ] Implement screenshot embedding:
  - Read screenshot files
  - Encode to base64
  - Embed as data URIs
- [ ] Add error handling for template errors, file I/O
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Generates valid HTML file
- ✅ Includes all required sections
- ✅ Embeds screenshots inline
- ✅ Links to video and script files
- ✅ Follows naming convention
- ✅ Renders correctly in browser

**References:**
- spec.md section 4 (Sub-Agent 3), section 11
- constitution.md section 12

---

### Task 2.7: Report Generator Agent - Playwright Script
- **Status:** pending
- **Priority:** medium
- **Dependencies:** Task 2.6
- **Files:** `agents/report_generator_agent.py` (Part B)
- **Estimated Time:** 1 hour

**Description:**
Implement Playwright Python script generation from execution results.

**Actions:**
- [ ] Implement `generate_playwright_script(state: TestAutomationState, logger: logging.Logger) -> str`:
  - Build Python script string with:
    - Imports: playwright, time
    - Browser setup code
    - Navigation to web_url
    - Login sequence with coordinates
    - For each step in execution_results:
      - Add comment with step description
      - Add click/type command with coordinates
      - Add wait command
    - Browser close
  - Format code with proper indentation
  - Generate filename: `{ticket_id}_script_{YYYYMMDD_HHMMSS}.py`
  - Save to config.folders.scripts
  - Return script file path
- [ ] Ensure generated script is executable standalone
- [ ] Add error handling
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Generates valid Python code
- ✅ Includes all executed actions with coordinates
- ✅ Has descriptive comments for each step
- ✅ Script is executable independently
- ✅ Follows naming convention
- ✅ Includes proper imports and setup

**References:**
- spec.md section 4 (Sub-Agent 3), section 11
- constitution.md section 12

---

### Task 2.8: Report Generator Agent - Main Orchestration
- **Status:** pending
- **Priority:** high
- **Dependencies:** Task 2.7, Task 2.6
- **Files:** `agents/report_generator_agent.py` (Part C)
- **Estimated Time:** 30 minutes

**Description:**
Implement main report generator agent function.

**Actions:**
- [ ] Implement `report_generator_agent(state: TestAutomationState) -> TestAutomationState`:
  - Get template path from config
  - Call generate_html_report
  - Call generate_playwright_script
  - Update state with:
    - report_path
    - script_path
    - report_generation_status ("success" or "failed")
  - Handle file write errors
  - Log generation actions
  - Return updated state
- [ ] Add error handling
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Generates both HTML report and Playwright script
- ✅ Updates state with file paths
- ✅ Handles errors gracefully
- ✅ Logs all generation actions
- ✅ Sets correct generation status

**References:**
- spec.md section 4 (Sub-Agent 3)
- constitution.md section 3

---

## PHASE 3: WORKFLOW INTEGRATION

### Task 3.1: Workflow Node Functions
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 1.4, Task 1.3
- **Files:** `workflows/test_workflow.py` (Part A)
- **Estimated Time:** 45 minutes

**Description:**
Implement LangGraph node functions for configuration and input handling.

**Actions:**
- [ ] Import langgraph, logging, models.state, utils.config_loader
- [ ] Implement `load_config_node(state: TestAutomationState) -> TestAutomationState`:
  - Load config from plcdtest_config.yaml
  - Update state["config"] with loaded config
  - Handle errors
  - Return updated state
- [ ] Implement `get_ticket_input_node(state: TestAutomationState) -> TestAutomationState`:
  - Prompt user: "Enter Jira ticket number: "
  - Read input
  - Validate format: TICKET-NUMBER (regex: `^[A-Z]+-\d+$`)
  - Update state["ticket_number"]
  - Handle invalid input (re-prompt)
  - Return updated state
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ load_config_node successfully loads config
- ✅ get_ticket_input_node prompts and validates input
- ✅ Both functions update state correctly
- ✅ Error handling works properly

**References:**
- spec.md section 6
- constitution.md section 4

---

### Task 3.2: LangGraph Workflow Definition
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 3.1, Task 2.1-2.8
- **Files:** `workflows/test_workflow.py` (Part B)
- **Estimated Time:** 1 hour

**Description:**
Create LangGraph StateGraph connecting all agents in workflow.

**Actions:**
- [ ] Import StateGraph, END from langgraph.graph
- [ ] Import all agent functions
- [ ] Implement `create_workflow() -> CompiledGraph`:
  - Create StateGraph instance with TestAutomationState
  - Add nodes:
    - "load_config" → load_config_node
    - "get_ticket_input" → get_ticket_input_node
    - "jira_parser" → jira_parser_agent
    - "vision_executor" → vision_executor_agent
    - "report_generator" → report_generator_agent
  - Define linear flow:
    - set_entry_point("load_config")
    - add_edge("load_config", "get_ticket_input")
    - add_edge("get_ticket_input", "jira_parser")
    - add_edge("jira_parser", "vision_executor")
    - add_edge("vision_executor", "report_generator")
    - add_edge("report_generator", END)
  - Compile and return workflow
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ StateGraph created with correct state type
- ✅ All nodes added successfully
- ✅ Flow matches spec.md section 6 exactly
- ✅ Workflow compiles without errors
- ✅ No circular dependencies
- ✅ State flows through all nodes

**References:**
- spec.md section 6
- constitution.md section 4

---

### Task 3.3: HTML Report Template
- **Status:** pending
- **Priority:** high
- **Dependencies:** none
- **Files:** `templates/report_template.html`
- **Estimated Time:** 1.5 hours

**Description:**
Create Jinja2 HTML template for test execution reports.

**Actions:**
- [ ] Create HTML5 document structure
- [ ] Add sections:
  1. Header with title and status badge (color-coded: green=PASSED, red=FAILED)
  2. Executive Summary:
     - Overall Status
     - Execution Time
     - Test Date
     - Ticket ID
  3. Test Information:
     - Module
     - Description
     - Steps count
     - Acceptance Criteria
  4. Step Results Table:
     - Columns: Step #, Description, Status, Confidence, Execution Time, Screenshot
     - Loop: {% for step in execution_results %}
     - Embed screenshots: <img src="data:image/png;base64,{{ step.screenshot }}">
  5. Media Links:
     - Video download link
     - Script download link
  6. Execution Log:
     - Detailed timestamped log
     - Show errors if any
- [ ] Add embedded CSS styling:
  - Professional appearance
  - Responsive layout
  - Printable format
  - Color-coded status badges
- [ ] Use Jinja2 variables and control structures
- [ ] Test template renders correctly

**Acceptance Criteria:**
- ✅ Valid HTML5 structure
- ✅ All sections present and properly formatted
- ✅ Jinja2 variables and loops work correctly
- ✅ Embedded CSS styling looks professional
- ✅ Screenshots display inline
- ✅ Links to video and script work
- ✅ Renders in all major browsers

**References:**
- spec.md section 11
- constitution.md section 12

---

## PHASE 4: MAIN ENTRY POINT & CONFIGURATION

### Task 4.1: Main Entry Point
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 3.2, Task 1.5
- **Files:** `main.py`
- **Estimated Time:** 30 minutes

**Description:**
Create executable entry point for the test automation tool.

**Actions:**
- [ ] Import sys, logging, workflows.test_workflow, utils.logger, models.state
- [ ] Implement `main()` function:
  - Print welcome banner
  - Initialize logger (use default log folder)
  - Create workflow using create_workflow()
  - Initialize empty state dictionary
  - Invoke workflow with state: `result = app.invoke(state)`
  - Extract final state
  - Print summary to console:
    - Ticket ID
    - Overall Status
    - Execution Time
    - Report Path
    - Video Path
    - Script Path
  - Handle workflow exceptions
  - Return exit code (0=success, 1=failure)
- [ ] Add `if __name__ == "__main__":` block:
  - Call main()
  - Handle KeyboardInterrupt (Ctrl+C)
  - Exit with appropriate code
- [ ] Add type hints and docstrings

**Acceptance Criteria:**
- ✅ Successfully creates and runs workflow
- ✅ Handles errors gracefully
- ✅ Prints clear summary to console
- ✅ Exits cleanly on completion or error
- ✅ Responds to keyboard interrupt

**References:**
- spec.md section 18
- constitution.md section 14

---

### Task 4.2: Sample Configuration File
- **Status:** pending
- **Priority:** critical
- **Dependencies:** none
- **Files:** `plcdtest_config.yaml`
- **Estimated Time:** 20 minutes

**Description:**
Create sample configuration file with all required settings.

**Actions:**
- [ ] Create YAML file with structure:
  ```yaml
  # Base Configuration
  base_folder: "C:/Projects/AI_Chat/PLCD/TA_AI_Project"

  # Web Application
  web_url: "http://fe0vm03313.de.bosch.com/rbplcd_t/client/login"
  browser: "edge"

  # Login Credentials
  login:
    username: "mechanic"
    password: "avalon"

  # Wait Times (milliseconds)
  wait_times:
    after_login: 3000
    after_navigation: 2000
    after_click: 1000
    after_type: 500
    after_dropdown: 1000
    page_load: 5000

  # Folder Paths (relative to base_folder)
  folders:
    jira: "Jira_Tickets"
    reports: "Reports"
    videos: "Videos"
    scripts: "Generated_Scripts"
    logs: "Logs"

  # Azure OpenAI Configuration
  azure_openai:
    api_key: "YOUR_API_KEY_HERE"  # Replace with actual API key
    endpoint: "https://ai2ets.openai.azure.com/"
    api_version: "2024-02-15-preview"
    deployment_gpt4o: "gpt-4o"

  # Execution Settings
  execution:
    max_retries: 3
    screenshot_on_every_step: true
    record_video: true
    generate_script: true
    headless: false
  ```
- [ ] Add descriptive comments
- [ ] Mark API key as placeholder
- [ ] Validate YAML syntax

**Acceptance Criteria:**
- ✅ Valid YAML syntax
- ✅ All required fields present
- ✅ Matches spec.md section 3 structure exactly
- ✅ Comments are clear and helpful
- ✅ Paths are correct for target system

**References:**
- spec.md section 3
- constitution.md sections 6, 13

---

### Task 4.3: Sample Jira Ticket Files
- **Status:** pending
- **Priority:** high
- **Dependencies:** Task 1.1
- **Files:** `Jira_Tickets/RBPLCD-8835.txt`, `Jira_Tickets/RBPLCD-8862.txt`
- **Estimated Time:** 15 minutes

**Description:**
Create sample Jira ticket files for testing.

**Actions:**
- [ ] Create `Jira_Tickets/RBPLCD-8835.txt` with content:
  ```
  [RBPLCD-8835] edit part details
  Status: Open
  Project: RB-PLCD
  Component/s: Teststep

  Steps to Reproduce:
  1. Login
  2. navigate to teststep
  3. click on teststep named as default_Measurement01
  4. open parts accordion
  5. click on edit button of part default_testobject_01
  6. Click on Type and select "Type 5" from drop down
  7. click on save
  8. "Successfully edited" message should be displayed

  Acceptance Criteria:
  "Successfully edited: 'TestObject' default_testobject_01" message should be displayed
  ```
- [ ] Create `Jira_Tickets/RBPLCD-8862.txt` with similar format (9 steps - create placeholder content)
- [ ] Ensure format matches parser expectations

**Acceptance Criteria:**
- ✅ Both files exist in Jira_Tickets folder
- ✅ Format matches spec.md section 3.2
- ✅ RBPLCD-8835.txt has all 8 steps
- ✅ Parser can successfully read both files

**References:**
- spec.md sections 3.2, 16
- constitution.md section 14

---

### Task 4.4: README Documentation
- **Status:** pending
- **Priority:** medium
- **Dependencies:** none
- **Files:** `README.md`
- **Estimated Time:** 45 minutes

**Description:**
Create comprehensive README with setup and usage instructions.

**Actions:**
- [ ] Create README.md with sections:
  1. **Project Overview**
     - Purpose
     - Key features (vision-only, 99% accuracy, 60-90s execution)
     - Innovation (no selectors, no codebase access)
  2. **Technology Stack**
     - Python 3.11+, LangGraph, GPT-4o, Playwright
  3. **Prerequisites**
     - Python 3.11+
     - Edge browser
     - Azure OpenAI API access
  4. **Installation**
     ```bash
     cd C:\Projects\AI_Chat\PLCD\TA_AI_Project
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     playwright install msedge
     ```
  5. **Configuration**
     - Edit plcdtest_config.yaml
     - Add Azure OpenAI API key
     - Adjust paths if needed
  6. **Usage**
     ```bash
     python main.py
     # Enter ticket number when prompted: RBPLCD-8835
     ```
  7. **Project Structure**
     - File/folder layout explanation
  8. **Output Files**
     - HTML reports (Reports/)
     - Execution videos (Videos/)
     - Generated scripts (Generated_Scripts/)
     - Logs (Logs/)
  9. **Troubleshooting**
     - Common issues and solutions
  10. **Success Criteria**
      - PoC goals from spec.md section 13
  11. **Out of Scope**
      - Features not included in PoC
- [ ] Include code blocks and examples
- [ ] Add badges/icons (optional)

**Acceptance Criteria:**
- ✅ Clear, well-structured documentation
- ✅ All installation steps are accurate
- ✅ Usage instructions are complete
- ✅ Examples are helpful
- ✅ Troubleshooting covers common scenarios

**References:**
- spec.md sections 18, 12-14
- constitution.md sections 1, 14

---

## PHASE 5: TESTING & VALIDATION

### Task 5.1: Unit Testing - Utilities
- **Status:** pending
- **Priority:** high
- **Dependencies:** Phase 1 complete
- **Files:** Manual testing
- **Estimated Time:** 1 hour

**Description:**
Test all utility functions independently.

**Test Cases:**
- [ ] **config_loader.py**:
  - Valid config file loads correctly
  - Missing required fields raise clear errors
  - Invalid YAML raises yaml.YAMLError
  - Relative paths resolve to absolute paths
- [ ] **logger.py**:
  - Log files created in correct location
  - File contains DEBUG level logs
  - Console shows INFO level logs
  - Sensitive data is masked (test with fake password)
- [ ] **vision_helper.py**:
  - Client initializes with valid config
  - Screenshot encoding produces valid base64
  - Invalid API key raises authentication error

**Acceptance Criteria:**
- ✅ All utility functions work as expected
- ✅ Error handling is robust
- ✅ No crashes or unexpected behavior
- ✅ Logs provide useful debugging information

**References:**
- constitution.md section 14

---

### Task 5.2: Unit Testing - Agents
- **Status:** pending
- **Priority:** high
- **Dependencies:** Phase 2 complete
- **Files:** Manual testing
- **Estimated Time:** 1.5 hours

**Description:**
Test each agent independently with mock state.

**Test Cases:**
- [ ] **jira_parser_agent**:
  - Successfully parses RBPLCD-8835.txt
  - Extracts ticket ID correctly
  - Extracts all 8 steps as list
  - Extracts acceptance criteria
  - Handles missing file gracefully
  - Handles malformed ticket format
- [ ] **vision_executor_agent** (limited without browser):
  - Browser initialization works
  - State updates correctly after execution
  - Error handling for browser crashes
- [ ] **report_generator_agent**:
  - HTML report generates with valid state
  - File saved to correct location
  - Playwright script generates valid Python
  - Script file saved correctly

**Acceptance Criteria:**
- ✅ Each agent works independently
- ✅ State is updated correctly
- ✅ Error handling works properly
- ✅ No crashes or data corruption
- ✅ Output files are valid

**References:**
- constitution.md section 14

---

### Task 5.3: Integration Testing - RBPLCD-8835
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Phase 4 complete
- **Files:** End-to-end test
- **Estimated Time:** 1 hour

**Description:**
Execute full workflow with RBPLCD-8835 ticket.

**Prerequisites:**
- [ ] Valid Azure OpenAI API key in config
- [ ] RBPLCD-8835.txt exists in Jira_Tickets/
- [ ] Web application is accessible
- [ ] Edge browser installed

**Test Execution:**
- [ ] Run: `python main.py`
- [ ] Enter ticket: RBPLCD-8835
- [ ] Observe execution:
  - Config loads successfully
  - Ticket parsed correctly
  - Browser opens and navigates to login
  - Login executes (credentials entered, button clicked)
  - All 8 steps execute sequentially
  - Screenshots captured before/after each step
  - Video recording active
  - Report generation starts after execution
  - Script generation completes
  - Console displays summary
- [ ] Verify outputs:
  - `Reports/RBPLCD-8835_report_*.html` exists and opens in browser
  - HTML contains all sections with screenshots
  - `Videos/RBPLCD-8835_*.mp4` exists and plays
  - `Generated_Scripts/RBPLCD-8835_script_*.py` exists and is valid Python
  - `Logs/RBPLCD-8835_*.log` exists with detailed logs
- [ ] Measure execution time (target: 60-90 seconds)
- [ ] Check overall status in report

**Acceptance Criteria:**
- ✅ Complete execution without crashes
- ✅ All 8 steps execute (99%+ accuracy target)
- ✅ Execution time: 60-90 seconds
- ✅ HTML report generated with all sections
- ✅ Video recorded and saved
- ✅ Playwright script generated
- ✅ Overall status correct (PASSED/FAILED)
- ✅ Zero manual intervention required

**References:**
- spec.md sections 12, 13, 16
- constitution.md section 15

---

### Task 5.4: Integration Testing - RBPLCD-8862
- **Status:** pending
- **Priority:** critical
- **Dependencies:** Task 5.3 passed
- **Files:** End-to-end test
- **Estimated Time:** 1 hour

**Description:**
Execute full workflow with RBPLCD-8862 ticket (9 steps).

**Prerequisites:**
- [ ] RBPLCD-8862.txt exists and is properly formatted
- [ ] Previous test (Task 5.3) passed

**Test Execution:**
- [ ] Run: `python main.py`
- [ ] Enter ticket: RBPLCD-8862
- [ ] Observe full execution (9 steps)
- [ ] Verify all outputs as in Task 5.3

**Acceptance Criteria:**
- ✅ Complete execution without crashes
- ✅ All 9 steps execute (99%+ accuracy target)
- ✅ Execution time: 60-90 seconds
- ✅ All outputs generated correctly
- ✅ Meets all PoC success criteria

**References:**
- spec.md sections 13, 16
- constitution.md section 15

---

### Task 5.5: Error Scenario Testing
- **Status:** pending
- **Priority:** medium
- **Dependencies:** Task 5.4 passed
- **Files:** Error handling validation
- **Estimated Time:** 1.5 hours

**Description:**
Validate graceful error handling for failure scenarios.

**Test Cases:**
- [ ] **Invalid ticket number**:
  - Enter: INVALID-123
  - Expected: Parser error, clear message, graceful exit
- [ ] **Missing Jira file**:
  - Enter: RBPLCD-9999
  - Expected: File not found error, clear message
- [ ] **Wrong credentials**:
  - Modify config with wrong password
  - Expected: Login fails, retry logic triggers, eventual failure reported
- [ ] **Network failure**:
  - Disconnect network during execution
  - Expected: Timeout errors logged, graceful degradation
- [ ] **API timeout** (simulate):
  - Modify vision_helper to force timeout
  - Expected: Retry logic triggers, max retries reached, continue or fail gracefully
- [ ] **Browser crash** (force close):
  - Kill Edge process during execution
  - Expected: Browser error caught, restart attempted or graceful exit

**Validation:**
- [ ] Check error messages are clear and actionable
- [ ] Verify logs contain useful debugging information
- [ ] Ensure no data corruption or partial files
- [ ] Confirm system is recoverable after errors

**Acceptance Criteria:**
- ✅ All error scenarios handled gracefully
- ✅ Clear error messages displayed
- ✅ Logs contain useful debugging info
- ✅ No crashes or data loss
- ✅ System recoverable after errors

**References:**
- spec.md sections 9, 15, 16
- constitution.md section 5

---

## PHASE 6: DOCUMENTATION & HANDOFF

### Task 6.1: Code Documentation Review
- **Status:** pending
- **Priority:** medium
- **Dependencies:** Phase 5 complete
- **Files:** All Python files
- **Estimated Time:** 1 hour

**Description:**
Review and enhance code documentation throughout the project.

**Actions:**
- [ ] Review all Python files for:
  - Function docstrings (description, args, returns, raises)
  - Type hints on all parameters and return values
  - Class docstrings
  - Module docstrings
  - Inline comments for complex logic
- [ ] Add missing documentation
- [ ] Ensure consistency in docstring format (Google or NumPy style)
- [ ] Update README if any changes needed
- [ ] Optionally create architecture diagram

**Acceptance Criteria:**
- ✅ All functions have docstrings
- ✅ Type hints present throughout
- ✅ Complex sections have explanatory comments
- ✅ Code is maintainable and readable
- ✅ Documentation style is consistent

**References:**
- constitution.md sections 2, 6

---

### Task 6.2: Final Validation Checklist
- **Status:** pending
- **Priority:** critical
- **Dependencies:** All previous tasks complete
- **Files:** Validation checklist
- **Estimated Time:** 30 minutes

**Description:**
Verify all success criteria from spec.md and constitution.md are met.

**Must-Have Criteria (spec.md section 13):**
- [ ] ✅ Execute RBPLCD-8835 successfully
- [ ] ✅ Execute RBPLCD-8862 successfully
- [ ] ✅ Achieve 99%+ accuracy
- [ ] ✅ Generate HTML report with screenshots
- [ ] ✅ Generate execution video
- [ ] ✅ Generate Playwright script
- [ ] ✅ Complete in under 90 seconds
- [ ] ✅ Zero manual intervention

**Implementation Criteria (constitution.md section 15):**
- [ ] ✅ Functionality: Both test cases pass
- [ ] ✅ Accuracy: 99%+ with retry logic
- [ ] ✅ Performance: 60-90 second execution
- [ ] ✅ Output Quality: Complete reports, video, script
- [ ] ✅ Reliability: Graceful error handling
- [ ] ✅ Maintainability: Clean, modular code
- [ ] ✅ Configuration: Zero hardcoded values
- [ ] ✅ Compliance: Follows all constitution principles

**Deliverables:**
- [ ] Document any known limitations
- [ ] Create handoff notes
- [ ] List potential post-PoC enhancements
- [ ] Note any deviations from spec (with justification)

**Acceptance Criteria:**
- ✅ All must-have criteria met
- ✅ No critical bugs remaining
- ✅ Documentation complete and accurate
- ✅ Ready for production PoC use
- ✅ Handoff documentation prepared

**References:**
- spec.md section 13
- constitution.md section 15

---

## Summary

**Total Tasks:** 31
**Estimated Duration:** ~16 hours
**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

**Key Milestones:**
- End of Phase 1: Foundation ready for agent development
- End of Phase 2: All agents implemented and tested independently
- End of Phase 3: Workflow integrated, end-to-end ready
- End of Phase 4: Executable application ready for testing
- End of Phase 5: Validated against success criteria
- End of Phase 6: Production-ready PoC with documentation

**Next Steps:**
Start with Phase 1, Task 1.1: Project Structure Setup
