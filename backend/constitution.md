# Constitution for AI-Powered Vision-Based Test Automation

## Purpose of This Document

This **constitution** defines the **principles, patterns, and rules** that govern how this test automation system is implemented. It serves as the **implementation guide** for the technical requirements defined in `spec.md`.

**Use this document when:**
- Adding new agents to the system
- Modifying existing agents
- Extending the workflow
- Making architectural decisions
- Ensuring code quality and consistency

---

## Core Principles

### 1. Code Organization

**File Structure Rules:**
- ✅ Follow the exact file structure defined in `spec.md` section 8
- ✅ Keep agents modular and independent in separate files
- ✅ Use `utils/` for shared functionality (logging, config, vision API)
- ✅ One agent = one file, no mixing responsibilities
- ✅ All agents in `agents/` directory
- ✅ Workflow orchestration only in `workflows/` directory

**Example:**
```
agents/
├── jira_parser_agent.py          # ONLY Jira parsing logic
├── vision_executor_agent.py      # ONLY vision execution logic
└── report_generator_agent.py     # ONLY report generation logic
```

❌ **Anti-pattern:** Mixing parsing logic with execution logic in one file
✅ **Correct pattern:** Separate agent files with single responsibility

---

### 2. Python Standards

**Code Quality Requirements:**
```python
# ✅ ALWAYS use type hints
def jira_parser_agent(state: TestAutomationState) -> TestAutomationState:
    pass

# ✅ ALWAYS use Pydantic for validation
from pydantic import BaseModel, Field

class VisionResult(BaseModel):
    element_description: str
    coordinates: dict
    confidence: float = Field(ge=0.0, le=1.0)

# ✅ ALWAYS use dataclasses or TypedDict for structured data
from typing import TypedDict

class JiraData(TypedDict):
    ticket_id: str
    module: str
    steps: List[Dict]

# ✅ ALWAYS follow PEP 8
# - 4 spaces for indentation
# - 2 blank lines between top-level functions
# - 1 blank line between methods in a class
# - Max line length: 100 characters (flexible to 120 for readability)
```

**Python Version:** 3.11+ features are allowed and encouraged

---

### 3. Agent Design Principles

**Stateless Agent Architecture:**

```python
# ✅ CORRECT: Agent receives state, returns updated state
def my_agent(state: TestAutomationState) -> TestAutomationState:
    # Read from state
    config = state['config']
    ticket_number = state['ticket_number']

    # Process
    result = do_something(config, ticket_number)

    # Update state
    state['my_result'] = result

    # Return updated state
    return state

# ❌ WRONG: Agent storing state in class variables
class MyAgent:
    def __init__(self):
        self.state = {}  # ❌ NO! State must flow through LangGraph

    def execute(self):
        pass
```

**Key Rules:**
- ✅ Each agent must be a **pure function** that takes `state` and returns `state`
- ✅ No class-based agents storing state internally
- ✅ All inter-agent communication through shared `state` dictionary
- ✅ No direct agent-to-agent function calls
- ✅ Each agent focuses on **single responsibility** (SRP)

**Agent Function Signature:**
```python
def agent_name(state: TestAutomationState) -> TestAutomationState:
    """
    Brief description of what this agent does.

    Args:
        state: Current workflow state containing required inputs

    Returns:
        Updated state with new data added

    Raises:
        SpecificError: When specific condition occurs
    """
    logger = logging.getLogger("TA_AI_Project")

    try:
        # 1. Read from state
        # 2. Process
        # 3. Update state
        # 4. Return state
        return state
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        state['errors'].append({
            'agent': 'agent_name',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        raise
```

---

### 4. LangGraph Workflow Patterns

**StateGraph Structure:**

```python
from langgraph.graph import StateGraph, END
from models.state import TestAutomationState

# Create graph
workflow = StateGraph(TestAutomationState)

# Add nodes (agents and utility functions)
workflow.add_node("load_config", load_config_node)
workflow.add_node("get_ticket_input", get_ticket_input_node)
workflow.add_node("jira_parser", jira_parser_agent)
workflow.add_node("vision_executor", vision_executor_agent)
workflow.add_node("report_generator", report_generator_agent)

# Define linear flow
workflow.set_entry_point("load_config")
workflow.add_edge("load_config", "get_ticket_input")
workflow.add_edge("get_ticket_input", "jira_parser")
workflow.add_edge("jira_parser", "vision_executor")
workflow.add_edge("vision_executor", "report_generator")
workflow.add_edge("report_generator", END)

# Compile
app = workflow.compile()
```

**Workflow Rules:**
- ✅ Use `StateGraph` for orchestration
- ✅ State schema must match `TestAutomationState` TypedDict
- ✅ No circular dependencies in workflow graph
- ✅ Linear flow for PoC: `START → config → input → parse → execute → report → END`
- ✅ Each node is a pure function taking and returning state

**Adding Conditional Routing (Future Enhancement):**
```python
def should_retry(state: TestAutomationState) -> str:
    """Route based on execution status"""
    if state.get('overall_status') == 'FAILED' and state.get('retry_count', 0) < 3:
        return "vision_executor"  # Retry execution
    else:
        return "report_generator"  # Proceed to report

workflow.add_conditional_edges(
    "vision_executor",
    should_retry,
    {
        "vision_executor": "vision_executor",
        "report_generator": "report_generator"
    }
)
```

---

### 5. Error Handling Standards

**Try-Except Pattern:**

```python
def my_agent(state: TestAutomationState) -> TestAutomationState:
    logger = logging.getLogger("TA_AI_Project")

    try:
        # External calls MUST be wrapped in try-except

        # File I/O
        with open(file_path, 'r') as f:
            content = f.read()

        # API calls
        result = call_vision_api(screenshot, prompt)

        # Browser operations
        page.mouse.click(x, y)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        state['errors'].append({'type': 'FileNotFoundError', 'message': str(e)})
        raise  # Re-raise for critical errors

    except openai.APIError as e:
        logger.error(f"API error: {e}")
        # Retry logic for transient errors
        if "timeout" in str(e):
            retry()
        else:
            raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        state['errors'].append({'type': type(e).__name__, 'message': str(e)})
        raise

    return state
```

**Error Handling Rules:**
- ✅ Wrap ALL external calls (API, browser, file I/O) in try-except
- ✅ Log errors with context and stack traces (`exc_info=True`)
- ✅ Add errors to `state['errors']` list for tracking
- ✅ Use specific exception types when possible
- ✅ Implement retry logic for transient failures
- ✅ Re-raise exceptions for critical errors
- ❌ NEVER swallow exceptions silently (`except: pass` is forbidden)

**Graceful Degradation Strategy:**
- ❌ Non-critical step fails after 3 retries → Mark `FAILED`, continue to next step
- ❌ Critical step fails (login) → Stop execution, report failure, exit gracefully
- ❌ Browser crashes → Restart browser, retry current step from checkpoint
- ❌ API unavailable → Wait and retry max 3 times, then fail with clear error message

---

### 6. Configuration Management

**Configuration Rules:**

```python
# ✅ CORRECT: Load config once at workflow start
def load_config_node(state: TestAutomationState) -> TestAutomationState:
    config = load_config("plcdtest_config.yaml")
    state['config'] = config
    return state

# ✅ CORRECT: Access config from state
def my_agent(state: TestAutomationState) -> TestAutomationState:
    web_url = state['config']['web_url']
    username = state['config']['login']['username']
    max_retries = state['config']['execution']['max_retries']

# ❌ WRONG: Hardcoded values
def my_agent(state: TestAutomationState) -> TestAutomationState:
    web_url = "http://fe0vm03313.de.bosch.com/rbplcd_t/client/login"  # ❌ NO!
    username = "mechanic"  # ❌ NO!
```

**Configuration Principles:**
- ✅ ALL configuration from `plcdtest_config.yaml` ONLY
- ✅ No hardcoded paths, URLs, credentials, or magic numbers in code
- ✅ Load config once at workflow start
- ✅ Pass config through state to all agents
- ✅ Validate config schema on load (use Pydantic if needed)
- ❌ Never commit secrets to code
- ❌ Never use environment variables for PoC (config file only)

---

### 7. GPT-4o Vision Integration

**Vision API Call Pattern:**

```python
def call_gpt4o_vision(
    screenshot: bytes,
    prompt: str,
    config: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Call GPT-4o Vision API with screenshot and prompt.

    Args:
        screenshot: Raw screenshot bytes
        prompt: Text prompt for vision model
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Parsed JSON response from GPT-4o
    """
    import base64
    from openai import AzureOpenAI

    try:
        # Encode screenshot
        base64_image = base64.b64encode(screenshot).decode('utf-8')

        # Initialize client
        client = AzureOpenAI(
            api_key=config["azure_openai"]["api_key"],
            api_version=config["azure_openai"]["api_version"],
            azure_endpoint=config["azure_openai"]["endpoint"]
        )

        # Call API
        response = client.chat.completions.create(
            model=config["azure_openai"]["deployment_gpt4o"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }}
                ]
            }],
            max_tokens=500,        # Control costs
            temperature=0.1        # Deterministic results
        )

        # Parse JSON response
        content = response.choices[0].message.content
        result = json.loads(content)

        logger.debug(f"Vision API response: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vision response: {e}")
        logger.debug(f"Raw response: {content}")
        raise

    except Exception as e:
        logger.error(f"Vision API call failed: {e}")
        raise
```

**Vision Prompt Structure:**

```python
def build_vision_prompt(step_text: str, context: Dict) -> str:
    """
    Build structured prompt for GPT-4o vision.

    Args:
        step_text: Test step description
        context: Current page context (module, previous action, etc.)

    Returns:
        Formatted prompt string
    """
    prompt = f"""
Context:
- Current page: {context.get('current_page', 'Unknown')}
- Module: {context.get('module', 'Unknown')}
- Previous action: {context.get('previous_action', 'None')}

Task: {step_text}

Find the UI element needed to complete this task and return JSON in this EXACT format:
{{
    "element_description": "Description of the element you identified",
    "coordinates": {{"x": 850, "y": 450}},
    "action_type": "click" | "type" | "select",
    "value": "text to type (only if action_type is 'type' or 'select')",
    "confidence": 0.95
}}

Constraints:
- Coordinates must be within viewport bounds (0-1920 x, 0-1080 y)
- Choose the most prominent element if multiple matches exist
- Confidence must be >= 0.85 to proceed
- Return ONLY valid JSON, no additional text or explanation
"""
    return prompt
```

**Vision API Rules:**
- ✅ Base64 encode all screenshots before API calls
- ✅ Structured prompts with context, task, format, constraints
- ✅ Parse JSON responses with error handling
- ✅ Temperature=0.1 for deterministic results
- ✅ Max tokens=500 to control costs
- ✅ Log API calls and responses (truncated) for debugging
- ❌ Never log full base64 images (too large)
- ❌ Never exceed 20 vision API calls per test (cost control)

---

### 8. Browser Automation

**Playwright Pattern:**

```python
from playwright.sync_api import sync_playwright
import time

def execute_with_browser(config: Dict, steps: List[Dict], logger: logging.Logger):
    """Execute test steps using Playwright browser automation."""

    with sync_playwright() as playwright:
        # Launch browser
        browser = playwright.chromium.launch(
            channel="msedge",
            headless=config['execution']['headless']
        )

        # Create context with viewport
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir="Videos/" if config['execution']['record_video'] else None
        )

        # Create page
        page = context.new_page()

        try:
            # Navigate to URL
            page.goto(config['web_url'])
            page.wait_for_load_state('networkidle')
            time.sleep(config['wait_times']['page_load'] / 1000)

            # Execute steps
            for step in steps:
                # Take screenshot before
                screenshot_before = page.screenshot()

                # Get action from vision
                action = get_vision_action(screenshot_before, step, config, logger)

                # Execute action
                if action['action_type'] == 'click':
                    page.mouse.click(action['coordinates']['x'], action['coordinates']['y'])
                    time.sleep(config['wait_times']['after_click'] / 1000)

                elif action['action_type'] == 'type':
                    page.mouse.click(action['coordinates']['x'], action['coordinates']['y'])
                    page.keyboard.type(action['value'])
                    time.sleep(config['wait_times']['after_type'] / 1000)

                elif action['action_type'] == 'select':
                    # Click dropdown
                    page.mouse.click(action['coordinates']['x'], action['coordinates']['y'])
                    time.sleep(config['wait_times']['after_dropdown'] / 1000)
                    # Select option (needs vision to find option)

                # Take screenshot after
                screenshot_after = page.screenshot()

                # Verify success
                verify_action_success(screenshot_after, step, logger)

        finally:
            # Always clean up
            context.close()
            browser.close()
```

**Browser Automation Rules:**
- ✅ Use Playwright with Edge browser (`channel="msedge"`)
- ✅ Coordinate-based clicking ONLY (no CSS selectors, no XPath)
- ✅ Take screenshots before AND after each action
- ✅ Implement wait times from config after each action
- ✅ Record video of entire execution if enabled in config
- ✅ Handle browser crashes with restart logic
- ✅ Always close context and browser in `finally` block
- ❌ NEVER use selectors for element location (vision only!)

**Wait Time Strategy:**
```python
# Use config values
time.sleep(config['wait_times']['after_click'] / 1000)  # Convert ms to seconds

# Wait for specific conditions
page.wait_for_load_state('networkidle')  # Wait for network idle
page.wait_for_timeout(2000)  # Explicit timeout if needed
```

---

### 9. Retry Logic

**3-Level Retry Strategy:**

```python
def execute_step_with_retry(
    step: Dict,
    page: Page,
    config: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Execute step with 3-level retry logic.

    Returns:
        Execution result with status, confidence, retries, etc.
    """
    max_retries = config['execution']['max_retries']  # 3

    for attempt in range(max_retries):
        try:
            logger.info(f"Step {step['num']}: Attempt {attempt + 1}/{max_retries}")

            # Take screenshot
            screenshot = page.screenshot()

            # Build prompt based on attempt
            if attempt == 0:
                # Standard prompt
                prompt = build_standard_prompt(step)
            elif attempt == 1:
                # Enhanced prompt + cropped screenshot
                prompt = build_enhanced_prompt(step, previous_failure)
                screenshot = crop_to_relevant_area(screenshot)
            else:
                # Most detailed prompt
                prompt = build_detailed_prompt(step, all_previous_failures)

            # Call vision
            action = call_gpt4o_vision(screenshot, prompt, config, logger)

            # Check confidence
            if action['confidence'] < 0.85:
                logger.warning(f"Low confidence: {action['confidence']}")
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    return {'status': 'FAILED', 'reason': 'Low confidence', 'retries': attempt + 1}

            # Execute action
            execute_browser_action(page, action, config)

            # Verify success
            screenshot_after = page.screenshot()
            verification = verify_action_success(screenshot_after, step, logger)

            if verification['success']:
                return {
                    'status': 'PASSED',
                    'action': action,
                    'retries': attempt,
                    'execution_time': verification['time']
                }
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Verification failed, retrying...")
                    continue
                else:
                    return {
                        'status': 'FAILED',
                        'reason': 'Verification failed',
                        'retries': attempt + 1
                    }

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    'status': 'FAILED',
                    'reason': str(e),
                    'retries': attempt + 1
                }

    # Should never reach here, but just in case
    return {'status': 'FAILED', 'reason': 'Max retries exceeded', 'retries': max_retries}
```

**Retry Rules:**
- ✅ Maximum 3 retries per step
- ✅ Attempt 1: Standard prompt
- ✅ Attempt 2: Enhanced prompt + cropped screenshot
- ✅ Attempt 3: Most detailed instructions
- ✅ Log retry count in execution results
- ✅ Mark FAILED after max retries, continue to next step (unless critical)
- ✅ Exponential backoff for API rate limits

---

### 10. Logging and Debugging

**Logging Setup:**

```python
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(log_folder: str, ticket_id: str, logger_name: str) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        log_folder: Folder to save log files
        ticket_id: Ticket ID for log filename
        logger_name: Name of logger

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create log folder
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_folder) / f"{ticket_id}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

**Logging Best Practices:**

```python
logger = logging.getLogger("TA_AI_Project")

# ✅ Log agent entry/exit
logger.info("=" * 70)
logger.info("JIRA PARSER AGENT - Starting")
logger.info("=" * 70)

# ✅ Log important operations
logger.info(f"Parsing ticket: {ticket_number}")
logger.info(f"Extracted {len(steps)} test steps")

# ✅ Log debug information
logger.debug(f"Looking for file: {ticket_file}")
logger.debug(f"Extracted ticket ID: {ticket_id}")

# ✅ Log warnings
logger.warning(f"Low confidence: {confidence}")

# ✅ Log errors with stack traces
logger.error(f"API call failed: {e}", exc_info=True)

# ❌ NEVER log sensitive data
# logger.info(f"Password: {password}")  # ❌ NO!
# logger.debug(f"API Key: {api_key}")  # ❌ NO!

# ❌ NEVER log full base64 images
# logger.debug(f"Screenshot: {base64_image}")  # ❌ NO! Too large

# ✅ Log truncated API responses
logger.debug(f"Vision response: {str(response)[:200]}...")
```

**Logging Rules:**
- ✅ Log all agent actions with timestamps
- ✅ Include step numbers in logs
- ✅ Log actions taken (click, type, select)
- ✅ Log API calls and responses (truncated to 200 chars)
- ✅ Use `exc_info=True` for exception stack traces
- ❌ Never log sensitive data (passwords, API keys, tokens)
- ❌ Never log full base64 images (too large)
- ✅ Save logs to `Logs/` folder with timestamp in filename

---

### 11. Output Generation

**HTML Report Rules:**

```python
# ✅ Use Jinja2 templates
from jinja2 import Template

template = Template(template_content)
html = template.render(**context)

# ✅ Embed screenshots as base64
def embed_screenshot(screenshot_path: str) -> str:
    with open(screenshot_path, 'rb') as f:
        image_data = f.read()
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

# ✅ Include all required sections
context = {
    'ticket_id': jira_data['ticket_id'],
    'module': jira_data['module'],
    'test_title': jira_data['title'],
    'overall_status': state['overall_status'],
    'execution_time': state['total_execution_time'],
    'execution_results': execution_results,
    'passed_count': len([r for r in results if r['status'] == 'PASSED']),
    'failed_count': len([r for r in results if r['status'] == 'FAILED']),
    # ... more context
}

# ✅ Save with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{ticket_id}_report_{timestamp}.html"
```

**Playwright Script Generation:**

```python
# ✅ Generate valid Python code
script_lines = [
    '"""',
    f'Generated Playwright script for {ticket_id}',
    f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    '"""',
    '',
    'from playwright.sync_api import sync_playwright',
    'import time',
    '',
    'def run():',
    '    with sync_playwright() as playwright:',
    # ... browser setup
]

# ✅ Include coordinates from vision execution
for result in execution_results:
    x = result['coordinates']['x']
    y = result['coordinates']['y']
    script_lines.append(f'        page.mouse.click({x}, {y})')
    script_lines.append(f'        time.sleep(1.0)')

# ✅ Save to Generated_Scripts folder
script_content = '\n'.join(script_lines)
script_path = Path(config['folders']['scripts']) / f"{ticket_id}_script_{timestamp}.py"
with open(script_path, 'w') as f:
    f.write(script_content)
```

**Output Rules:**
- ✅ HTML reports use Jinja2 templates
- ✅ Embed ALL screenshots inline as base64 (no external file dependencies)
- ✅ Generate valid, executable Playwright Python scripts
- ✅ Save outputs to configured folders from config
- ✅ Follow naming convention: `{ticket_id}_{type}_{timestamp}.{ext}`
- ✅ Include metadata (generation time, version, etc.)

---

### 12. Security & Secrets Management

**Security Rules:**

```python
# ✅ CORRECT: Read secrets from config file
api_key = config['azure_openai']['api_key']
username = config['login']['username']
password = config['login']['password']

# ❌ WRONG: Hardcoded secrets
api_key = "98dkVOUDLG4wm9OCmF8pxnR48BoUCPYKzfI9p4zYGP5uVh7TiLLwJQQJ99BDAC5RqLJXJ3w3AAABACOGWUeW"  # ❌ NO!

# ✅ Mask passwords in logs
logger.info(f"Logging in as {username}")  # ✅ OK
# logger.info(f"Password: {password}")  # ❌ NO!

# ✅ Mask passwords in screenshots (if possible)
# This is challenging with vision-only approach, but document the limitation

# ✅ Validate file paths to prevent traversal
from pathlib import Path

def safe_path(base_folder: str, relative_path: str) -> Path:
    base = Path(base_folder).resolve()
    target = (base / relative_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Path traversal attempt: {relative_path}")
    return target

# ✅ Sanitize user inputs
import re

def validate_ticket_id(ticket_id: str) -> bool:
    return bool(re.match(r'^[A-Z]+-\d+$', ticket_id))
```

**Security Principles:**
- ❌ Never commit API keys or credentials to code or version control
- ✅ Read ALL secrets from config file only
- ✅ Mask passwords in logs and console output
- ✅ Validate all file paths to prevent directory traversal
- ✅ Sanitize user inputs (ticket numbers, file paths)
- ✅ Use `.gitignore` to exclude `plcdtest_config.yaml` if it contains secrets

---

### 13. Testing and Validation

**State Validation:**

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.state import TestAutomationState

def validate_state(state: TestAutomationState, required_fields: List[str]) -> None:
    """Validate that required fields exist in state."""
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field in state: {field}")

# Use at agent entry
def my_agent(state: TestAutomationState) -> TestAutomationState:
    validate_state(state, ['config', 'ticket_number'])
    # ... agent logic
```

**File Existence Checks:**

```python
from pathlib import Path

# ✅ Always check file existence before reading
ticket_file = Path(jira_folder) / f"{ticket_number}.txt"
if not ticket_file.exists():
    raise FileNotFoundError(f"Ticket file not found: {ticket_file}")

# ✅ Create output directories if they don't exist
output_folder = Path(config['folders']['reports'])
output_folder.mkdir(parents=True, exist_ok=True)
```

**API Response Validation:**

```python
# ✅ Validate vision API responses
def validate_vision_response(response: Dict) -> None:
    required_fields = ['element_description', 'coordinates', 'action_type', 'confidence']
    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing field in vision response: {field}")

    # Validate coordinates
    x = response['coordinates'].get('x', -1)
    y = response['coordinates'].get('y', -1)
    if not (0 <= x <= 1920 and 0 <= y <= 1080):
        raise ValueError(f"Coordinates out of viewport bounds: ({x}, {y})")

    # Validate confidence
    if not (0.0 <= response['confidence'] <= 1.0):
        raise ValueError(f"Invalid confidence value: {response['confidence']}")
```

**Testing Rules:**
- ✅ Validate state schema at each node transition
- ✅ Check required fields before agent execution
- ✅ Verify file existence before reading
- ✅ Validate API responses before using data
- ✅ Test with provided tickets: RBPLCD-8835, RBPLCD-8862
- ✅ Test error scenarios (missing files, invalid formats, API failures)

---

### 14. Dependencies Management

**Requirements:**

```
# Core frameworks
langchain==0.1.0
langgraph==0.1.0
playwright==1.40.0
openai==1.0.0
pydantic==2.0.0

# Utilities
pyyaml==6.0
pillow==10.0.0
jinja2==3.1.0
python-dotenv==1.0.0
```

**Dependency Rules:**
- ✅ Use ONLY libraries listed in `requirements.txt`
- ✅ Pin versions for reproducibility
- ✅ No external dependencies without updating requirements.txt AND documenting in spec.md
- ✅ Import only what's needed per file (no wildcard imports)
- ❌ No installing libraries globally (use virtual environment)

**Import Style:**

```python
# ✅ Specific imports
from pathlib import Path
from typing import Dict, List
from models.state import TestAutomationState

# ✅ Module imports
import logging
import json
import time

# ❌ Wildcard imports (forbidden)
from models.state import *  # ❌ NO!
```

---

## Implementation Guidelines

### When Adding a New Agent

**Checklist:**

1. **Create Agent File**
   - [ ] Create new file in `agents/` directory
   - [ ] Name follows pattern: `{agent_name}_agent.py`
   - [ ] Add to `agents/__init__.py` if needed

2. **Define Agent Function**
   - [ ] Function signature: `def agent_name(state: TestAutomationState) -> TestAutomationState:`
   - [ ] Add comprehensive docstring
   - [ ] Include type hints for all parameters

3. **Implement Agent Logic**
   - [ ] Read required fields from state
   - [ ] Validate inputs
   - [ ] Implement core functionality
   - [ ] Handle errors with try-except
   - [ ] Update state with results
   - [ ] Return updated state

4. **Add Logging**
   - [ ] Log agent entry/exit
   - [ ] Log important operations
   - [ ] Log errors with context

5. **Add to Workflow**
   - [ ] Add node to workflow in `workflows/test_workflow.py`
   - [ ] Define edge connections
   - [ ] Test workflow compilation

6. **Update Documentation**
   - [ ] Update `spec.md` with agent description
   - [ ] Update this `constitution.md` if new patterns introduced
   - [ ] Add example usage

**Example New Agent:**

```python
"""
Example New Agent

Description: What this agent does
"""

import logging
from typing import Dict
from models.state import TestAutomationState


def example_agent(state: TestAutomationState) -> TestAutomationState:
    """
    Brief description of agent functionality.

    Args:
        state: Current workflow state containing required inputs

    Returns:
        Updated state with new data added

    Raises:
        ValueError: If required input is missing
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("=" * 70)
    logger.info("EXAMPLE AGENT - Starting")
    logger.info("=" * 70)

    try:
        # 1. Validate inputs
        if 'required_field' not in state:
            raise ValueError("Missing required_field in state")

        # 2. Read from state
        config = state['config']
        required_data = state['required_field']

        # 3. Process
        result = process_data(required_data, config, logger)

        # 4. Update state
        state['example_result'] = result

        logger.info("EXAMPLE AGENT - Complete")
        logger.info("=" * 70)

        return state

    except Exception as e:
        logger.error(f"Example agent failed: {e}", exc_info=True)
        state['errors'].append({
            'agent': 'example_agent',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        raise


def process_data(data: Dict, config: Dict, logger: logging.Logger) -> Dict:
    """Helper function for processing."""
    # Implementation
    pass
```

---

### File Creation Order (for New Projects)

**Recommended Implementation Sequence:**

1. `models/state.py` - State schema (TypedDict)
2. `utils/config_loader.py` - Configuration loader
3. `utils/logger.py` - Logging setup
4. `utils/vision_helper.py` - Vision API wrapper
5. `agents/jira_parser_agent.py` - Jira parser
6. `agents/vision_executor_agent.py` - Vision executor
7. `agents/report_generator_agent.py` - Report generator
8. `workflows/test_workflow.py` - LangGraph workflow
9. `main.py` - Entry point (interactive)
10. `run_test.py` - Entry point (command-line)
11. `templates/report_template.html` - HTML template
12. `requirements.txt` - Dependencies
13. `README.md` - Documentation

**Rationale:**
- Build foundation first (models, utils)
- Then agents sequentially (parser → executor → reporter)
- Integrate workflow last
- Each component can be tested independently

---

## Must NOT Do (Anti-Patterns)

### Forbidden Practices

❌ **Use CSS selectors or XPath for element location**
```python
# ❌ FORBIDDEN
element = page.locator("#username")
element.click()

# ✅ CORRECT: Use vision + coordinates
coordinates = get_vision_coordinates(screenshot, "find username field")
page.mouse.click(coordinates['x'], coordinates['y'])
```

❌ **Hardcode any paths, URLs, or credentials**
```python
# ❌ FORBIDDEN
web_url = "http://fe0vm03313.de.bosch.com/rbplcd_t/client/login"
username = "mechanic"

# ✅ CORRECT: Read from config
web_url = config['web_url']
username = config['login']['username']
```

❌ **Create monolithic files mixing multiple responsibilities**
```python
# ❌ FORBIDDEN: One file doing parsing, execution, and reporting
def do_everything(state):
    # parse jira
    # execute tests
    # generate report
    pass

# ✅ CORRECT: Separate agents
def jira_parser_agent(state): pass
def vision_executor_agent(state): pass
def report_generator_agent(state): pass
```

❌ **Skip type hints or validation**
```python
# ❌ FORBIDDEN
def my_agent(state):
    return state

# ✅ CORRECT
def my_agent(state: TestAutomationState) -> TestAutomationState:
    return state
```

❌ **Use global variables for state**
```python
# ❌ FORBIDDEN
GLOBAL_STATE = {}

def my_agent():
    GLOBAL_STATE['result'] = 'value'

# ✅ CORRECT: State flows through LangGraph
def my_agent(state: TestAutomationState) -> TestAutomationState:
    state['result'] = 'value'
    return state
```

❌ **Implement features marked "Out of Scope" for PoC**
```python
# ❌ FORBIDDEN for PoC
def run_multiple_tests_in_parallel(): pass
def integrate_with_ci_cd(): pass
def deploy_to_cloud(): pass

# ✅ CORRECT: Focus on core PoC features
def run_single_test(): pass
def execute_locally(): pass
```

❌ **Add libraries not in requirements.txt**
```python
# ❌ FORBIDDEN
import selenium  # Not in requirements.txt!
import requests  # Not needed for PoC

# ✅ CORRECT: Use only approved libraries
import playwright
import openai
```

❌ **Create circular dependencies between agents**
```python
# ❌ FORBIDDEN
# In jira_parser_agent.py
from agents.vision_executor_agent import vision_executor_agent

# In vision_executor_agent.py
from agents.jira_parser_agent import jira_parser_agent

# ✅ CORRECT: Agents communicate only through state
# No direct imports between agents
```

❌ **Store state in agent classes**
```python
# ❌ FORBIDDEN
class MyAgent:
    def __init__(self):
        self.state = {}

    def execute(self):
        self.state['result'] = 'value'

# ✅ CORRECT: Stateless functions
def my_agent(state: TestAutomationState) -> TestAutomationState:
    state['result'] = 'value'
    return state
```

❌ **Skip error handling for external calls**
```python
# ❌ FORBIDDEN
result = call_vision_api(screenshot, prompt)  # What if it fails?
with open(file_path, 'r') as f:  # What if file doesn't exist?
    content = f.read()

# ✅ CORRECT
try:
    result = call_vision_api(screenshot, prompt)
except openai.APIError as e:
    logger.error(f"Vision API failed: {e}")
    raise

try:
    with open(file_path, 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
```

---

## Must Do (Required Practices)

### Mandatory Patterns

✅ **Follow exact file structure from spec.md**
- See section 8 of spec.md for complete file organization

✅ **Use LangGraph StateGraph for orchestration**
```python
from langgraph.graph import StateGraph, END
workflow = StateGraph(TestAutomationState)
```

✅ **Implement retry logic with 3 attempts**
```python
for attempt in range(3):
    # Try to execute
    # Refine approach on each attempt
```

✅ **Take screenshots before/after each action**
```python
screenshot_before = page.screenshot()
execute_action()
screenshot_after = page.screenshot()
```

✅ **Log all actions and errors**
```python
logger.info("Executing step...")
logger.error("Step failed", exc_info=True)
```

✅ **Validate state at each transition**
```python
def validate_state(state, required_fields):
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing: {field}")
```

✅ **Use configuration from YAML file**
```python
config = load_config("plcdtest_config.yaml")
state['config'] = config
```

✅ **Generate HTML report with screenshots**
```python
# Embed screenshots as base64
# Use Jinja2 template
# Save to Reports folder
```

✅ **Record execution video**
```python
context = browser.new_context(
    record_video_dir="Videos/"
)
```

✅ **Generate Playwright script**
```python
# Convert execution log to Python code
# Save to Generated_Scripts folder
```

✅ **Handle all error scenarios gracefully**
```python
try:
    # Execute
except SpecificError as e:
    logger.error(f"Error: {e}")
    # Handle gracefully
```

✅ **Stay within PoC scope**
- Focus on core features
- Don't over-engineer
- Keep it simple and working

---

## Success Criteria for Implementation

### Code Quality Checklist

**Before Considering Implementation Complete:**

1. **Functionality**
   - [ ] Successfully executes RBPLCD-8835
   - [ ] Successfully executes RBPLCD-8862
   - [ ] Achieves 99%+ accuracy with retry logic
   - [ ] Completes in 60-90 seconds
   - [ ] Zero manual intervention required

2. **Output Quality**
   - [ ] Generates HTML report with embedded screenshots
   - [ ] Records execution video
   - [ ] Generates executable Playwright script
   - [ ] All outputs saved to correct folders

3. **Code Quality**
   - [ ] All agents are stateless functions
   - [ ] Type hints on all functions
   - [ ] Comprehensive error handling
   - [ ] No hardcoded values
   - [ ] Follows PEP 8 style guide

4. **Reliability**
   - [ ] Handles browser crashes gracefully
   - [ ] Handles API errors gracefully
   - [ ] Handles missing files gracefully
   - [ ] Retry logic works correctly
   - [ ] No silent failures

5. **Logging**
   - [ ] All agents log entry/exit
   - [ ] All actions logged with timestamps
   - [ ] All errors logged with stack traces
   - [ ] No sensitive data in logs

6. **Security**
   - [ ] No hardcoded credentials
   - [ ] Config file not committed to git
   - [ ] Passwords masked in logs
   - [ ] File path validation implemented

7. **Documentation**
   - [ ] All functions have docstrings
   - [ ] spec.md updated if architecture changed
   - [ ] constitution.md updated if new patterns added
   - [ ] README.md has usage instructions

8. **Testing**
   - [ ] Tested with RBPLCD-8835
   - [ ] Tested with RBPLCD-8862
   - [ ] Tested error scenarios
   - [ ] Tested with missing files
   - [ ] Tested with invalid credentials

---

## Notes for Extending the System

### Adding New Agents

**When to add a new agent:**
- New distinct responsibility emerges
- Current agents become too complex
- New external integration needed (e.g., Slack notifications)
- New analysis or validation step required

**How to add a new agent:**
1. Follow "When Adding a New Agent" checklist above
2. Ensure agent is stateless (takes state, returns state)
3. Add to workflow with proper edge connections
4. Update spec.md section 12.1 with agent description
5. Test integration with existing workflow

**Example: Adding a "Validation Agent"**

```python
# agents/validation_agent.py
def validation_agent(state: TestAutomationState) -> TestAutomationState:
    """
    Validate execution results against acceptance criteria.

    Compares execution outcome with expected criteria from Jira ticket.
    """
    logger = logging.getLogger("TA_AI_Project")

    acceptance_criteria = state['acceptance_criteria']
    execution_results = state['execution_results']

    # Use GPT-4o to compare results with criteria
    validation_result = validate_with_gpt4o(
        criteria=acceptance_criteria,
        results=execution_results,
        config=state['config']
    )

    state['validation_result'] = validation_result
    state['validated'] = validation_result['passed']

    return state

# Update workflow
workflow.add_node("validation", validation_agent)
workflow.add_edge("vision_executor", "validation")
workflow.add_edge("validation", "report_generator")
```

### Modifying Workflow

**When to modify workflow:**
- Adding conditional routing
- Adding new agents
- Changing execution order
- Adding parallel execution

**Example: Adding Conditional Routing**

```python
def should_retry_execution(state: TestAutomationState) -> str:
    """Decide whether to retry execution or proceed to report."""
    if state.get('overall_status') == 'FAILED':
        retry_count = state.get('retry_count', 0)
        if retry_count < 2:
            return "retry"
    return "report"

workflow.add_conditional_edges(
    "vision_executor",
    should_retry_execution,
    {
        "retry": "vision_executor",
        "report": "report_generator"
    }
)
```

---

## References

- **Specification Document:** `spec.md` - Technical requirements and architecture
- **LangGraph Documentation:** https://langchain-ai.github.io/langgraph/
- **Playwright Documentation:** https://playwright.dev/python/
- **OpenAI Vision API:** https://platform.openai.com/docs/guides/vision
- **Pydantic Documentation:** https://docs.pydantic.dev/
- **PEP 8 Style Guide:** https://pep8.org/

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-07 | Initial constitution document |
| 1.1 | 2025-10-20 | Enhanced with detailed patterns, examples, and extension guidelines |

---

**END OF CONSTITUTION**
