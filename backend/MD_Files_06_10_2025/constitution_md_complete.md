# AI-Powered Test Automation Tool - Constitution

## Document Overview

**Version:** 1.0  
**Date:** 2025-10-06  
**Purpose:** Technical standards and architectural principles for PoC development

---

## 1. Language & Framework

### Core Languages
- Python 3.11+
- TypeScript 5.0+ (for Action Libraries)
- JavaScript ES6+ (for frontend)

### Primary Frameworks
- Backend: FastAPI 0.104+
- AI: LangChain 0.1+, LangGraph 0.1+
- Testing: Cypress 13.0+
- Browser automation: Playwright 1.40+ (for runtime extraction)

### Required Libraries
- Azure OpenAI SDK
- Pydantic 2.0+ (data validation)
- Uvicorn (ASGI server)
- WebSocket support (real-time chat)
- python-docx (Jira file reading)

---

## 2. Code Style & Standards

### Python Code
- Follow PEP 8 strictly
- Type hints required for all functions
- Docstrings: Google style
- Line length: 100 characters max
- Use f-strings for formatting
- Prefer pathlib over os.path

### TypeScript Code
- Strict mode enabled
- Type annotations required
- Interface over type where applicable
- Use const/let, never var
- Method naming: camelCase

### Code Quality
- No unused imports
- No commented-out code in production
- Descriptive variable names (no single letters except i, j in loops)
- Functions: max 50 lines (split if longer)
- Files: max 500 lines

---

## 3. Architecture Principles

### Multi-Agent Design
- Each agent: single responsibility
- Agents communicate only via shared state (no direct calls)
- LangGraph StateGraph orchestrates all agents
- TypedDict for state schema (Pydantic validation)
- No global state variables

### Agent Isolation
- Each agent in separate file: agents/<agent_name>.py
- Agent has clear input/output contract
- No side effects outside designated outputs
- Idempotent operations (re-runnable safely)

### State Management
- All state in TypedDict class
- Immutable state transitions
- State updates return new state dict
- No in-place modifications

---

## 4. Complete State Schema

```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class AutomationState(TypedDict):
    """
    Complete state schema for test automation workflow.
    Organized into three sections: Core Test Data, Conversation Context, Session Data.
    """
    
    # ===== SECTION 1: CORE TEST DATA =====
    
    # Input
    jira_file_path: str                         # Path to .txt file from tester's folder
    jira_raw_content: str                       # Full text content from Jira file
    
    # Parsing Output (Agent 1)
    parsed_steps: List[Dict]                    # [{step_num, action, target, expected}, ...]
    jira_metadata: Dict                         # {ticket_id, module, priority, reporter, ...}
    
    # Module Detection (Agent 2)
    detected_modules: List[str]                 # ["teststeps", "equipment"]
    confidence_scores: Dict[str, float]         # {"teststeps": 0.95, "equipment": 0.10}
    
    # Historical Analysis (Agent 3)
    similar_tests: List[Dict]                   # [{file, similarity_score, selectors}, ...]
    extracted_patterns: Dict                    # {wait_times, common_sequences, helper_functions}
    
    # Action Library Loading (Agent 4)
    available_methods: Dict[str, List[Dict]]    # {"teststeps": [{name, params, code}, ...]}
    method_signatures: Dict[str, str]           # {"selectTeststepByName": "name: string"}
    
    # Gap Analysis (Agent 5)
    missing_methods: List[Dict]                 # [{name, params, suggested_impl, confidence}, ...]
    method_coverage_percent: float              # 0.0 - 100.0
    
    # Test Generation (Agent 6)
    generated_test_code: str                    # Complete Cypress TypeScript test
    generation_metadata: Dict                   # {model_used, tokens, temperature, timestamp}
    
    # Quality Analysis (Agent 7)
    quality_score: int                          # 0-100
    quality_report: Dict                        # {cy_get_count, violations, metrics}
    violations: List[Dict]                      # [{type, line, message, severity}, ...]
    
    # Execution (Agent 8)
    execution_status: str                       # "passed" | "failed" | "pending" | "error"
    execution_time: float                       # Seconds (e.g., 87.3)
    screenshots: List[str]                      # [paths to screenshot files]
    video_path: Optional[str]                   # Path to video recording
    error_logs: List[str]                       # Error messages from Cypress
    
    # Reporting (Agent 9)
    report_path: Optional[str]                  # Path to generated HTML report
    report_generation_status: str               # "success" | "failed"
    
    
    # ===== SECTION 2: CONVERSATION CONTEXT =====
    
    conversation_history: List[Dict]            # [{role: "user"|"assistant", content, timestamp}, ...]
    current_workflow_stage: str                 # "parsing" | "generation" | "execution" | "complete"
    user_decisions: List[Dict]                  # [{decision_type, choice, timestamp, context}, ...]
    pending_approvals: List[Dict]               # [{method_name, status, attempts, last_result}, ...]
    workflow_errors: List[Dict]                 # [{agent, error, timestamp, resolved}, ...]
    
    
    # ===== SECTION 3: SESSION DATA =====
    
    session_id: str                             # Unique session identifier (UUID)
    user_id: str                                # Current tester ID
    user_preferences: Dict                      # {jira_folder, default_timeout, notification_prefs}
    project_config: Dict                        # Loaded from testproject_config.yaml
    timestamp_started: datetime                 # When workflow started
```

---

## 5. Agent Communication Protocol

### Rules
- Agents only read/write specific state fields (defined per agent)
- State updates must be immutable (return new dict, never modify in place)
- No global variables shared between agents
- No direct agent-to-agent calls (only through state)

### Field Access by Agent

**Agent 1 (Jira Parser):**
- Reads: jira_file_path, user_preferences
- Writes: jira_raw_content, parsed_steps, jira_metadata

**Agent 2 (Module Detector):**
- Reads: parsed_steps, jira_metadata, project_config
- Writes: detected_modules, confidence_scores

**Agent 3 (Historical Analyzer):**
- Reads: detected_modules, parsed_steps, project_config
- Writes: similar_tests, extracted_patterns

**Agent 4 (Action Library Loader):**
- Reads: detected_modules, project_config
- Writes: available_methods, method_signatures

**Agent 5 (Method Gap Analyzer):**
- Reads: parsed_steps, available_methods
- Writes: missing_methods, method_coverage_percent

**Agent 6 (Test Generator):**
- Reads: parsed_steps, available_methods, similar_tests, extracted_patterns
- Writes: generated_test_code, generation_metadata

**Agent 7 (Code Quality Analyzer):**
- Reads: generated_test_code, available_methods
- Writes: quality_score, quality_report, violations

**Agent 8 (Execution Agent):**
- Reads: generated_test_code, project_config
- Writes: execution_status, execution_time, screenshots, video_path, error_logs

**Agent 9 (Reporting Agent):**
- Reads: All core test data fields (for complete report)
- Writes: report_path, report_generation_status

### Example State Update (Immutable Pattern)

```python
def parse_jira(state: AutomationState) -> AutomationState:
    """Agent 1: Parse Jira file"""
    # Read input
    jira_path = state["jira_file_path"]
    
    # Process
    content = Path(jira_path).read_text()
    steps = parse_steps(content)
    metadata = extract_metadata(content)
    
    # Return NEW state dict (immutable)
    return {
        **state,  # Copy existing state
        "jira_raw_content": content,
        "parsed_steps": steps,
        "jira_metadata": metadata
    }
```

---

## 6. Error Handling Strategy

### Exception Hierarchy
```python
class TestAutomationError(Exception):
    """Base exception for all automation errors"""
    pass
    
class SelectorExtractionError(TestAutomationError):
    """Selector extraction failed"""
    pass
    
class ActionLibraryGenerationError(TestAutomationError):
    """Action library generation failed"""
    pass
    
class JiraParsingError(TestAutomationError):
    """Jira file parsing failed"""
    pass
    
class TestExecutionError(TestAutomationError):
    """Test execution failed"""
    pass
```

### Error Recovery
- **Retry transient failures:** 3 attempts with exponential backoff
- **Degrade gracefully:** If runtime extraction fails, continue with static + historical
- **User notification:** Clear error messages in chat
- **State preservation:** Save partial results before failure

### Retry Logic Example
```python
@retry(max_attempts=3, backoff_factor=2)
def call_azure_openai(prompt: str) -> str:
    """Call Azure OpenAI with retry logic"""
    # Implementation with exponential backoff
    pass
```

---

## 7. Project Structure

```
project_root/
├── backend/
│   ├── agents/              # One file per agent
│   │   ├── jira_parser.py
│   │   ├── module_detector.py
│   │   ├── historical_analyzer.py
│   │   ├── action_library_loader.py
│   │   ├── method_gap_analyzer.py
│   │   ├── test_generator.py
│   │   ├── code_quality_analyzer.py
│   │   ├── execution_agent.py
│   │   └── reporting_agent.py
│   ├── workflows/           # LangGraph workflows
│   │   ├── setup_workflow.py
│   │   └── testing_workflow.py
│   ├── models/              # Pydantic models
│   │   ├── state.py
│   │   ├── config.py
│   │   └── schemas.py
│   ├── api/                 # FastAPI routes
│   │   ├── admin.py
│   │   ├── tester.py
│   │   └── websocket.py
│   ├── services/            # Business logic
│   │   ├── auth.py
│   │   ├── file_manager.py
│   │   └── cypress_runner.py
│   ├── adapters/            # Framework adapters (for future)
│   │   ├── cypress_adapter.py
│   │   └── angular_parser.py
│   ├── utils/               # Helper functions
│   │   ├── logger.py
│   │   └── validators.py
│   └── main.py              # FastAPI app entry
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
│       ├── login.html
│       ├── admin_dashboard.html
│       └── tester_dashboard.html
├── data/                    # Generated during runtime
│   ├── selectors/
│   ├── action_libraries/
│   ├── generated_tests/
│   ├── reports/
│   └── users.json
├── tests/                   # Unit tests
│   ├── test_agents/
│   └── test_workflows/
├── spec.md
├── constitution.md
├── requirements.txt
├── .env                     # Environment variables (Azure OpenAI keys, etc.)
├── .env.example
└── README.md
```

---

## 8. File Organization Rules

### Maximum Sizes
- Python file: 500 lines max
- TypeScript file: 400 lines max
- If exceeding, split into logical modules

### Naming Conventions
- Files: snake_case.py
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE
- Private methods: _leading_underscore

---

## 9. Dependency Management

### Allowed Dependencies
- Standard library: unlimited
- LangChain/LangGraph: required
- FastAPI ecosystem: required
- Azure OpenAI: required
- Cypress/Playwright: required
- File parsing: python-docx, beautifulsoup4
- Utilities: pyyaml, pydantic

### Restricted Dependencies
- No heavy ML frameworks (TensorFlow, PyTorch) - not needed
- No database ORMs (SQLAlchemy) - using JSON files for PoC
- No unnecessary web frameworks beyond FastAPI
- Minimize dependencies for simplicity

### Version Pinning
- Pin all versions in requirements.txt
- Format: package==X.Y.Z
- Update only when necessary
- Test after any dependency update

---

## 10. Testing Requirements

### Unit Tests (Simplified for PoC)
- Core agents tested with sample inputs
- Focus on: Jira Parser, Module Detector, Test Generator, Historical Analyzer
- Coverage target: Key functionality working (not strict percentage)
- Use pytest framework
- Mock external dependencies where appropriate

### Test Naming
- Pattern: test_<feature>_<scenario>_<expected>
- Example: test_jira_parser_valid_file_returns_steps

---

## 11. Logging Standards

### Log Levels (Simplified)
- **INFO:** Agent starts/completes, major steps
- **ERROR:** Failures with context and stack trace

### Log Format
```python
logger.info(f"[{agent_name}] Action: {action} | Result: {result_status}")
```

### What to Log
- Agent invocations (start/complete)
- Errors with full context

### What NOT to Log
- Passwords or API keys
- Full file contents (log summaries only)
- Personal user data

---

## 12. Azure OpenAI Configuration

### Model Usage
- Primary: GPT-4o (gpt-4o deployment)
- Alternative: GPT-4.1 (gpt-4.1 deployment)
- Embeddings: text-embedding-3-small

**Note:** Use GPT-4o as primary model for test generation. GPT-4.1 available as alternative if needed.

### API Settings
- Temperature: 0.1 (low for consistency)
- Max tokens: 4000 (enough for test generation)
- Timeout: 60 seconds
- Retry: 3 attempts on failure

### Environment Variables

**Location:** `.env` file in project root directory

```
AZURE_OPENAI_API_KEY=<your_key>
AZURE_OPENAI_ENDPOINT=https://ai2ets.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_DEPLOYMENT_GPT41=gpt-4.1
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

**Note:** The `.env` file is in the base project folder and contains other configuration beyond Azure OpenAI settings. Never commit this file to version control.

---

## 13. Performance Requirements

### Response Times (Target)
- Admin setup (complete): <10 minutes
- Test generation: <10 seconds
- Test execution: 90-300 seconds (depends on test complexity)

---

## 14. Security Principles

### Authentication
- Simple username/password for PoC
- Passwords hashed with bcrypt
- Session tokens for web UI
- No JWT complexity needed for PoC

### Data Storage
- users.json: hashed passwords only
- .env file: gitignored, secrets never committed
- Local filesystem only (no cloud for PoC)

### Input Validation
- Validate all file paths (prevent directory traversal)
- Validate config YAML structure
- Sanitize user inputs in chat

---

## 15. Platform Compatibility

### Primary Target
- Windows 10/11
- Python 3.11+ installed
- Node.js 18+ installed (for Cypress)

### Path Handling
- Use pathlib for all file operations
- Handle Windows backslashes correctly
- Support UNC paths if needed
- Normalize paths before storage

### Process Execution
- Use subprocess for Cypress/npm commands
- Handle Windows command extensions (.cmd, .bat)
- Set proper working directories
- Capture stdout/stderr correctly

---

## 16. Code Documentation

### Module Docstrings
```python
"""
Module description: what this module does

Key classes:
- ClassName: brief description

Key functions:
- function_name: brief description
"""
```

### Function Docstrings
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
    """
```

---

## 17. Action Library Standards

### TypeScript Structure
```typescript
export class ModuleActions {
  // Category: Navigation
  navigateToModule(): void { }
  
  // Category: Selection
  selectByName(name: string): void { }
  
  // Category: Edit Operations
  editByName(name: string): void { }
  
  // Category: Verifications
  verifySuccess(message?: string): void { }
}
```

### Method Naming
- Pattern: `<action><target><modifier>`
- Examples: selectTeststepByName, fillPartField, verifySuccessMessage
- Consistent across all modules

### Selector Encapsulation
- All selectors hidden inside methods
- AI never sees raw selectors
- TypeScript compilation ensures method exists

### Wait Strategy
- Include waits inside methods
- Wait times based on historical patterns
- Document why specific wait times chosen

### Action Library Versioning & Updates

**Preservation Rule:** Never overwrite tester-approved methods during admin re-setup

**Method Metadata Format:**
```typescript
/**
 * Method description
 * @generated auto | tester_approved
 * @date 2025-10-06
 * @approver john.doe (if tester_approved)
 */
```

**Re-Generation Process:**
1. Load existing Action Libraries
2. Identify tester-approved methods (parse metadata)
3. Generate new methods from updated selectors
4. **Merge:** Keep approved methods, add/update auto-generated only
5. Backup old version to `./data/action_libraries/backups/`

**Admin sees:**
```
Action Library Update:
- Preserved: 12 tester-approved methods
- Added: 5 new methods
- Updated: 3 auto-generated methods
```

---

## 18. Helper Function Import Strategy

### Reusable Helpers
Action Libraries can import existing helper functions:

```typescript
import { login } from '../Helper_Functions/login'
import { verify } from '../Helper_Functions/verify'
import { clickOn } from '../Helper_Functions/buttons'
```

### When to Import vs Write New
- **Import:** Existing helper does exactly what's needed
- **Write new:** Need module-specific logic with selectors

### Example Action Library Method Using Helper
```typescript
export class CommonActions {
  /**
   * Login using existing helper function
   * Reuses tested login logic from Helper_Functions
   */
  login(): void {
    login.willAllPermission()  // Imported helper
    cy.wait(30000)
  }
}
```

### Available Helpers (from historical analysis)
- **login.ts:** willAllPermission(), bulkEditLogin()
- **verify.ts:** verifyStatusCodeOk(), elementVisible(), elementContainsText()
- **buttons.ts:** saveButton(), deleteButton(), filterButton(), createButton()
- **intercepts.ts:** Various API interception methods

---

## 19. Framework Compatibility & Adapters

### Current Implementation (PoC)
- **Hard-coded:** Angular + Cypress
- **Justified:** Prove Action Library pattern first, then extend

### Core Logic is Framework-Agnostic

**These components work for ANY framework:**
- Selector extraction algorithm (find attributes → save)
- Historical pattern mining algorithm (parse → count → analyze)
- Module detection (dictionary mapping)
- Method gap analysis (text matching)
- Action Library generation algorithm (group → generate → combine)
- LangGraph workflow (agent orchestration)

**Only these need adapters (15% of code):**
1. Syntax templates (method code generation)
2. File parsers (HTML vs JSX vs Vue)
3. Test execution commands
4. Result parsers

### Adapter Pattern for Extension

```python
# Core logic stays in agents/*.py (framework-agnostic)
# Framework-specific in adapters/*.py

class TestFrameworkAdapter(ABC):
    """Interface for framework-specific syntax"""
    
    @abstractmethod
    def generate_method(self, action_type: str, selector: str, wait: int) -> str:
        """Generate method code in framework-specific syntax"""
        pass
    
    @abstractmethod
    def execute_test(self, test_file: str) -> ExecutionResult:
        """Execute test using framework-specific command"""
        pass

class CypressAdapter(TestFrameworkAdapter):
    def generate_method(self, action_type: str, selector: str, wait: int) -> str:
        if action_type == "click":
            return f"cy.get('{selector}').click()\ncy.wait({wait})"
        # ... other action types
    
    def execute_test(self, test_file: str) -> ExecutionResult:
        cmd = ['npx', 'cypress', 'run', '--spec', test_file]
        result = subprocess.run(cmd, capture_output=True)
        return self.parse_result(result)

class PlaywrightAdapter(TestFrameworkAdapter):
    def generate_method(self, action_type: str, selector: str, wait: int) -> str:
        if action_type == "click":
            return f"await page.locator('{selector}').click()\nawait page.waitForTimeout({wait})"
        # ... other action types
    
    def execute_test(self, test_file: str) -> ExecutionResult:
        cmd = ['npx', 'playwright', 'test', test_file]
        result = subprocess.run(cmd, capture_output=True)
        return self.parse_result(result)

# Usage in agents
class ActionLibraryGenerator:
    def __init__(self, adapter: TestFrameworkAdapter):
        self.adapter = adapter
    
    def generate_method(self, action_type, selector, wait):
        # Core algorithm stays same
        return self.adapter.generate_method(action_type, selector, wait)
```

### Extension Points
- `agents/action_library_generator.py` uses `adapter.generate_method()`
- `agents/execution_agent.py` uses `adapter.execute_test()`
- Core logic unchanged, swap adapters per project

### Future Configuration
```yaml
# testproject_config.yaml
frontend_framework: "angular"  # or "react", "vue"
testing_framework: "cypress"   # or "playwright", "selenium"
```

System loads appropriate adapter based on config.

---

## 20. Notes for Developers

### Common Pitfalls
- Don't hardcode file paths (use config)
- Don't mix agent responsibilities
- Don't skip error handling
- Don't forget to log important operations
- Don't use localStorage in artifacts (not supported)

### Best Practices
- Start simple, refactor later
- Test edge cases
- Document decisions
- Keep functions small and focused
- Use type hints everywhere

---

**This constitution guides all development decisions. When in doubt, refer back to these principles.**