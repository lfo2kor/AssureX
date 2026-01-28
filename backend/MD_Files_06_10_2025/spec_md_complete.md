# AI-Powered Test Automation Tool - Specification

## Document Overview

**Version:** 1.0  
**Date:** 2025-10-06  
**Status:** Initial specification for PoC development  
**Purpose:** Define requirements for AI-powered test automation system using multi-agent architecture

---

## 1. Project Purpose

Build a web-based Windows application that automates end-to-end testing of Angular web applications using AI agents (LangChain + LangGraph). Tests are generated from Jira tickets using the Action Library pattern to achieve 0% selector invention rate.

### Key Innovation
Action Library pattern: AI uses high-level methods instead of raw selectors, eliminating hallucinated selectors completely.

### Target Users
- **Test Admin:** Configure projects, manage system
- **Testers:** Generate and execute tests from Jira tickets

---

## 2. Technology Stack

### Backend
- Python 3.11+ with FastAPI
- LangChain + LangGraph (multi-agent orchestration)
- Azure OpenAI (GPT-4o, GPT-4.1, embeddings)
- Playwright (runtime selector extraction)

### Frontend
- HTML5 + CSS3 + JavaScript
- WebSocket (real-time chat communication)
- Runs in browser on localhost:5000

### Testing Framework
- Cypress 13.0+ (test execution)
- TypeScript 5.0+ (Action Libraries)

### Deployment
- Local Windows laptop (PoC scope)
- No cloud, no containers
- Access via browser: localhost:5000

---

## 3. Technology Stack Constraints

### PoC Scope
- **Angular web applications only** - Static extraction scans .html files
- **Cypress testing framework only** - Action Libraries generate Cypress syntax
- **Reason:** Framework abstraction deferred to post-PoC phase

### Why This Constraint
Proving the Action Library pattern works is the PoC goal. Multi-framework support adds 3-4 weeks development time and risks over-engineering before validation.

### Extension Strategy
- Design includes adapter pattern for future frameworks (React, Vue, Playwright, Selenium)
- Core logic (selector extraction algorithm, pattern mining, gap analysis) is framework-agnostic
- Only syntax generation and file parsing need adapters
- See constitution.md "Framework Compatibility" section for technical details

### Scalability Assessment
- **100% scalable:** Any Angular + Cypress project (different modules, domains, complexity)
- **Requires adapters:** React/Vue projects, Playwright/Selenium frameworks
- **Requires redesign:** Mobile apps, desktop apps, non-web automation

---

## 4. User Roles & Access

### Test Admin
**Access:** Admin credentials, full system access

**Capabilities:**
- Create and configure test projects
- Run initial setup (selector extraction, Action Library generation)
- Register tester accounts
- Monitor system health
- View usage metrics

### Tester
**Access:** Tester credentials, testing functions only

**Capabilities:**
- Submit Jira tickets for test generation
- Review AI-generated test code
- Approve/modify Action Library methods
- Execute tests
- Download test reports
- Configure personal settings (Jira folder path)

---

## 5. Configuration File Structure

### testproject_config.yaml

```yaml
project:
  id: "plcd"                          # Project identifier (used in file paths)
  name: "PLCD Test Automation"
  web_url: "http://localhost:4200"

runtime_extraction:
  enabled: true
  login:
    username: "mechanic"
    password: "avalon"
    wait_after_login_ms: 30000
  navigation:
    wait_after_navigation_ms: 15000
  module_routes:
    teststeps:
      - route: "/teststeps"
    equipment:
      - route: "/equipment"

codebase:
  path: "C:/Projects/PLCD/cri-webapp/client/src/app"
  modules:
    - code_name: "teststeps"
      ui_names: ["Teststep", "Teststeps", "Runs"]
      jira_keywords: ["teststep", "test step", "runs"]
      paths: ["teststeps"]
    - code_name: "equipment"
      ui_names: ["Equipment", "TestEquipment"]
      jira_keywords: ["equipment", "test equipment"]
      paths: ["equipment", "equipment-calendar"]
  common_components:
    paths: ["shared", "shared-services"]

historical_tests:
  path: "C:/Projects/PLCD/tests/cypress/e2e"
  helper_functions_path: "C:/Projects/PLCD/tests/cypress/Helper_Functions"

output:
  selectors_path: "./data/selectors"
  action_libraries_path: "./data/action_libraries"
  generated_tests_path: "./data/generated_tests"
  reports_path: "./data/reports"
```

### What Admin Must Fill

**Project Information:**
- Project name and web application URL
- Test credentials for automation
- Wait times based on application performance

**Codebase Mapping:**
- Path to Angular application source code
- Module definitions (code name, UI names, Jira keywords, folder paths)
- Common components paths

**Historical Context:**
- Path to existing Cypress test files
- Path to helper function files

**Module Mapping Rules:**
- `code_name`: Internal folder name in codebase
- `ui_names`: How module appears in UI navigation
- `jira_keywords`: Keywords in Jira tickets that indicate this module
- `paths`: Folder paths within codebase for this module

---

## 6. Pre-Setup Validation (Phase 0)

Before starting the 7-10 minute setup process, system validates all configuration assumptions in 30-60 seconds.

### Check 1: Configuration File Validation
- YAML structure is valid
- All required fields present
- Field values are correct types

**If fails:**
```
Configuration Error
Issue: codebase.path does not exist
Fix: Update path in testproject_config.yaml
```

### Check 2: File System Access
- Codebase path exists and is readable
- Historical tests path exists
- Helper functions path exists
- Output directories can be created

### Check 3: Codebase Structure Validation
- Looks like Angular app (app.component.ts or app.module.ts exists)
- Module paths exist within codebase
- HTML files exist (at least 1 found)

### Check 4: Historical Tests Validation
- At least one .cy.ts file exists
- At least one .ts file exists (helpers)
- Sample test file is valid TypeScript

### Check 5: Helper Functions Validation
- Expected helper files exist (login.ts, verify.ts recommended)
- Files are valid TypeScript

### Check 6: Web Application Accessibility
- URL is reachable
- Application responds within 10 seconds

### Check 7: Credentials Test
- Launch headless browser
- Navigate to login page
- Enter credentials
- Verify login succeeds

### Check 8: Required Dependencies
- Node.js installed
- Cypress installed
- Playwright installed
- TypeScript compiler available

### Validation Summary Screen

After all checks pass:
```
Pre-Setup Validation Complete

Configuration:
✓ YAML structure valid
✓ All paths accessible
✓ Codebase found: 1,247 HTML files
✓ Historical tests found: 247 .cy.ts files
✓ Helper functions found: 8 files
✓ Web application accessible
✓ Login credentials verified
✓ All dependencies installed

Modules detected:
- teststeps (142 HTML files)
- equipment (89 HTML files)
- common (1,016 shared files)

Estimated setup time: 8-10 minutes

[Start Setup] [Review Config] [Cancel]
```

### Validation Failure Workflow
If validation fails:
1. Show specific errors with actionable fixes
2. Admin corrects configuration file
3. Admin clicks "Start Setup" again
4. Validation runs again from beginning
5. Only proceeds when all critical checks pass

---

## 7. Test Admin Workflow - Complete Setup Process

### Phase 1: Configuration (5 minutes)
Admin creates testproject_config.yaml with all required information.

### Phase 2: Automated Setup Process (7-10 minutes)

Admin clicks "Start Setup" → System executes 5 phases automatically.

#### Step 1: Static Selector Extraction (2 min)
**Agent: Selector Extractor Agent**

**Process:**
1. Scan codebase path from config
2. Find all .html files in module paths
3. Extract all [data-*] attributes using regex
4. Tag each selector with source module
5. Save to data/selectors/static_selectors.json

**Progress shown:** "Extracting selectors from HTML files... 342 selectors found"

#### Step 2: Historical Test Mining (3-4 min)
**Agent: Historical Analyzer Agent**

**Process:**
1. Scan historical_tests.path for .cy.ts files
2. Parse each test file for selectors, waits, patterns
3. Scan helper_functions_path for .ts files
4. Build pattern database
5. Save to data/selectors/historical_patterns.json

**Output includes:**
- wait_patterns (after_login: 30000ms, etc.)
- force_click_elements
- helper_functions (login, verify, clickOn)
- common_sequences

**Progress shown:** "Mining 247 historical test files..."

#### Step 3: Runtime Selector Extraction (5 min)
**Agent: Selector Extractor Agent (runtime mode)**

**Process:**
1. Launch Playwright browser
2. Navigate to project.web_url
3. Auto-login
4. For each module: navigate, capture DOM
5. Extract dynamic [data-*] attributes
6. Save to data/selectors/runtime_selectors.json

**Progress shown:** "Extracting runtime selectors... Visiting teststeps module..."

#### Step 4: Merge & Deduplicate (30 sec)
**Agent: Selector Extractor Agent**

**Process:**
1. Load all selector sources
2. Merge and remove duplicates
3. Calculate coverage percentage
4. Save to data/selectors/common_selectors.json

#### Step 5: Generate Action Libraries (1-2 min)
**Agent: Action Library Generator Agent**

**Process:**
1. Load common_selectors.json and historical_patterns.json
2. Generate common_actions.ts (login, verify, etc.)
3. For each module, generate <module>_actions.ts
4. Import helper functions where appropriate
5. Apply wait patterns from historical data
6. Compile TypeScript to validate

**Output Example:**
```typescript
import { login } from '../Helper_Functions/login'

export class CommonActions {
  login(): void {
    login.willAllPermission()
    cy.wait(30000)
  }
}

export class TeststepsActions {
  navigateToTeststeps(): void {
    cy.get('[data-test="sidebar-nav-item-nav_item_teststeps"]')
      .contains('Runs').click()
    cy.wait(15000)
  }
  
  selectTeststepByName(name: string): void {
    cy.get('[data-commandbar] [data-test="commandbar-search-input"]')
      .type(name)
    cy.wait(1000)
    cy.get(`[data-attribute="${name}"]`).click({force: true})
    cy.wait(15000)
  }
}
```

#### Step 6: Validation & Summary (30 sec)

**Admin sees:**
```
Setup Complete!

Summary:
- Selectors extracted: 342 (96.5% coverage)
- Historical patterns mined from 247 test files
- Helper functions identified: 23 methods
- Action Libraries generated: 3 files
  - common_actions.ts: 8 methods
  - teststeps_actions.ts: 18 methods
  - equipment_actions.ts: 15 methods

System ready for test generation!
```

---

## 8. Tester Workflow - Complete Scenarios

### Scenario A: All Methods Exist (2-3 minutes)

**Step 1:** Tester types: `"Test RBPLCD-8835"`

**Step 2:** Agent Processing (5-10 seconds)

**Agent Flow:**
1. **Jira Parser:** Reads .txt file, extracts steps
2. **Module Detector:** Maps "Teststep" → teststeps
3. **Historical Analyzer:** Finds similar tests
4. **Action Library Loader:** Loads common + teststeps methods
5. **Method Gap Analyzer:** Maps each step to method
   - Login → common.login() ✓
   - Navigate → teststeps.navigateToTeststeps() ✓
   - Select → teststeps.selectTeststepByName() ✓
   - (all 8 steps mapped) ✓

**Tester sees:** "All required methods available! Generating test..."

**Step 3:** Test Generation (5 seconds)

**Agent Flow:**
6. **Test Generator:** Calls Azure OpenAI, generates Cypress test
7. **Code Quality Analyzer:** Validates code, calculates score

**Generated Code:**
```typescript
import { CommonActions } from '../actions/common_actions'
import { TeststepsActions } from '../actions/teststeps_actions'

const common = new CommonActions()
const teststeps = new TeststepsActions()

describe('[RBPLCD-8835] Edit part details', () => {
  it('should edit part type successfully', () => {
    common.login()
    teststeps.navigateToTeststeps()
    teststeps.selectTeststepByName('default_Measurement01')
    teststeps.openPartsAccordion()
    teststeps.editPartByName('default_testobject_01')
    teststeps.selectTypeFromDropdown('Type 5')
    teststeps.saveEdit()
    common.verifySuccessMessage("Successfully edited")
  })
})

Quality Score: 100/100
[Execute Test] [Download Code]
```

**Step 4:** Tester clicks "Execute Test" (90 seconds)

**Agent Flow:**
8. **Execution Agent:** Runs Cypress, captures results
9. **Reporting Agent:** Generates HTML report

**Tester sees:** "Test PASSED (87.3 seconds) [Download Report]"

### Scenario B: Missing Methods (10-15 minutes first time)

**Step 1:** Tester types: `"Test RBPLCD-9001"`

**Step 2:** Method Gap Detection

Agent 5 detects missing methods:
```
Missing Methods Detected (4 methods)

Method 1/4: clickExportButton()
Suggested implementation...
Confidence: 85%

[Test & Approve] [Approve Without Testing] [Modify] [Reject]
```

**Step 3:** Tester clicks "Test & Approve"

System generates mini test, executes, shows video.

**Step 4:** Tester approves, method added to library

Repeat for remaining 3 methods.

**Step 5:** After all approved, generate and execute full test

**Future benefit:** Next tester finds all methods ready.

---

## 9. Agent Architecture

### Agent 1: Jira Parser Agent
- **Responsibility:** Parse .txt Jira files, extract structured steps
- **Input:** jira_file_path, user_preferences
- **Output:** jira_raw_content, parsed_steps, jira_metadata
- **Code Size:** ~150 lines

### Agent 2: Module Detector Agent
- **Responsibility:** Identify module(s) from Jira content
- **Input:** parsed_steps, jira_metadata, project_config
- **Output:** detected_modules, confidence_scores
- **Code Size:** ~200 lines

### Agent 3: Historical Analyzer Agent
- **Responsibility:** Mine historical tests and helpers for patterns
- **Input:** detected_modules, parsed_steps, project_config
- **Output:** similar_tests, extracted_patterns, helper_functions
- **Code Size:** ~300 lines

### Agent 4: Action Library Loader Agent
- **Responsibility:** Load and parse Action Library methods
- **Input:** detected_modules, project_config
- **Output:** available_methods, method_signatures
- **Code Size:** ~150 lines

### Agent 5: Method Gap Analyzer Agent
- **Responsibility:** Check method availability, identify missing
- **Input:** parsed_steps, available_methods
- **Output:** missing_methods, method_coverage_percent
- **Code Size:** ~200 lines

### Agent 6: Test Generator Agent
- **Responsibility:** Generate Cypress test using Action Libraries
- **Input:** parsed_steps, available_methods, similar_tests, patterns
- **Output:** generated_test_code, generation_metadata
- **Code Size:** ~300 lines

### Agent 7: Code Quality Analyzer Agent
- **Responsibility:** Analyze code, calculate quality score
- **Input:** generated_test_code, available_methods
- **Output:** quality_score, quality_report, violations
- **Code Size:** ~200 lines

### Agent 8: Execution Agent
- **Responsibility:** Run Cypress test, capture results
- **Input:** generated_test_code, project_config
- **Output:** execution_status, execution_time, screenshots, video_path, error_logs
- **Code Size:** ~250 lines

### Agent 9: Reporting Agent
- **Responsibility:** Generate HTML report
- **Input:** All core test data fields
- **Output:** report_path, report_generation_status
- **Code Size:** ~300 lines

---

## 10. State Schema Definition

```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class AutomationState(TypedDict):
    # Core Test Data
    jira_file_path: str
    jira_raw_content: str
    parsed_steps: List[Dict]
    jira_metadata: Dict
    detected_modules: List[str]
    confidence_scores: Dict[str, float]
    similar_tests: List[Dict]
    extracted_patterns: Dict
    available_methods: Dict[str, List[Dict]]
    method_signatures: Dict[str, str]
    missing_methods: List[Dict]
    method_coverage_percent: float
    generated_test_code: str
    generation_metadata: Dict
    quality_score: int
    quality_report: Dict
    violations: List[Dict]
    execution_status: str
    execution_time: float
    screenshots: List[str]
    video_path: Optional[str]
    error_logs: List[str]
    report_path: Optional[str]
    report_generation_status: str
    
    # Conversation Context
    conversation_history: List[Dict]
    current_workflow_stage: str
    user_decisions: List[Dict]
    pending_approvals: List[Dict]
    workflow_errors: List[Dict]
    
    # Session Data
    session_id: str
    user_id: str
    user_preferences: Dict
    project_config: Dict
    timestamp_started: datetime
```

---

## 11. LangGraph Workflow

```
START → Jira Parser → Module Detector → Historical Analyzer 
→ Action Library Loader → Method Gap Analyzer 
→ [Missing Methods? YES → Approval Workflow | NO → Test Generator] 
→ Code Quality Analyzer → Execution Agent → Reporting Agent → END
```

### Conditional Edges
- After Agent 5: if missing_methods → Approval Workflow, else → Agent 6
- Approval Loop: for each missing method, get tester decision
- After Agent 8: if error and retry_count < 1 → retry, else → Agent 9

---

## 12. Jira .txt File Format & Parsing Rules

### File Structure
```
[RBPLCD-8835] edit part details
Status: Open
Project: RB-PLCD
Component/s: Teststep

Steps to Reproduce:
1. Login
2. navigate to teststep
3. click on teststep named as default_Measurement01
...

Acceptance Criteria:
"Successfully edited" message should be displayed
```

### Parsing Logic
1. Extract ticket ID: `\[([A-Z]+-\d+)\]`
2. Extract module: Component/s field → map using config
3. Extract steps: Find numbered lines `^\d+\.\s+(.*)`
4. Extract targets: Text in quotes `["']([^"']+)["']`
5. Extract expected results: Acceptance Criteria section

---

## 13. Historical Test Mining Rules

### From Test Files (.cy.ts)
- Extract cy.get() selectors
- Extract cy.wait() timings
- Detect cy.click({force: true}) patterns
- Map wait times to action types

### From Helper Files (.ts)
- Extract class and method names
- Build import path mappings
- Identify reusable functions

### Output Format
```json
{
  "wait_patterns": {"after_login": 30000, ...},
  "force_click_elements": ["[data-test='...']", ...],
  "helper_functions": {"login": ["willAllPermission"], ...}
}
```

---

## 14. Action Library Generation Rules

### Method Naming Convention
Pattern: `<verb><target><modifier>`
Examples: navigateToTeststeps(), selectTeststepByName(name: string)

### Implementation Rules
- Include waits from historical patterns
- Use force clicks where historical data shows necessity
- Parameterize dynamic values
- Reuse helper functions

### Example
```typescript
import { login } from '../Helper_Functions/login'

export class TeststepsActions {
  navigateToTeststeps(): void {
    cy.get('[data-test="sidebar-nav-item-nav_item_teststeps"]')
      .contains('Runs').click()
    cy.wait(15000)  // From historical patterns
  }
  
  selectTeststepByName(name: string): void {
    cy.get('[data-commandbar] [data-test="commandbar-search-input"]')
      .type(name)
    cy.wait(1000)
    cy.get(`[data-attribute="${name}"]`)
      .click({force: true})  // From historical patterns
    cy.wait(15000)
  }
}
```

---

## 15. Selector Extraction: Three Methods

### Method 1: Static HTML Parsing
- Scan .html files
- Extract data-* attributes
- Coverage: ~80%
- Time: 1-2 minutes

### Method 2: Historical Test Mining
- Parse .cy.ts files
- Extract cy.get() selectors
- Coverage: +5%
- Time: 30 seconds

### Method 3: Runtime Extraction
- Launch browser
- Auto-login and navigate
- Capture rendered DOM
- Coverage: +10-15%
- Time: 5 minutes

### Combined Result
Total coverage: 95%+, saved to common_selectors.json

---

## 16. Test Code Generation

### AI Prompt Strategy
**System:** "Use ONLY methods from Action Libraries. NEVER use cy.get() directly."
**User:** Jira content + available methods + strict rules

**Configuration:**
- Model: GPT-4o
- Temperature: 0.1
- Max tokens: 4000

### Quality Analysis
- Count cy.get() occurrences: Should be 0
- Verify all methods exist
- Calculate quality score: 0-100
- Target: 100/100 with 0 invented selectors

---

## 17. Test Execution

**Process:**
1. Write test to cypress/e2e/<ticket_id>.cy.ts
2. Run: `npx cypress run --spec <file> --headless --browser electron`
3. Capture: exit code, timing, logs, video, screenshots
4. Parse Cypress output for results

---

## 18. HTML Report Generation

**Sections:**
- Header (ticket ID, timestamp)
- Status banner (PASSED/FAILED)
- Summary (status, time, quality score)
- Test steps table
- Generated code
- Execution logs
- Video path

**Features:** Self-contained HTML, syntax highlighting, downloadable

**Filename:** `<ticket_id>_execution_report_<YYYYMMDD_HHMMSS>.html`

---

## 19. Performance Targets

| Operation | Target Time |
|-----------|-------------|
| Admin setup | 8-10 minutes |
| Test generation | <10 seconds |
| Test execution | 90-300 seconds |

---

## 20. Success Metrics

- Selector invention rate: 0%
- Quality score: ≥90/100
- Test generation success: ≥95%
- First-run pass rate: ≥80%

---

## 21. Data Storage Schema

### users.json
Location: `./data/users.json`

```json
{
  "users": [
    {
      "id": "uuid-string",
      "username": "admin",
      "password_hash": "$2b$12$...",
      "role": "admin",
      "created_at": "2025-10-06T10:00:00",
      "preferences": {}
    },
    {
      "id": "uuid-string",
      "username": "john.doe",
      "password_hash": "$2b$12$...",
      "role": "tester",
      "created_at": "2025-10-06T11:00:00",
      "preferences": {
        "jira_folder_path": "C:/Users/john.doe/Documents/Jira_Tickets",
        "default_timeout": 30000,
        "video_recording": true
      }
    }
  ]
}
```

### sessions.json (if WebSocket sessions need persistence)
Location: `./data/sessions.json`

```json
{
  "sessions": [
    {
      "session_id": "uuid-string",
      "user_id": "uuid-string",
      "started_at": "2025-10-06T14:30:00",
      "last_active": "2025-10-06T14:35:00",
      "workflow_state": "execution",
      "current_ticket": "RBPLCD-8835"
    }
  ]
}
```

### Project Config Storage
**PoC Limitation:** Single project only

Config file: `./testproject_config.yaml` (single file in root)

**Future:** Multi-project support would use `./configs/<project_name>_config.yaml`

---

## 22. WebSocket Protocol Specification

### Message Format
All WebSocket messages use JSON format:

```json
{
  "type": "message_type",
  "payload": { ... },
  "timestamp": "ISO-8601 string",
  "session_id": "uuid-string"
}
```

### Message Types

**From Client (Tester/Admin):**
```json
{
  "type": "user_input",
  "payload": {
    "text": "Test RBPLCD-8835"
  }
}

{
  "type": "method_decision",
  "payload": {
    "method_name": "clickExportButton",
    "decision": "approve" | "reject" | "modify" | "test_and_approve"
  }
}

{
  "type": "execute_test",
  "payload": {
    "ticket_id": "RBPLCD-8835"
  }
}
```

**From Server (System):**
```json
{
  "type": "system_response",
  "payload": {
    "text": "Generating test...",
    "show_typing": true
  }
}

{
  "type": "progress_update",
  "payload": {
    "stage": "execution",
    "progress": 75,
    "message": "Running test... (67s elapsed)"
  }
}

{
  "type": "code_generated",
  "payload": {
    "code": "import { CommonActions } from ...",
    "quality_score": 100,
    "ticket_id": "RBPLCD-8835"
  }
}

{
  "type": "test_result",
  "payload": {
    "status": "passed",
    "execution_time": 87.3,
    "report_path": "./data/reports/RBPLCD-8835_..."
  }
}

{
  "type": "error",
  "payload": {
    "error_type": "execution_timeout",
    "message": "Test execution exceeded 360 seconds",
    "recovery_options": ["retry", "cancel"]
  }
}

{
  "type": "method_suggestion",
  "payload": {
    "method_name": "clickExportButton",
    "suggested_code": "...",
    "confidence": 0.85,
    "actions": ["test_and_approve", "approve", "modify", "reject"]
  }
}
```

### Connection Handling

**Initial Connection:**
1. Client connects to `ws://localhost:5000/ws`
2. Server sends `{"type": "connected", "payload": {"session_id": "..."}}`
3. Client stores session_id for reconnection

**Disconnection:**
- If client disconnects during test execution, execution continues
- On reconnection: Server sends current workflow state
- Client can resume from last known state

**Timeout:**
- WebSocket idle timeout: 30 minutes
- Send ping/pong every 60 seconds to keep alive

**No Fallback:** PoC uses WebSocket only (no polling fallback)

---

## 23. Error Recovery & User-Facing Messages

### Setup Errors

**Historical Mining Fails:**
```
⚠️ Historical Test Mining Failed
Issue: No .cy.ts files found in specified path
Impact: Wait times will use defaults, not historical averages
Options:
  [Continue Without Historical Patterns]
  [Update Path and Retry]
  [Cancel Setup]
```

**Runtime Extraction Fails:**
```
❌ Runtime Extraction Failed
Issue: Cannot connect to http://localhost:4200
Possible causes: Application not running, incorrect URL, network issue
Impact: Some dynamic selectors may be missed (coverage ~85% instead of 95%)
Options:
  [Continue Without Runtime Extraction]
  [Fix Application and Retry]
  [Cancel Setup]
```

**Action Library Compilation Error:**
```
❌ TypeScript Compilation Failed
Issue: Syntax error in teststeps_actions.ts line 45
Error: Expected ';' after method declaration
Options:
  [Regenerate Action Library]
  [View Full Error Log]
  [Cancel Setup]
```

### Generation Errors

**Azure OpenAI Unavailable:**
```
❌ Test Generation Failed
Issue: Azure OpenAI service unavailable (HTTP 503)
Retrying in 5 seconds... (Attempt 1/3)
Options:
  [Wait for Retry]
  [Cancel]
```

**Token Limit Exceeded:**
```
⚠️ Test Too Complex
Issue: Generated test exceeds 4000 token limit
The Jira ticket has 25 steps, which is too many for single test generation
Suggestion: Split into multiple smaller tests or simplify steps
Options:
  [Retry with Simplified Prompts]
  [Cancel]
```

### Execution Errors

**Test Timeout:**
```
⚠️ Test Execution Timeout
Issue: Test exceeded 360 seconds limit
Last step completed: "Open parts accordion"
Options:
  [Retry with 600s Timeout]
  [Download Partial Results]
  [Cancel]
```

**Cypress Crash:**
```
❌ Test Execution Failed
Issue: Cypress process crashed
Error: Browser disconnected unexpectedly
Logs saved to: ./logs/execution_error_20251006.log
Options:
  [Retry Test]
  [View Error Log]
  [Cancel]
```

**Application Not Accessible:**
```
❌ Cannot Access Application
Issue: Connection refused to http://localhost:4200
Ensure the application is running before executing tests
Options:
  [Retry (Check Application)]
  [Cancel]
```

### Recovery Actions
- All errors show clear cause and impact
- Provide actionable recovery options
- Save error context to logs for debugging
- Never silent failures - always notify user

---

## 24. Action Library Versioning & Update Strategy

### Preservation of Tester-Approved Methods

**Critical Rule:** Never overwrite tester-approved methods during re-setup

### Method Metadata
Each method in Action Library has metadata comment:

```typescript
export class TeststepsActions {
  /**
   * Navigate to Teststeps module
   * @generated auto (2025-10-06)
   * @source static_selectors
   */
  navigateToTeststeps(): void { }
  
  /**
   * Click export button
   * @generated tester_approved (2025-10-08 by john.doe)
   * @tested mini_test_passed
   * @confidence 0.95
   */
  clickExportButton(): void { }
}
```

### Admin Re-runs Setup

**Process:**
1. Load existing Action Library files
2. Parse and extract tester-approved methods (check `@generated tester_approved`)
3. Generate new methods from updated selectors
4. **Merge strategy:**
   - Keep all tester-approved methods unchanged
   - Add new auto-generated methods
   - Update auto-generated methods if selector changed
   - Mark conflicts for admin review

**Admin sees summary:**
```
Action Library Update Complete

Preserved:
- 12 tester-approved methods (unchanged)

Added:
- 5 new methods from updated selectors

Updated:
- 3 auto-generated methods (selectors changed)

Conflicts:
- None
```

### Version Backup
Before any re-generation, backup existing library:
```
./data/action_libraries/backups/
  teststeps_actions_20251006_140522.ts
  teststeps_actions_20251008_093011.ts (current)
```

### Single Project Limitation (PoC)
- PoC supports ONE project only
- Config: `./testproject_config.yaml`
- Action Libraries: `./data/action_libraries/`
- No project selection UI needed

**Future:** Multi-project would use:
- Configs: `./configs/<project>_config.yaml`
- Libraries: `./data/<project>/action_libraries/`

---

## 25. Out of Scope (PoC)

Not included: Multi-user concurrency, cloud deployment, CI/CD integration, mobile responsive UI, database, API endpoints, scheduled execution, real-time collaboration, email/Slack notifications, analytics dashboard.

**Reason:** PoC goal is proving Action Library pattern works. These are post-PoC enhancements.

---

**END OF SPECIFICATION**