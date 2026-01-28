# AI-Powered Test Automation with Feedback Loop - Complete Specification v2.0

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Document Title** | AI-Powered Test Automation with Feedback Loop - Complete Specification |
| **Version** | 2.0 |
| **Date** | 2025-10-30 |
| **Status** | In Development |
| **Previous Version** | v1.0 (spec_UPDATED.md - 3-agent system) |
| **Author** | AI Test Automation Team |
| **Classification** | Internal - Technical Specification |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-20 | Team | Initial 3-agent system with L1/L2/L3 strategy |
| 2.0 | 2025-10-30 | Team | Added Feedback Agent, interactive failure correction |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What's New in v2.0](#2-whats-new-in-v20)
3. [System Architecture](#3-system-architecture)
4. [Agent Specifications](#4-agent-specifications)
5. [Feedback Agent (NEW)](#5-feedback-agent-detailed-specification)
6. [Failure Handling System](#6-failure-handling-system)
7. [Configuration Files](#7-configuration-files)
8. [Input Requirements](#8-input-requirements)
9. [User Workflows](#9-user-workflows)
10. [State Management](#10-state-management)
11. [Implementation Guide](#11-implementation-guide)
12. [Migration Guide](#12-migration-guide-v10--v20)
13. [Success Metrics](#13-success-metrics)
14. [Known Limitations](#14-known-limitations)
15. [API Reference](#15-api-reference)
16. [Appendices](#16-appendices)

---

# 1. Executive Summary

## 1.1 What This System Does

This is a **Python-based intelligent test automation system** that executes functional tests from Jira tickets using a **4-agent architecture** with **interactive feedback loop**:

**Core Agents:**
1. **Jira Parser Agent** - Extracts test steps from Jira tickets
2. **Vision Executor Agent** - Executes tests using 3-level selector strategy (L1/L2/L3)
3. **Feedback Agent** - Collects corrections when tests fail (NEW in v2.0)
4. **Report Generator Agent** - Creates comprehensive HTML reports

**Key Innovation (v2.0):** **Human-in-the-Loop Learning**

Instead of requiring manual selector discovery and maintenance, the system:
- ✅ Detects failures automatically
- ✅ Presents interactive Element Picker for visual selection
- ✅ Validates corrections on live browser
- ✅ Updates configuration files automatically
- ✅ Learns patterns from corrections
- ✅ Retries failed steps automatically

## 1.2 Key Benefits

| Benefit | v1.0 | v2.0 | Improvement |
|---------|------|------|-------------|
| **Test Execution** | 30-45 seconds | 30-45 seconds | Same (no degradation) |
| **Failure Resolution** | 15-30 minutes (manual) | **2-5 minutes** | **6-10x faster** |
| **Selector Discovery** | Manual DevTools | **Visual Element Picker** | **Intuitive** |
| **Learning Capability** | None | **Automatic pattern learning** | **New feature** |
| **Audit Trail** | Logs only | **Complete feedback history** | **New feature** |
| **Test Accuracy** | 99%+ | 99%+ | Maintained |

## 1.3 Target Users

**Primary:** QA Testers (non-technical users)
- Submit Jira ticket number
- System executes test automatically
- If failures occur, Element Picker guides correction
- System learns and improves

**Secondary:** Test Automation Engineers
- Configure selectors for optimal performance
- Review feedback history to identify patterns
- Extend system with custom patterns

## 1.4 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.11+ |
| **Orchestration** | LangChain + LangGraph | Latest |
| **AI/Vision** | Azure OpenAI GPT-4o | 2024-02-15 |
| **Browser Automation** | Playwright | Latest |
| **State Management** | Pydantic | Latest |
| **Templating** | Jinja2 | Latest |
| **Config Format** | YAML + JSON | Standard |

---

# 2. What's New in v2.0

## 2.1 Major Features Added

### 2.1.1 Feedback Agent (Primary Addition)

**Purpose:** Enable testers to correct failures interactively without code changes

**Capabilities:**
- Detects and categorizes all failure types
- Presents failures with visual context (screenshots)
- Collects corrections via interactive CLI
- Validates corrections on live browser
- Updates configuration files automatically
- Triggers retry of failed steps

**User Experience:**
```
Test fails → Feedback Agent activates → Element Picker shown
→ Tester clicks element → Selector captured → Auto-validated
→ Saved to config → Test retries → Success!
```

**Time Savings:** 15-30 minutes → 2-5 minutes per failure

### 2.1.2 Interactive Element Picker

**Purpose:** Visual selector discovery without DevTools knowledge

**How It Works:**
- JavaScript injected into browser
- Overlay with instructions shown
- Tester hovers over elements (highlighted)
- Tester clicks target element
- System extracts all possible selectors
- Tester chooses best selector
- Validation happens automatically

**Benefits:**
- ✅ No DevTools knowledge required
- ✅ No manual selector writing
- ✅ Visual feedback (element highlighted)
- ✅ Instant validation
- ✅ Error-proof (can't enter invalid selector)

### 2.1.3 Intelligent Failure Diagnostics

**Purpose:** Categorize failures to provide specific guidance

**Failure Categories:**
1. **Selector Issues** (selector not found, ambiguous, wrong element)
2. **Element State Issues** (not visible, not clickable, disabled)
3. **Timing Issues** (slow loading, animation delay)
4. **Action Execution Issues** (dropdown didn't open, value not in list)
5. **Context Issues** (wrong page, modal blocking, iframe)
6. **Verification Issues** (message not shown, wrong message)

**For Each Category:**
- Specific diagnostic data collected
- Tailored questions asked
- Recommended fix suggested
- Auto-fix attempted when possible

### 2.1.4 Configuration-Based Learning

**Purpose:** System learns from corrections and improves over time

**What Gets Learned:**
- Row scoping patterns (how to identify specific table rows)
- Timing adjustments (wait times for slow elements)
- Selector aliases (multiple names for same element)
- Common failure patterns

**Storage:** `Project_Config/feedback_rules.json`

**Auto-Application:** Learned rules applied automatically in future tests

### 2.1.5 Complete Feedback History

**Purpose:** Audit trail and analytics

**Stored Data:**
- Every feedback session
- Questions asked and answers provided
- Corrections made and validation results
- Resolution time per failure
- Success rate of corrections

**Storage:** `Project_Config/feedback_history.json`

**Benefits:**
- Track what selectors were added when/why
- Identify most common failure types
- Measure feedback effectiveness
- Training data for future AI improvements

## 2.2 Architecture Changes

### Agent Changes

| Agent | v1.0 Status | v2.0 Status | Changes |
|-------|-------------|-------------|---------|
| Jira Parser | ✅ Exists | ✅ Unchanged | No changes |
| Vision Executor | ✅ Exists | ⚠️ Enhanced | Keeps browser open, better diagnostics |
| **Feedback Agent** | ❌ N/A | ✅ **NEW** | Complete new agent |
| Report Generator | ✅ Exists | ⚠️ Enhanced | Shows feedback session data |

**Total Agents:** 3 → 4 (+1)

### Workflow Changes

**v1.0 Workflow:**
```
Jira Parser → Vision Executor → Report Generator → END
```

**v2.0 Workflow:**
```
Jira Parser → Vision Executor
    ↓
Has Failures? ─No─→ Report Generator → END
    ↓ Yes
Feedback Agent → Retry Failed Steps → Report Generator → END
```

**New:** Conditional feedback loop based on failure detection

### Configuration Files Added

| File | Purpose | Required? | Size |
|------|---------|-----------|------|
| `feedback_history.json` | Complete audit trail | Auto-created | Grows over time |
| `feedback_rules.json` | Learned patterns | Auto-created | Small (KB) |
| `selector_patterns.json` | Custom L2 patterns | Optional | Small (KB) |

**Total Config Files:** 1 → 4 (+3)

## 2.3 Backward Compatibility

### ✅ Fully Backward Compatible

**v1.0 Projects Work Without Modification:**
- If `feedback_enabled: false` → Behaves exactly like v1.0
- If config files missing → Auto-created with defaults
- Existing selectors.json → Used as-is, no migration needed

**Zero Breaking Changes:**
- All v1.0 agent interfaces unchanged
- State structure extended (not modified)
- Existing tests run without changes

### Migration Effort

| Scenario | Effort | Steps |
|----------|--------|-------|
| **Keep v1.0 behavior** | 0 minutes | Set `feedback_enabled: false` in config |
| **Enable v2.0 features** | 5 minutes | Set `feedback_enabled: true`, create empty JSONs |
| **Full optimization** | 30 minutes | Run 5-10 tests with feedback to build knowledge |

## 2.4 Performance Impact

### Execution Speed

| Scenario | v1.0 Time | v2.0 Time | Change |
|----------|-----------|-----------|--------|
| **All steps pass** | 30-45s | 30-45s | ✅ No change |
| **Failures occur** | Test stops | +2-5 min (feedback) | ⚠️ Interactive time |
| **After feedback** | N/A | 30-45s (retry) | ✅ Same as passing |

**Key Insight:** v2.0 adds NO overhead when tests pass. Only adds time when failures need correction.

### Resource Usage

| Resource | v1.0 | v2.0 | Impact |
|----------|------|------|--------|
| **Memory** | ~200 MB | ~220 MB | +10% (browser kept open) |
| **Disk** | Logs + Reports | +Feedback JSONs | +100 KB per test |
| **Network** | OpenAI L3 calls | Same | No change |
| **CPU** | Low | Low | No change |

---

# 3. System Architecture

## 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           USER                                   │
│           (Runs: python run_test.py TICKET-ID)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENTRY POINT                                  │
│                    run_test.py                                   │
│  • Parse CLI arguments                                           │
│  • Create initial state: {'ticket_number': 'TICKET-ID'}         │
│  • Invoke workflow orchestrator                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR LAYER                              │
│             workflows/test_workflow.py                           │
│              (LangGraph StateGraph)                              │
│  • Controls execution flow                                       │
│  • Manages shared state                                          │
│  • Calls agents sequentially                                     │
│  • Handles conditional branching                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┬──────────────┐
        ▼                     ▼                     ▼              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐  ┌──────────────┐
│  Agent 1     │    │  Agent 2     │    │  Agent 3     │  │  Agent 4     │
│ Jira Parser  │    │   Vision     │    │  Feedback    │  │   Report     │
│              │    │  Executor    │    │   (NEW!)     │  │  Generator   │
│ • Read file  │    │ • 3-Level    │    │ • Detect     │  │ • HTML       │
│ • Parse      │    │   strategy   │    │   failures   │  │ • Script     │
│   steps      │    │ • Execute    │    │ • Collect    │  │ • Summary    │
│ • Validate   │    │   tests      │    │   feedback   │  │              │
└──────────────┘    └──────────────┘    │ • Update     │  └──────────────┘
                              │          │   configs    │
                              │          │ • Retry      │
                              ▼          └──────────────┘
                    ┌──────────────┐             │
                    │  Utilities   │◄────────────┘
                    ├──────────────┤
                    │ • Element    │
                    │   Picker     │
                    │ • Config     │
                    │   Manager    │
                    │ • Feedback   │
                    │   Collector  │
                    │ • Vision     │
                    │   Client     │
                    └──────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │ SHARED STATE │
                    │ (Dictionary) │
                    └──────────────┘
```

## 3.2 Agent Communication Pattern

### Design Pattern: **StateGraph with Shared State**

**Key Principles:**
1. **No Direct Communication:** Agents NEVER call each other
2. **State-Based:** All communication through shared state object
3. **Orchestrator-Driven:** Workflow controls execution order
4. **Pure Functions:** Each agent is `state → updated_state`

### Communication Flow

```
User Input (ticket_id)
    ↓
Entry Point creates initial_state = {'ticket_number': ticket_id}
    ↓
Orchestrator.invoke(initial_state)
    ↓
Agent 1: jira_parser_agent(state)
    → Reads: state['ticket_number']
    → Writes: state['jira_data']
    → Returns: updated_state
    ↓
Orchestrator receives updated_state
    ↓
Agent 2: vision_executor_agent(state)
    → Reads: state['jira_data'], state['selectors']
    → Writes: state['execution_results'], state['browser_session']
    → Returns: updated_state
    ↓
Orchestrator checks: has_failures?
    ├─ Yes → Call Agent 3: feedback_agent(state)
    │         → Reads: state['execution_results'], state['browser_session']
    │         → Writes: state['feedback_session'], updates state['execution_results']
    │         → Returns: updated_state
    └─ No  → Skip to Agent 4
    ↓
Agent 4: report_generator_agent(state)
    → Reads: state['jira_data'], state['execution_results'], state['feedback_session']
    → Writes: state['report_path'], state['overall_status']
    → Returns: final_state
    ↓
Orchestrator returns final_state to Entry Point
    ↓
Entry Point displays results to user
```

## 3.3 Directory Structure

```
TA_AI_Project/
├── run_test.py                    # Entry point (receives user query)
│
├── workflows/                     # Orchestration layer
│   └── test_workflow.py          # LangGraph StateGraph definition
│
├── agents/                        # Primary agents (4)
│   ├── __init__.py
│   ├── jira_parser_agent.py      # Agent 1
│   ├── vision_executor_agent.py  # Agent 2
│   ├── feedback_agent.py         # Agent 3 (NEW)
│   └── report_generator_agent.py # Agent 4
│
├── utils/                         # Utility/helper classes
│   ├── config_manager.py         # Config file I/O (NEW)
│   ├── feedback_collector.py     # CLI interactions (NEW)
│   ├── element_picker.py         # JavaScript element picker (NEW)
│   ├── selector_loader.py        # Load selectors.json
│   ├── step_executor.py          # 3-level strategy execution
│   ├── vision_helper.py          # Azure OpenAI client
│   ├── module_mapper.py          # Jira↔Web module mapping
│   └── logger.py                 # Logging utilities
│
├── models/                        # Data models
│   └── state.py                  # TestAutomationState definition
│
├── templates/                     # Report templates
│   └── report_template.html     # Jinja2 HTML template
│
├── docs/                          # Documentation (NEW)
│   ├── specifications/
│   │   ├── spec_v1.0_original.md
│   │   ├── spec_v2.0_with_feedback.md  # This document
│   │   └── spec_changelog.md
│   ├── architecture/
│   ├── setup/
│   └── user_guides/
│
├── Project_Config/                # Project-specific data (NOT code)
│   ├── selectors.json            # L1 custom selectors
│   ├── selector_patterns.json    # L2 custom patterns (optional)
│   ├── feedback_rules.json       # Learned rules (NEW)
│   └── feedback_history.json     # Audit trail (NEW)
│
├── Jira_Tickets/                  # Test case files (input)
│   └── TICKET-ID.txt
│
├── Reports/                       # Generated HTML reports (output)
├── Screenshots/                   # Step screenshots (output)
├── Videos/                        # Test execution videos (output)
├── Generated_Scripts/             # Playwright scripts (output)
├── Logs/                          # Execution logs (output)
│
├── plcdtest_config.yaml          # Main configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 3.4 Data Flow Diagram

```
┌─────────────┐
│ User Input  │
│  TICKET-ID  │
└─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              SHARED STATE (Evolves Through Workflow)    │
├─────────────────────────────────────────────────────────┤
│ Initial:     {'ticket_number': 'RBPLCD-8835'}          │
│              ↓                                          │
│ After Agent1: + jira_data: {steps: [...]}              │
│              ↓                                          │
│ After Agent2: + execution_results: [...]               │
│              + browser_session: {...}                   │
│              ↓                                          │
│ After Agent3: + feedback_session: {...}  [if failures] │
│              + execution_results updated (retry)        │
│              ↓                                          │
│ After Agent4: + report_path: "Reports/..."             │
│              + overall_status: "PASSED/FAILED"          │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Output    │
│ • HTML      │
│ • Video     │
│ • Script    │
│ • Logs      │
└─────────────┘
```

---

# 4. Agent Specifications

## 4.1 Agent 1: Jira Parser Agent

### 4.1.1 Overview

| Property | Value |
|----------|-------|
| **File** | `agents/jira_parser_agent.py` |
| **Status in v2.0** | ✅ Unchanged from v1.0 |
| **Complexity** | Low (simple text parsing) |
| **AI/LLM** | No |
| **Dependencies** | None (standard Python libraries) |

### 4.1.2 Responsibilities

1. Read Jira ticket file from `Jira_Tickets/{ticket_number}.txt`
2. Extract ticket metadata (ID, module, title)
3. Parse test steps into structured format
4. Extract acceptance criteria
5. Validate required fields exist
6. Update state with parsed data

### 4.1.3 Input/Output

**Input:**
```python
state = {
    'ticket_number': 'RBPLCD-8835',
    'config': {...}
}
```

**Output:**
```python
state = {
    ...,
    'jira_data': {
        'ticket_id': 'RBPLCD-8835',
        'module': 'Teststep',
        'title': 'edit part details',
        'steps': [
            {'num': 1, 'text': 'Login'},
            {'num': 2, 'text': 'navigate to teststep'},
            ...
        ],
        'acceptance_criteria': '...'
    }
}
```

---

## 4.2 Agent 2: Vision Executor Agent

### 4.2.1 Overview

| Property | Value |
|----------|-------|
| **File** | `agents/vision_executor_agent.py` |
| **Status in v2.0** | ⚠️ Enhanced (keeps browser open for feedback) |
| **Complexity** | High (3-level strategy, browser automation) |
| **AI/LLM** | Yes (GPT-4o for L3) |
| **Dependencies** | Playwright, StepExecutor, VisionClient |

### 4.2.2 Key Changes from v1.0

| Change | v1.0 | v2.0 | Reason |
|--------|------|------|--------|
| Browser handling | Always closes | Keeps open if failures | Allow feedback agent to use live browser |
| Diagnostics | Basic error message | **Categorized diagnosis** | Enable intelligent feedback |
| Browser session in state | Not saved | **Saved to state** | Pass to feedback agent |

### 4.2.3 3-Level Selector Strategy

**Level 1 (L1): Custom Selectors**
- Source: `Project_Config/selectors.json`
- Speed: 1-2ms, Success Rate: 80%, Cost: $0

**Level 2 (L2): Generic HTML Patterns**
- Source: Hardcoded patterns in `step_executor.py`
- Speed: 10-20ms, Success Rate: 15%, Cost: $0

**Level 3 (L3): AI Vision Guidance**
- Source: GPT-4o analyzes screenshot
- Speed: 2-3 seconds, Success Rate: 5%, Cost: $0.01 per call

---

## 4.3 Agent 3: Feedback Agent (NEW)

### 4.3.1 Overview

| Property | Value |
|----------|-------|
| **File** | `agents/feedback_agent.py` |
| **Status in v2.0** | ✅ **NEW** - Core v2.0 feature |
| **Complexity** | High (interactive, multi-utility) |
| **AI/LLM** | Optional (AI suggestions feature) |
| **Dependencies** | ConfigManager, FeedbackCollector, ElementPicker, ValidationHelper |

### 4.3.2 Responsibilities

1. Detect failures from `execution_results`
2. Analyze and categorize each failure
3. Present failures to tester with visual context
4. Collect corrections via interactive CLI
5. **Activate Element Picker** for visual selector discovery
6. Validate corrections on live browser
7. Update configuration files
8. Learn patterns from corrections
9. Trigger retry of failed steps
10. Update state with retry results

### 4.3.3 Workflow

```
1. Detect failures
2. For each failure:
   a. Show failure details + screenshot
   b. Ask: How to fix? (Element Picker / Manual / Skip)
   c. If Element Picker → Activate JS picker
   d. Collect selected selector
   e. Validate on live page
   f. Save to selectors.json
   g. Learn patterns
3. Ask: Retry now?
4. If yes → Re-execute failed steps
5. Return updated state
```

---

## 4.4 Agent 4: Report Generator Agent

### 4.4.1 Overview

| Property | Value |
|----------|-------|
| **File** | `agents/report_generator_agent.py` |
| **Status in v2.0** | ⚠️ Enhanced (includes feedback data) |
| **Complexity** | Medium (template rendering) |
| **AI/LLM** | No |

### 4.4.2 Changes from v1.0

**v2.0 Enhancements:**
- Shows feedback session data (corrections made)
- Highlights which selectors were added via feedback
- Shows retry results
- Includes feedback metrics

---

# 5. Feedback Agent - Detailed Specification

## 5.1 Element Picker Feature

### 5.1.1 How It Works

```
1. System injects JavaScript into browser
2. Overlay appears: "Click on element you want to select"
3. Tester hovers → Elements highlight in blue
4. Tester clicks → Element data captured
5. JavaScript extracts possible selectors:
   - [data-editbtn]
   - .edit-button
   - #edit-btn-5
   - [aria-label="Edit"]
6. System presents list to tester
7. Tester selects preferred selector
8. System validates on live page
9. If valid → Save to selectors.json
```

### 5.1.2 JavaScript Code

The Element Picker injects a JavaScript overlay that:
- Creates visual feedback (blue highlight on hover)
- Captures element on click
- Extracts all possible selectors (data-*, classes, IDs, ARIA)
- Stores data in `window.__pickedElementData`
- Provides cancel option (ESC key)

---

## 5.2 Failure Categorization

### 5.2.1 Category Definitions

**A. Selector Issues**
- A1: Not Found (selector doesn't exist)
- A2: Ambiguous (multiple matches)
- A3: Wrong Element (matched incorrect element)

**B. Element State Issues**
- B1: Not Visible (hidden/display:none)
- B2: Not Clickable (covered by overlay)
- B3: Disabled (disabled attribute)
- B4: Out of Viewport (element below fold)

**C. Timing Issues**
- C1: Slow Loading (AJAX delay)
- C2: Animation Delay (CSS transition)
- C3: Race Condition (timing-dependent)

**D. Action Execution Issues**
- D1: Click Failed (JavaScript error)
- D2: Dropdown Didn't Open (click didn't trigger)
- D3: Value Not in List (option missing)
- D4: Input Rejected (validation failed)

**E. Context Issues**
- E1: Wrong Page (navigation failed)
- E2: Modal/Dialog (blocking access)
- E3: Iframe (different context)
- E4: Shadow DOM (encapsulated)

**F. Verification Issues**
- F1: Message Not Shown (expected text missing)
- F2: Wrong Message (unexpected error)
- F3: No State Change (action had no effect)

---

## 5.3 Pattern Learning

### 5.3.1 What Gets Learned

**Row Scoping Patterns**
```
When: Step mentions "named as X"
Learns: `:text-is('{identifier}') >> {selector}`
Applied: Future steps with row identifiers
```

**Timing Adjustments**
```
When: Timing failures occur
Learns: Increased wait times for specific actions
Applied: Same action types in future
```

**Selector Aliases**
```
When: Multiple selectors work for same element
Learns: Alternative selector mappings
Applied: Fallback options
```

---

# 6. Failure Handling System

## 6.1 Complete Failure Taxonomy

### Category A: Selector Issues

| Type | Symptom | Root Cause | Handler |
|------|---------|------------|---------|
| **A1: Not Found** | Count = 0 | Selector doesn't exist | Element Picker |
| **A2: Ambiguous** | Count > 1 | Multiple matches | Scope Suggester |
| **A3: Wrong Element** | Found but wrong | Matched incorrect | Element Picker |

### Category B: Element State Issues

| Type | Symptom | Root Cause | Handler |
|------|---------|------------|---------|
| **B1: Not Visible** | Hidden | Not rendered | Check prerequisites |
| **B2: Not Clickable** | Covered | Overlay blocking | Wait for overlay |
| **B3: Disabled** | Disabled attr | Can't interact | Check preconditions |
| **B4: Out of Viewport** | Below fold | Not in view | Auto-scroll |

### Category C: Timing Issues

| Type | Symptom | Root Cause | Handler |
|------|---------|------------|---------|
| **C1: Slow Loading** | Not present | AJAX delay | Increase wait time |
| **C2: Animation** | Transitioning | CSS animation | Add animation wait |
| **C3: Race Condition** | Intermittent | Timing-dependent | Explicit wait |

---

# 7. Configuration Files

## 7.1 plcdtest_config.yaml

**Enhanced for v2.0 with feedback settings:**

```yaml
project_name: "ProjectName"
web_url: "http://your-app.com/login"
browser: "edge"

login:
  username: "user"
  password: "pass"

wait_times:
  after_login: 3000
  after_click: 1000

azure_openai:
  api_key: "your_key"
  endpoint: "https://your-endpoint.openai.azure.com/"
  deployment_gpt4o: "gpt-4o"

execution:
  feedback_enabled: true          # NEW v2.0
  max_retries: 3
  screenshot_on_every_step: true
  record_video: true

feedback:                          # NEW v2.0
  element_picker_enabled: true
  ai_suggestions_enabled: true
  save_feedback_history: true
  auto_retry_after_feedback: true
```

---

## 7.2 selectors.json

```json
{
  "version": "1.0",
  "project": "ProjectName",
  "selectors": [
    {
      "id": "sel_001",
      "module": "Teststep",
      "attr": "data-editbtn",
      "value": "",
      "label": "Edit button",
      "source": "feedback",
      "feedback_metadata": {
        "from_ticket": "RBPLCD-8835",
        "from_step": 5,
        "added_date": "2025-10-30T15:30:00"
      }
    }
  ]
}
```

---

## 7.3 feedback_history.json (NEW)

Complete audit trail of all feedback sessions:

```json
{
  "version": "1.0",
  "feedback_sessions": [
    {
      "session_id": "fb_20251030_153000",
      "ticket_id": "RBPLCD-8835",
      "timestamp": "2025-10-30T15:30:00",
      "tester": "qa_user",
      "failures_count": 2,
      "feedback_items": [
        {
          "step_num": 5,
          "failure_type": "selector_not_found",
          "answers_provided": {
            "method": "picker",
            "selector": "[data-editbtn]"
          },
          "resolution": {
            "action_taken": "add_to_selectors_json",
            "retry_passed": true,
            "time_to_resolve_seconds": 120
          }
        }
      ]
    }
  ]
}
```

---

## 7.4 feedback_rules.json (NEW)

Learned patterns auto-applied in future tests:

```json
{
  "version": "1.0",
  "row_scoping_rules": [
    {
      "rule_id": "rule_001",
      "pattern": ":text-is('{identifier}') >> {selector}",
      "applies_to_keywords": ["part", "named as"],
      "usage_count": 5
    }
  ],
  "timing_adjustments": [
    {
      "action_type": "dropdown_select",
      "wait_after_click_ms": 800
    }
  ]
}
```

---

# 8. Input Requirements

## 8.1 Mandatory Files (3 minimum)

### 1. plcdtest_config.yaml
- **Purpose:** Main configuration
- **Time:** 10 minutes
- **Required fields:** web_url, login, azure_openai

### 2. Jira_Tickets/TICKET-ID.txt
- **Purpose:** Test case
- **Time:** 5 minutes
- **Format:** Jira ticket format with steps

### 3. Project_Config/selectors.json
- **Purpose:** L1 selectors
- **Time:** 1 minute (empty) or 30 minutes (populated)
- **Can start empty:** Yes

## 8.2 Optional Files (Auto-Created)

- `feedback_history.json` - Created on first feedback
- `feedback_rules.json` - Created when patterns learned
- `selector_patterns.json` - Manual creation only

---

# 9. User Workflows

## 9.1 Workflow 1: Test Passes (No Feedback)

```
python run_test.py TEST-001
→ All steps pass
→ Report generated
→ Time: 30-45 seconds
```

## 9.2 Workflow 2: Test Fails, Feedback Corrects

```
python run_test.py TEST-002
→ Step 3 fails
→ Feedback Agent activates
→ Element Picker shown
→ Tester clicks element
→ Selector [data-editbtn] captured
→ Validated ✅
→ Saved to selectors.json
→ Retry Step 3 → Passes ✅
→ Report generated
→ Time: 40s + 2min feedback + 5s retry = ~3 minutes
```

## 9.3 Workflow 3: Multiple Failures

```
→ 3 steps fail
→ Feedback Agent detects blocker (Step 5)
→ Fix Step 5 with Element Picker
→ Steps 6-7 auto-fix (cascading)
→ All pass after retry
→ Time: ~5-6 minutes total
```

---

# 10. State Management

## 10.1 State Definition

```python
class TestAutomationState(TypedDict, total=False):
    # Input
    ticket_number: str

    # From Config Loader
    config: Dict
    selectors: List[Dict]

    # From Jira Parser
    jira_data: Dict

    # From Vision Executor
    execution_results: List[Dict]
    browser_session: Dict  # NEW v2.0

    # From Feedback Agent (NEW v2.0)
    feedback_session: Dict

    # From Report Generator
    report_path: str
    overall_status: str
```

---

# 11. Implementation Guide

## 11.1 Phase-Based Development

### Phase 1: Foundation (Week 1)
- Create documentation structure
- Define enhanced state model
- Implement ConfigManager utility
- Implement FeedbackCollector utility

### Phase 2: Diagnostic Enhancement (Week 2)
- Create DiagnosticHelper utility
- Enhance Vision Executor with diagnostics

### Phase 3: Feedback Agent MVP (Week 3)
- Basic feedback agent (manual entry only)
- Update selectors.json
- No Element Picker yet

### Phase 4: Element Picker (Week 4)
- JavaScript element picker
- Integration with Feedback Agent

### Phase 5: Retry Logic (Week 5)
- Complete feedback loop
- Automatic retry

### Phase 6: Pattern Learning (Week 6)
- Pattern detection
- Auto-apply learned rules

### Phase 7: Testing & Polish (Week 7)
- Integration testing
- Documentation
- User guides

---

# 12. Migration Guide (v1.0 → v2.0)

## 12.1 Backward Compatibility

✅ **v2.0 is 100% backward compatible with v1.0**

## 12.2 Migration Options

### Option A: Keep v1.0 Behavior (0 minutes)
```yaml
execution:
  feedback_enabled: false
```

### Option B: Enable v2.0 Features (5 minutes)
```yaml
execution:
  feedback_enabled: true

# Create empty files:
# - feedback_history.json
# - feedback_rules.json
```

### Option C: Full Optimization (30-60 minutes)
- Enable feedback
- Run 5-10 tests with feedback
- Build selector library
- System learns patterns

---

# 13. Success Metrics

## 13.1 Performance Metrics

| Metric | v1.0 | v2.0 Target |
|--------|------|-------------|
| Test Execution (passing) | 30-45s | 30-45s |
| Failure Resolution | 15-30 min | <5 min |
| Selector Discovery | 10-15 min | <2 min |

## 13.2 Quality Metrics

| Metric | v1.0 | v2.0 Target |
|--------|------|-------------|
| Test Accuracy | 99%+ | 99%+ |
| Retry Success Rate | N/A | >85% |

---

# 14. Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single browser session | No parallel tests | Run sequentially |
| CLI-only feedback | Requires terminal | Use SSH |
| Element Picker requires visible element | Can't pick hidden | Manual entry |

---

# 15. API Reference

## 15.1 Agent Functions

### jira_parser_agent()
```python
def jira_parser_agent(state: TestAutomationState) -> TestAutomationState:
    """Parse Jira ticket and extract test data."""
```

### vision_executor_agent()
```python
def vision_executor_agent(state: TestAutomationState) -> TestAutomationState:
    """Execute test steps using 3-level selector strategy."""
```

### feedback_agent()
```python
def feedback_agent(state: TestAutomationState) -> TestAutomationState:
    """Collect feedback for failures and update configurations. NEW in v2.0."""
```

### report_generator_agent()
```python
def report_generator_agent(state: TestAutomationState) -> TestAutomationState:
    """Generate HTML report and Playwright script."""
```

---

# 16. Appendices

## Appendix A: Failure Type Quick Reference

| Code | Type | Handler | Auto-Fix? |
|------|------|---------|-----------|
| A1 | Selector Not Found | Element Picker | ❌ |
| A2 | Ambiguous | Scope Suggester | ⚠️ |
| B1 | Not Visible | Prerequisites | ⚠️ |
| C1 | Slow Loading | Auto-Wait | ✅ |
| D3 | Value Not in List | Value Suggester | ❌ |

## Appendix B: Glossary

- **L1/L2/L3:** 3-level selector strategy
- **Element Picker:** Interactive selector discovery tool
- **Feedback Agent:** Agent that collects corrections
- **Learned Rule:** Pattern auto-applied from feedback
- **StateGraph:** LangGraph orchestration pattern

## Appendix C: Troubleshooting

**Issue:** Feedback Agent not activating
- **Solution:** Check `feedback_enabled: true` in config

**Issue:** Element Picker not working
- **Solution:** Check browser console, try manual entry

**Issue:** Selectors not saving
- **Solution:** Check file permissions

---

# END OF SPECIFICATION v2.0

**Document Status:** Complete
**Last Updated:** 2025-10-30
**Document Version:** 2.0.0
**Total Pages:** 48
