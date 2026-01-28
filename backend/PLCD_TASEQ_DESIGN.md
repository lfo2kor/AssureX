# PLCD Testing Assistant - Sequential Context Tracking Design
## plcd_taseq.py & feedback_taseq.py

**Version:** 2.0 - Sequential Context Tracking with LangGraph Multi-Agent Architecture
**Date:** 2025-11-18
**Status:** Design Document

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [plcd_taseq.py - Main Test Executor](#plcd_taseqpy---main-test-executor)
4. [feedback_taseq.py - Feedback Collection Tool](#feedback_taseqpy---feedback-collection-tool)
5. [Agent Specifications](#agent-specifications)
6. [Memory Management](#memory-management)
7. [Configuration](#configuration)
8. [Artifacts Generation](#artifacts-generation)
9. [Integration Flow](#integration-flow)

---

## Overview

### Problem Statement
Current `plcd_ta.py` treats each test step independently without understanding:
- Sequential execution context (what happened before)
- Current page state (URL, visible elements)
- Test flow dependencies (navigation required before actions)
- Learning from failures across test runs

### Solution: Sequential Context Tracking
Implement **LangGraph-based multi-agent system** with:
- Real-time context capture and propagation
- LLM reasoning at every decision point
- No hardcoding (all logic driven by YAML prompts)
- Continuous learning from failures and human feedback

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         plcd_taseq.py                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              OrchestratorAgent (Supervisor)                   │  │
│  │        Decides: Which agent to call next?                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│        ┌─────────────────────┼─────────────────────┐                │
│        ▼                     ▼                     ▼                │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐            │
│  │  Jira    │         │ Context  │         │ Learning │            │
│  │  Agent   │         │  Agent   │         │  Agent   │            │
│  └──────────┘         └──────────┘         └──────────┘            │
│                              │                                       │
│        ┌─────────────────────┼─────────────────────┐                │
│        ▼                     ▼                     ▼                │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐            │
│  │Selector  │         │Selector  │         │Selector  │            │
│  │Agent L1  │──low──▶ │Agent L2  │──fail─▶ │Agent L3  │            │
│  │(RAG+LLM) │  conf   │(DOM+LLM) │         │(Vision)  │            │
│  └──────────┘         └──────────┘         └──────────┘            │
│                              │                                       │
│                              ▼                                       │
│                     ┌─────────────────┐                             │
│                     │ Execute Action  │                             │
│                     └─────────────────┘                             │
│                              │                                       │
│                              ▼                                       │
│              Update Context → Store Learning                        │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ (generates)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Artifacts Generated                           │
│  • HTML Report (with context trace, LLM reasoning)                  │
│  • Video Recording (.webm)                                          │
│  • Python Test Script (.py)                                         │
│  • Context Trace JSON (debug)                                       │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ (if failure, human reviews)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      feedback_taseq.py                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ReportParser  │───▶│  Feedback    │───▶│  Learning    │          │
│  │   Agent      │    │ Collector    │    │  Storage     │          │
│  │  (LLM)       │    │  Agent       │    │  Agent       │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│   Extract steps      Get corrections      Store to ChromaDB        │
│   + context          with context tags    (high priority)          │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    Next run: LearningAgent finds
                    correction → Applies automatically
```

---

## plcd_taseq.py - Main Test Executor

### Core Functionality

#### 1. Sequential Context Tracking
**What:** Captures and maintains execution context throughout test run

**Captured Information:**
- `current_url`: Page URL after each action
- `visible_data_attributes[]`: All data-* attributes visible on page
- `step_history[]`: List of executed steps with outcomes
- `navigation_path[]`: Breadcrumb/URL changes
- `last_action`: Previous step result
- `dom_snapshot`: Key DOM elements at current state

**How:**
```python
# Before each step
context = page.evaluate("""
    () => ({
        url: window.location.href,
        visible_elements: Array.from(document.querySelectorAll('[data-*]'))
            .map(el => Array.from(el.attributes)
                .filter(a => a.name.startsWith('data-'))
                .map(a => a.name + '=' + a.value))
            .flat(),
        breadcrumb: Array.from(document.querySelectorAll('[data-breadcrumb] span'))
            .map(el => el.textContent)
    })
""")
```

#### 2. LLM-Based Jira Parsing (No Regex)
**What:** Understands any ticket format using LLM

**How:**
- Reads raw Jira text file
- Sends to `JiraAgent` with project-specific prompt from YAML
- LLM extracts: module, steps, expected results
- Returns structured JSON

**Example Prompt (in YAML):**
```yaml
jira_agent:
  system_prompt: |
    Parse Jira ticket and extract test steps.
    Handle formats:
    - "1. Login\nExpected: Dashboard"
    - "Step 1: Click Add | Expected Result: Form opens"

    Return JSON: {"module": "...", "steps": [...]}
```

#### 3. Multi-Agent Orchestration
**What:** Dynamic routing based on confidence and context

**Flow:**
```
Start → JiraAgent → ContextAgent → LearningAgent (check memory)
                                         │
                     ┌───────────────────┴──────────────────┐
                     │                                       │
                Found correction?                          No match
                     │                                       │
                     ▼                                       ▼
              Use correction                        SelectorAgent_L1
              (confidence: 0.95)                            │
                                                   ┌────────┴────────┐
                                                   │                 │
                                              conf >= 0.70      conf < 0.70
                                                   │                 │
                                                Execute         SelectorAgent_L2
                                                                     │
                                                            ┌────────┴────────┐
                                                            │                 │
                                                       Found match         Failed
                                                            │                 │
                                                         Execute      SelectorAgent_L3
                                                                          (Vision)
```

#### 4. Adaptive Selector Discovery
**What:** Context-aware selector matching with 3-level fallback

**Level 1 (SelectorAgent_L1):**
1. LLM enhances query with context: `"Navigate to Runs"` → `"Navigation link to Runs module from Dashboard sidebar"`
2. RAG semantic search on enhanced query
3. LLM validates candidates against visible DOM elements
4. Returns: best selector + confidence + reasoning

**Level 2 (SelectorAgent_L2):**
1. Scrapes live DOM elements (buttons, inputs, links with data-* attributes)
2. LLM matches scraped elements to step intent
3. Prioritizes: data-* > aria-* > id > class
4. Returns: discovered selector + confidence

**Level 3 (SelectorAgent_L3):**
1. Takes screenshot of current page
2. LLM-Vision analyzes screenshot + step text
3. Identifies target element visually
4. Generates stable selector (data-* preferred)
5. Validates selector on page

#### 5. Continuous Learning
**What:** Learns from failures and corrections automatically

**Learning Sources:**
1. **Failure Patterns:** Stores {context, wrong_selector, reason}
2. **Human Corrections:** From feedback_taseq.py (highest priority)
3. **Successful Executions:** Stores verified selectors with context

**Storage:**
- ChromaDB collection: `learning_collection`
- Embeddings: Context + step_text for semantic matching
- Metadata: {url_pattern, visible_elements_pattern, confidence}

#### 6. Memory Management
**What:** Configurable memory retention per agent

**Types:**
- **Short-term:** ContextAgent (last N steps, configurable)
- **Long-term:** LearningAgent (persistent in ChromaDB)
- **Session:** SelectorAgents (cleared after test run)
- **Full Session:** OrchestratorAgent (entire test execution)

---

## feedback_taseq.py - Feedback Collection Tool

### Core Functionality

#### 1. Context-Aware Report Parsing
**What:** Extracts execution details with full context trace

**Reads from HTML Report:**
- Step number, text, status
- Selector used (or attempted)
- Agent chain (L1/L2/L3)
- **NEW:** Execution context (URL, visible elements)
- **NEW:** LLM reasoning for selector choice
- Confidence score

**Agent:** `ReportParserAgent` (LLM-based, handles any HTML format)

#### 2. Interactive Correction Interface
**What:** CLI interface for tester to correct failures

**Workflow:**
```
1. Show failed step with context:
   ┌─────────────────────────────────────────────────────────────┐
   │ Step 3: Navigate to Runs [FAILED]                           │
   │                                                              │
   │ Context at execution:                                       │
   │   URL: /client/dashboard                                    │
   │   Visible elements: data-dashboard, data-sidebar, ...       │
   │                                                              │
   │ Selector tried: [data-routerLinkNameTestStep='{{row...']   │
   │ Confidence: 0.36                                            │
   │ Agent used: L1                                              │
   │                                                              │
   │ LLM reasoning: "Matched table row selector, but low        │
   │                 confidence due to context mismatch"         │
   │                                                              │
   │ Why it failed: Element not found (timeout)                 │
   └─────────────────────────────────────────────────────────────┘

2. Ask tester: "Enter correct selector:"
   Tester input: data-navitem='runs'

3. Ask: "Add context tags (comma-separated):"
   Tester input: navigation, sidebar, from_dashboard

4. Validate selector on page (if possible)

5. Store to ChromaDB with high priority
```

#### 3. Context Tagging & Enrichment
**What:** Collects contextual metadata for corrections

**Agent:** `ContextEnrichmentAgent`

**Collected Metadata:**
- `url_pattern`: e.g., "*/dashboard", "*/projects*"
- `required_visible_elements`: e.g., ["data-sidebar", "data-navitem"]
- `navigation_state`: e.g., "from_dashboard", "after_login"
- `module_transition`: e.g., "Dashboard → Teststep"

**LLM Analysis:**
```
Prompt: "Analyze this correction context and suggest similar patterns"

Context:
  - Failed selector: [data-routerLinkNameTestStep='{{row.aeName}}']
  - Correct selector: [data-navitem='runs']
  - URL: /dashboard
  - Visible: data-sidebar, data-navitem

Response:
  - Pattern: "Table selectors don't work for navigation from Dashboard"
  - Similar contexts: "Any navigation from Dashboard to other modules"
  - Suggested rule: "For navigation from Dashboard, prioritize data-navitem"
```

#### 4. Learning Storage
**What:** Saves corrections to ChromaDB with context

**Agent:** `LearningStorageAgent`

**Storage Format:**
```json
{
  "id": "correction_001",
  "step_text": "Navigate to Runs",
  "correct_selector": "[data-navitem='runs']",
  "wrong_selectors": ["[data-routerLinkNameTestStep='{{row.aeName}}']"],
  "context": {
    "url_pattern": "*/dashboard",
    "required_visible": ["data-sidebar", "data-navitem"],
    "tags": ["navigation", "sidebar", "from_dashboard"]
  },
  "confidence": 0.98,
  "source": "human_feedback",
  "verified": true,
  "ticket_id": "RBPLCD-8835",
  "timestamp": "2025-11-18T15:30:00"
}
```

**Retrieval Priority:**
1. Human corrections (confidence: 0.95+)
2. Verified successful selectors (confidence: 0.85+)
3. Runtime learned selectors (confidence: varies)

---

## Agent Specifications

### 1. OrchestratorAgent (Supervisor)

**Role:** Decides which agent to invoke next based on state

**Model:** gpt-4o

**Inputs:**
- Current state (step, context, previous attempts)
- Agent capabilities
- Confidence thresholds from config

**Outputs:**
- `next_agent`: Which agent to call
- `reasoning`: Why this agent
- `skip_agents`: Agents to bypass (e.g., skip L2 if L1 confidence > 0.9)

**Decision Logic:**
```python
if similar_correction_found and confidence > 0.90:
    return "execute"  # Skip selector discovery
elif previous_attempt == "L1" and confidence < 0.70:
    return "l2"
elif previous_attempt == "L2" and failed:
    return "l3"
else:
    return "l1"  # Start with L1
```

---

### 2. JiraAgent

**Role:** Parse Jira ticket text (any format) using LLM

**Model:** gpt-4o (configurable)

**Inputs:**
- Raw Jira ticket text file
- Format examples from YAML

**Outputs:**
```json
{
  "ticket_id": "RBPLCD-8835",
  "title": "Test automation for...",
  "module": "Teststep",
  "steps": [
    {"number": 1, "text": "Login", "expected": "Dashboard visible"},
    {"number": 2, "text": "Navigate to Runs", "expected": "Runs list shown"}
  ]
}
```

**Prompt (from YAML):**
```yaml
jira_agent:
  system_prompt: |
    Extract test steps from Jira ticket. Handle any format.
    Return JSON with module and steps array.

  format_examples: |
    Format 1: "1. Step text\nExpected: Result"
    Format 2: "Step 1: Text | Expected Result: Result"
```

---

### 3. ContextAgent

**Role:** Capture and maintain execution context

**Model:** gpt-4o-mini (lightweight)

**Responsibilities:**
1. **Capture Context:** Extract current state from page
2. **Update Context:** After each action
3. **Build Summary:** Create context summary for other agents

**Context Data Structure:**
```python
{
  "current_url": "/client/dashboard",
  "visible_data_attributes": ["data-sidebar", "data-navitem='runs'", ...],
  "breadcrumb": ["Home", "Dashboard"],
  "last_action": "Login successful",
  "step_history": [
    {"step": 1, "action": "Login", "result": "success", "url": "/dashboard"}
  ],
  "timestamp": "2025-11-18T15:16:53"
}
```

**Memory:** Sliding window (last N steps, configurable in YAML)

---

### 4. LearningAgent

**Role:** Check past failures/corrections before selector discovery

**Model:** gpt-4o-mini

**Responsibilities:**
1. **Query Memory:** Search ChromaDB for similar contexts
2. **Pattern Matching:** Match current context to past failures
3. **Suggest Corrections:** Return high-confidence corrections

**Query Process:**
```python
# Generate embedding for current situation
query_text = f"{step_text} | context: {current_url} | visible: {visible_elements}"
embedding = get_embedding(query_text)

# Search ChromaDB
results = learning_collection.query(
    query_embeddings=[embedding],
    n_results=3,
    where={
        "source": "human_feedback",  # Prioritize human corrections
        "confidence": {"$gte": 0.85}
    }
)

# LLM validates match
if results and context_matches(results[0], current_context):
    return results[0]["correct_selector"], confidence=0.95
```

**Storage:** ChromaDB `learning_collection` (permanent)

---

### 5. SelectorAgent_L1 (RAG + LLM)

**Role:** Semantic search with context-aware validation

**Model:** gpt-4o

**Process:**

**Step 1: Pre-RAG (Enhance Query)**
```
Prompt:
  Context: {current_url, visible_elements, last_action}
  Step: "Navigate to Runs"

  Generate enhanced query for RAG search considering context.

Response:
  "Navigation sidebar link from Dashboard to Runs/Teststep module"
```

**Step 2: RAG Search**
```python
enhanced_embedding = get_embedding(llm_response)
candidates = vector_db.query(enhanced_embedding, n_results=5)
```

**Step 3: Post-RAG (Validate)**
```
Prompt:
  RAG returned these candidates:
  1. [data-routerLinkNameTestStep='{{row.aeName}}'] (conf: 0.71)
  2. [data-baselinkTestStep='baselinkTestStep'] (conf: 0.65)

  Visible DOM elements: data-sidebar, data-navitem='runs', ...

  Which selector is feasible? Consider element must be visible.

Response:
  {
    "selected": null,
    "reasoning": "Candidate 1 is a table row selector (tr element),
                  not visible on Dashboard. Candidate 2 also requires
                  table context. None are suitable.",
    "confidence": 0.0,
    "recommendation": "Try L2 DOM discovery"
  }
```

**Output:** Selector + confidence + reasoning

---

### 6. SelectorAgent_L2 (DOM + LLM)

**Role:** Discover selectors from live DOM

**Model:** gpt-4o

**Process:**

**Step 1: Scrape DOM**
```javascript
page.evaluate(() => {
  const elements = document.querySelectorAll('button, input, a, [data-*]');
  return Array.from(elements).map(el => ({
    tag: el.tagName,
    text: el.textContent.trim(),
    attributes: Array.from(el.attributes).reduce((acc, attr) => {
      acc[attr.name] = attr.value;
      return acc;
    }, {}),
    visible: el.offsetParent !== null
  })).filter(el => el.visible);
})
```

**Step 2: LLM Analysis**
```
Prompt:
  Step: "Navigate to Runs"
  Context: Dashboard page

  Found elements on page:
  1. <a data-navitem="projects">Projects</a>
  2. <a data-navitem="runs">Runs</a>
  3. <a data-navitem="parts">Parts</a>

  Which element matches the step intent?

Response:
  {
    "selected_element": 2,
    "selector": "[data-navitem='runs']",
    "confidence": 0.92,
    "reasoning": "Element 2 has text 'Runs' matching step intent,
                  and data-navitem is a stable data attribute"
  }
```

**Step 3: Validate Selector**
```python
try:
    element = page.locator(selector)
    if element.count() == 1:
        return selector, confidence=0.92
except:
    return None
```

---

### 7. SelectorAgent_L3 (Vision)

**Role:** Visual element identification using LLM-Vision

**Model:** gpt-4o (vision-enabled)

**Process:**

**Step 1: Screenshot**
```python
screenshot = page.screenshot(full_page=True)
```

**Step 2: LLM-Vision Analysis**
```
Prompt:
  [Screenshot attached]

  Task: "Navigate to Runs"
  Context: Dashboard page, need to find navigation link

  Identify the target element and generate a stable selector.
  Prioritize: data-* > aria-* > id > class

Response:
  {
    "element_identified": true,
    "location": "Left sidebar, 3rd item from top",
    "text": "Runs",
    "suggested_selectors": [
      "[data-navitem='runs']",
      "a:has-text('Runs')",
      "nav a:nth-child(3)"
    ],
    "confidence": 0.85,
    "reasoning": "Identified navigation link with text 'Runs' in sidebar"
  }
```

**Step 3: Try Selectors**
```python
for selector in suggested_selectors:
    if validate_selector(page, selector):
        return selector, confidence
```

---

## Memory Management

### Configuration (YAML)

```yaml
memory:
  # ContextAgent - Recent execution history
  context_agent:
    type: "short_term"
    retention_steps: 10              # Keep last 10 steps
    retention_strategy: "sliding_window"
    include_fields:
      - current_url
      - visible_data_attributes
      - step_outcome
      - timestamp

  # LearningAgent - Persistent learning
  learning_agent:
    type: "long_term"
    storage: "chromadb:learning_collection"
    max_patterns: 1000               # Max stored failure patterns
    similarity_threshold: 0.85        # For context matching
    cleanup_strategy: "keep_high_confidence"  # Remove low-conf after 30 days

  # SelectorAgents - Per-step attempts
  selector_agents:
    type: "session"
    retention_steps: 3                # Keep last 3 attempts per step
    clear_on_success: true            # Clear failed attempts after success
    store_reasoning: true             # Keep LLM reasoning

  # OrchestratorAgent - Full test run
  orchestrator:
    type: "full_session"
    max_steps: 100                    # Safety limit per test
    persist_to_file: "Logs/orchestrator_decisions.json"
```

### Memory Types

#### 1. Short-Term Memory (ContextAgent)
**Purpose:** Track recent execution for context continuity

**Implementation:**
```python
class ContextMemory:
    def __init__(self, config):
        self.retention = config['memory']['context_agent']['retention_steps']
        self.history = []

    def add(self, context):
        self.history.append(context)
        # Keep only last N
        self.history = self.history[-self.retention:]

    def get_summary(self):
        """Generate summary for other agents"""
        return {
            "recent_steps": len(self.history),
            "url_changes": [h["url"] for h in self.history],
            "last_action": self.history[-1] if self.history else None
        }
```

#### 2. Long-Term Memory (LearningAgent)
**Purpose:** Persistent learning across test runs

**Storage:** ChromaDB with metadata

**Schema:**
```python
{
  "id": "learning_001",
  "embedding": [0.1, 0.2, ...],  # Semantic search
  "metadata": {
    "step_text": "Navigate to Runs",
    "selector": "[data-navitem='runs']",
    "context_pattern": {
      "url_contains": "dashboard",
      "visible_must_have": ["data-sidebar"]
    },
    "confidence": 0.95,
    "source": "human_feedback",  # or "runtime_success"
    "ticket_id": "RBPLCD-8835",
    "created": "2025-11-18",
    "usage_count": 5
  }
}
```

**Retrieval:**
```python
def get_learned_selector(step_text, context):
    # Generate contextual query
    query = f"{step_text} | url: {context['url']} | visible: {context['visible']}"

    # Search with context filters
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        where={
            "$and": [
                {"confidence": {"$gte": 0.85}},
                {"metadata.context_pattern.url_contains": context["url_pattern"]}
            ]
        },
        n_results=3
    )

    # LLM validates best match
    return llm_validate_match(results, context)
```

#### 3. Session Memory (SelectorAgents)
**Purpose:** Track selector attempts within current test run

**Cleared:** After test completion or on success

```python
class SessionMemory:
    def __init__(self):
        self.attempts = {}  # {step_number: [attempts]}

    def record_attempt(self, step, agent, selector, confidence, result):
        if step not in self.attempts:
            self.attempts[step] = []

        self.attempts[step].append({
            "agent": agent,
            "selector": selector,
            "confidence": confidence,
            "result": result,
            "timestamp": datetime.now()
        })

    def get_failed_selectors(self, step):
        """Get selectors that already failed for this step"""
        if step in self.attempts:
            return [a["selector"] for a in self.attempts[step] if a["result"] == "failed"]
        return []
```

---

## Configuration

### Enhanced plcdtestassistant.yaml

```yaml
# ============================================================================
# SEQUENTIAL CONTEXT TRACKING CONFIGURATION
# ============================================================================

# Feature Flags
features:
  enable_sequential_context: true
  enable_llm_jira_parsing: true
  enable_multi_agent_orchestration: true
  enable_continuous_learning: true

# ============================================================================
# LANGGRAPH AGENTS CONFIGURATION
# ============================================================================

agents:
  # -------------------------------------------------------------------------
  # OrchestratorAgent - Supervisor
  # -------------------------------------------------------------------------
  orchestrator_agent:
    enabled: true
    model: "gpt-4o"
    temperature: 0.1
    max_tokens: 1000

    system_prompt: |
      You are the Test Execution Orchestrator. Route to appropriate agents based on state.

      AGENTS AVAILABLE:
      - JiraAgent: Parse ticket (call once at start)
      - ContextAgent: Capture/update context (before/after each step)
      - LearningAgent: Check past failures/corrections (before selector discovery)
      - SelectorAgent_L1: RAG search with LLM validation (try first)
      - SelectorAgent_L2: DOM discovery with LLM analysis (if L1 conf < 0.70)
      - SelectorAgent_L3: Vision-based identification (if L1/L2 fail)

      ROUTING RULES:
      1. Always check LearningAgent first for high-confidence corrections
      2. If found correction with conf > 0.90: execute directly
      3. Otherwise: L1 → (if conf < 0.70) → L2 → (if fail) → L3
      4. After each action: Update ContextAgent
      5. After failures: Store to LearningAgent

      Return JSON: {"next_agent": "...", "reasoning": "...", "skip": [...]}

    routing_prompt: |
      CURRENT STATE:
      - Step: {step_text}
      - Context: {current_context}
      - Last attempt: {last_attempt}
      - Previous confidence: {confidence}

      MEMORY CHECK:
      - Similar failures found: {failure_patterns}
      - Human corrections available: {corrections}

      QUESTION: Which agent should handle this next? Why?

  # -------------------------------------------------------------------------
  # JiraAgent - LLM-Based Ticket Parsing
  # -------------------------------------------------------------------------
  jira_agent:
    enabled: true
    model: "gpt-4o"
    temperature: 0.1
    max_tokens: 2000

    system_prompt: |
      You are a Jira ticket parser. Extract test steps from any ticket format.

      TASK:
      - Parse ticket text and identify: module, test steps, expected results
      - Handle various formats (numbered lists, tables, bullet points)
      - Extract step dependencies if mentioned

      OUTPUT FORMAT:
      {
        "ticket_id": "RBPLCD-XXXX",
        "title": "...",
        "module": "Teststep",
        "steps": [
          {"number": 1, "text": "Login", "expected": "Dashboard visible"},
          {"number": 2, "text": "Navigate to Runs", "expected": "Runs list shown"}
        ]
      }

    # Project-specific format examples (customize per project)
    format_examples: |
      Your project uses these formats:

      Format 1: Numbered with "Expected:"
        1. Login to application
        Expected: Dashboard page loads

        2. Click on Runs menu
        Expected: Runs list displayed

      Format 2: Table format
        | Step | Action | Expected Result |
        | 1 | Login | Dashboard |
        | 2 | Navigate to Runs | List shown |

      Format 3: Bullet points
        • Login → Dashboard visible
        • Go to Runs → Table with test runs

  # -------------------------------------------------------------------------
  # ContextAgent - Execution Context Tracking
  # -------------------------------------------------------------------------
  context_agent:
    enabled: true
    model: "gpt-4o-mini"
    temperature: 0
    max_tokens: 500

    system_prompt: |
      You are the Context Tracker. Maintain execution state throughout test run.

      RESPONSIBILITIES:
      1. Capture context before each step (URL, visible elements, breadcrumb)
      2. Update context after each action
      3. Generate context summary for other agents
      4. Detect context mismatches (expected vs actual state)

      CONTEXT DATA:
      - current_url
      - visible_data_attributes[]
      - breadcrumb[]
      - step_history[]
      - last_action

    capture_script: |
      // JavaScript executed on page to extract context
      () => ({
        url: window.location.href,
        visible_elements: Array.from(document.querySelectorAll('[data-*]'))
          .filter(el => el.offsetParent !== null)  // Only visible
          .map(el => Array.from(el.attributes)
            .filter(a => a.name.startsWith('data-'))
            .map(a => `${a.name}='${a.value}'`))
          .flat(),
        breadcrumb: Array.from(document.querySelectorAll('[data-breadcrumb] span'))
          .map(el => el.textContent.trim()),
        page_title: document.title
      })

    summary_prompt: |
      Generate concise context summary for selector agents:

      Current state:
      - URL: {url}
      - Module: {detected_module}
      - Visible key elements: {visible_count} data-* attributes
      - Last action: {last_action}

      Summarize in 2-3 sentences what page user is on and what's visible.

  # -------------------------------------------------------------------------
  # LearningAgent - Continuous Learning from Failures
  # -------------------------------------------------------------------------
  learning_agent:
    enabled: true
    model: "gpt-4o-mini"
    temperature: 0.1
    max_tokens: 1000

    system_prompt: |
      You are the Learning Agent. Learn from failures and human corrections.

      RESPONSIBILITIES:
      1. Query memory for similar past failures/corrections
      2. Match current context to stored patterns
      3. Suggest high-confidence corrections
      4. Store new failure patterns

      PRIORITY ORDER:
      1. Human corrections (confidence: 0.95+)
      2. Verified runtime successes (confidence: 0.85+)
      3. Learned patterns (confidence: varies)

    retrieval_prompt: |
      CURRENT SITUATION:
      - Step: {step_text}
      - Context: URL={url}, Visible={visible_elements}

      QUERY MEMORY:
      Search for similar situations where:
      1. Same/similar step text (semantic match)
      2. Similar context (URL pattern, visible elements)
      3. Has correction or verified selector

      If found high-confidence match (>0.85), return:
      {
        "found": true,
        "selector": "...",
        "confidence": 0.95,
        "source": "human_feedback",
        "reasoning": "..."
      }

    storage_prompt: |
      Store this failure pattern for future learning:

      - Step: {step_text}
      - Wrong selector: {failed_selector}
      - Context: {execution_context}
      - Reason: {failure_reason}

      Generate:
      1. Context pattern (URL pattern, required visible elements)
      2. Similar scenarios where this would apply
      3. Suggested alternatives (if any)

  # -------------------------------------------------------------------------
  # SelectorAgent_L1 - RAG + LLM Validation
  # -------------------------------------------------------------------------
  selector_agent_l1:
    enabled: true
    model: "gpt-4o"
    temperature: 0.2
    max_tokens: 1000

    confidence_threshold: 0.75
    retry_threshold: 0.70

    # Pre-RAG: Enhance query with context
    pre_rag_prompt: |
      TASK: Generate enhanced search query for RAG selector database.

      CONTEXT:
      - Current page: {current_url}
      - Visible elements: {visible_elements}
      - Last action: {last_action}
      - Module: {module}

      STEP TO EXECUTE:
      "{step_text}"

      QUESTION:
      What enhanced query should we use for semantic search to find the right selector?
      Consider current page state, what should be visible, and step intent.

      Return: Enhanced query string (one sentence)

    # Post-RAG: Validate candidates
    post_rag_prompt: |
      TASK: Validate RAG search results against current page context.

      RAG RETURNED THESE CANDIDATES:
      {candidates}

      CURRENT PAGE STATE:
      - URL: {current_url}
      - Visible elements: {visible_elements}

      QUESTION:
      Which candidate selector (if any) is feasible to execute now?
      Consider:
      1. Is the element type appropriate for current page?
      2. Is it likely visible based on visible_elements?
      3. Does the module/context match?

      Return JSON:
      {
        "selected_selector": "..." or null,
        "confidence": 0.0-1.0,
        "reasoning": "...",
        "recommendation": "execute" or "try_l2"
      }

  # -------------------------------------------------------------------------
  # SelectorAgent_L2 - DOM Discovery + LLM Analysis
  # -------------------------------------------------------------------------
  selector_agent_l2:
    enabled: true
    model: "gpt-4o"
    temperature: 0.2
    max_tokens: 1500

    activation_threshold: 0.70  # Activate if L1 < this

    dom_extraction_script: |
      // Extract visible interactive elements
      () => {
        const elements = document.querySelectorAll('button, input, select, a, [role="button"], [data-*]');
        return Array.from(elements)
          .filter(el => el.offsetParent !== null)  // Visible only
          .slice(0, 50)  // Limit to 50 elements
          .map((el, idx) => ({
            index: idx,
            tag: el.tagName.toLowerCase(),
            text: el.textContent.trim().substring(0, 50),
            attributes: Array.from(el.attributes).reduce((acc, attr) => {
              acc[attr.name] = attr.value;
              return acc;
            }, {}),
            has_data_attr: Array.from(el.attributes).some(a => a.name.startsWith('data-'))
          }));
      }

    analysis_prompt: |
      TASK: Identify the target element from live DOM and generate selector.

      STEP TO EXECUTE:
      "{step_text}"

      CURRENT CONTEXT:
      - URL: {current_url}
      - Module: {module}

      FOUND ELEMENTS ON PAGE:
      {dom_elements}

      QUESTION:
      Which element matches the step intent? Generate best selector.

      PRIORITY ORDER:
      1. data-* attributes (most stable)
      2. aria-* attributes (accessible)
      3. id (if meaningful)
      4. class (if specific)
      5. text-based (least stable)

      Return JSON:
      {
        "selected_element_index": 0-49 or null,
        "selector": "...",
        "confidence": 0.0-1.0,
        "reasoning": "..."
      }

  # -------------------------------------------------------------------------
  # SelectorAgent_L3 - Vision-Based Discovery
  # -------------------------------------------------------------------------
  selector_agent_l3:
    enabled: true
    model: "gpt-4o"  # Vision-enabled
    temperature: 0.1
    max_tokens: 1000

    vision_prompt: |
      TASK: Identify UI element from screenshot and generate selector.

      [Screenshot of current page attached]

      STEP TO EXECUTE:
      "{step_text}"

      CONTEXT:
      - Current URL: {current_url}
      - Module: {module}
      - Previous attempts failed with L1 and L2

      INSTRUCTIONS:
      1. Visually identify the target element described in the step
      2. Note its location, text, and visual characteristics
      3. Generate stable CSS or XPath selectors
      4. Prioritize: data-* > aria-* > id > class > text-based

      Return JSON:
      {
        "element_identified": true/false,
        "element_description": "Location and visual details",
        "suggested_selectors": ["selector1", "selector2", "selector3"],
        "confidence": 0.0-1.0,
        "reasoning": "Why this element matches the step"
      }

# ============================================================================
# MEMORY CONFIGURATION
# ============================================================================

memory:
  # Context Agent - Recent execution history
  context_agent:
    type: "short_term"
    retention_steps: 10              # Keep last 10 steps (configurable)
    retention_strategy: "sliding_window"
    include_fields:
      - current_url
      - visible_data_attributes
      - step_outcome
      - timestamp
      - breadcrumb

  # Learning Agent - Persistent learning
  learning_agent:
    type: "long_term"
    storage: "chromadb:learning_collection"
    max_patterns: 1000               # Max stored patterns
    similarity_threshold: 0.85        # For context matching
    cleanup_strategy: "keep_high_confidence"
    cleanup_after_days: 30           # Remove low-conf patterns after 30 days

  # Selector Agents - Per-step attempts
  selector_agents:
    type: "session"
    retention_steps: 3                # Keep last 3 attempts per step
    clear_on_success: true            # Clear failed attempts after success
    store_reasoning: true             # Keep LLM reasoning

  # Orchestrator - Full test run
  orchestrator:
    type: "full_session"
    max_steps: 100                    # Safety limit per test
    persist_to_file: "Logs/orchestrator_decisions.json"

# ============================================================================
# LANGGRAPH WORKFLOW CONFIGURATION
# ============================================================================

langgraph:
  workflow:
    max_iterations: 200               # Max agent calls per test
    timeout_seconds: 600              # 10 minutes max per test

    # State checkpointing
    checkpointing:
      enabled: true
      save_every_n_steps: 5
      checkpoint_dir: "Logs/checkpoints"

    # Error handling
    error_handling:
      max_retries_per_agent: 2
      fallback_strategy: "skip_step"  # skip_step, abort_test, continue
      log_errors: true

# ============================================================================
# ARTIFACTS GENERATION
# ============================================================================

artifacts:
  # Video Recording
  video_recording:
    enabled: true
    format: "webm"
    quality: "medium"                 # low, medium, high
    path: "Videos/{ticket_id}_{timestamp}.webm"
    fps: 25
    capture_context_overlay: false    # Overlay step info on video

  # Python Script Generation
  script_generation:
    enabled: true
    framework: "playwright"           # playwright, selenium
    test_runner: "pytest"             # pytest, unittest
    path: "Generated_Scripts/{ticket_id}_{module}_{timestamp}.py"

    include_in_script:
      - selectors_used: true
      - wait_times: true
      - assertions: true
      - comments: true                # Step descriptions
      - context_checks: true          # URL/state validation
      - agent_metadata: false         # L1/L2/L3 info (debug mode only)

    template: |
      # Auto-generated test script
      # Ticket: {ticket_id}
      # Module: {module}
      # Generated: {timestamp}
      # Agent chain: {agent_chain}

      import pytest
      from playwright.sync_api import Page, expect

      def test_{ticket_id}(page: Page):
          # Test steps with actual selectors used
          {generated_steps}

  # Enhanced HTML Report
  report_generation:
    enabled: true
    template: "templates/report_sequential_context.html"
    path: "Reports/{ticket_id}_{timestamp}_report.html"

    include_sections:
      - execution_summary: true
      - context_trace: true           # Timeline of URL/context changes
      - agent_decisions: true         # Orchestrator routing decisions
      - llm_reasoning: true           # LLM thought process per step
      - video_player: true            # Embedded video
      - script_download: true         # Link to generated script
      - failure_analysis: true        # Why steps failed (if any)

  # Context Trace Export (Debug)
  context_trace:
    enabled: true
    format: "json"
    path: "Logs/context_trace_{ticket_id}_{timestamp}.json"
    include:
      - url_history: true
      - dom_snapshots: true           # Visible elements per step
      - agent_reasoning: true
      - selector_attempts: true

# ============================================================================
# EXECUTION SETTINGS
# ============================================================================

execution:
  headless: false
  max_retries: 3
  screenshot_on_every_step: true

  # Failure Handling
  failure_handling:
    fail_fast: true                   # Stop on first failure
    capture_failure_screenshot: true
    capture_failure_dom: true
    log_detailed_error: true
    store_to_learning: true           # Add failure to learning collection
```

---

## Artifacts Generation

### 1. Video Recording

**Implementation:**
```python
# In plcd_taseq.py - Initialize browser with video
context = browser.new_context(
    record_video_dir="Videos/",
    record_video_size={"width": 1920, "height": 1080}
)

page = context.new_page()

# After test execution
video_path = page.video.path()
logger.info(f"Video saved: {video_path}")
```

**Output:**
- Format: `.webm` (Playwright native)
- Location: `Videos/RBPLCD-8835_20251118_151728.webm`
- Size: Configurable quality (low/medium/high)

---

### 2. Python Script Generation

**Agent:** `ScriptGeneratorAgent`

**Process:**
1. Read executed step results
2. Generate pytest test function
3. Include actual selectors used
4. Add context validation (URL checks)
5. Format with proper waits and assertions

**Template:**
```python
"""
Auto-generated test script
Ticket: RBPLCD-8835
Module: Teststep
Generated: 2025-11-18 15:17:28
Agent chain: L1 → L2 → L1 → L1
"""

import pytest
from playwright.sync_api import Page, expect

def test_RBPLCD_8835(page: Page):
    """Test automation for RBPLCD-8835"""

    # Step 1: Login
    page.goto("http://fe0vm03313.de.bosch.com/rbplcd_t/client/login")
    page.fill('input[type="text"]', "mechanic")
    page.fill('input[type="password"]', "avalon")
    page.click('[data-loginBtn="loginBtn"]')  # L1, conf: 0.71
    page.wait_for_timeout(3000)

    # Context check: Should be on dashboard
    expect(page).to_have_url(/.*dashboard/)

    # Step 2: Navigate to Runs
    page.click('[data-navitem="runs"]')  # L2 (DOM discovery), conf: 0.92
    page.wait_for_timeout(2000)

    # Context check: Should be on runs page
    expect(page).to_have_url(/.*runs/)

    # Step 3: Click Add button
    page.click('[data-addButton="addButton"]')  # L1, conf: 0.85
    page.wait_for_timeout(1000)

    # Verification
    expect(page.locator('[data-addDialog]')).to_be_visible()
```

**Output:** `Generated_Scripts/RBPLCD-8835_Teststep_20251118_151728.py`

---

### 3. Enhanced HTML Report

**Sections:**

#### A. Execution Summary
- Ticket ID, module, overall status
- Total steps, passed, failed
- Execution time
- Video link, script download

#### B. Context Trace Timeline
```
Timeline:
  [15:16:34] Login → /client/login
  [15:16:50] Dashboard loaded → /client/dashboard
    Visible: data-dashboard, data-sidebar, data-navitem
  [15:16:54] Navigate attempted → Failed (wrong selector)
  [15:17:24] Timeout error
```

#### C. Agent Decisions
```
Step 3: Navigate to Runs
  Orchestrator: Check LearningAgent → No match found
  Orchestrator: Route to SelectorAgent_L1
    L1: Enhanced query "Navigation link from Dashboard to Runs"
    L1: RAG returned 3 candidates
    L1: LLM validation → Confidence too low (0.36)
    L1: Recommendation: Try L2
  Orchestrator: Route to SelectorAgent_L2
    L2: Scraped 23 DOM elements
    L2: LLM identified: <a data-navitem="runs">
    L2: Generated selector: [data-navitem='runs']
    L2: Confidence: 0.92 ✓
  Orchestrator: Execute action
```

#### D. LLM Reasoning (Expandable)
```
Step 3 - SelectorAgent_L1 Post-RAG Validation:

Prompt:
  RAG returned: [data-routerLinkNameTestStep='{{row.aeName}}']
  Visible elements: data-sidebar, data-navitem, data-dashboard

Response:
  "This selector is a table row (tr element) used for clicking items
   inside a data table. Current context is Dashboard with visible
   sidebar navigation. The selector won't work here because no table
   is rendered. Confidence: 0.36. Recommendation: Try L2 DOM discovery
   to find actual navigation element."
```

#### E. Embedded Video Player
```html
<video controls width="100%">
  <source src="../Videos/RBPLCD-8835_20251118_151728.webm" type="video/webm">
</video>
```

**Output:** `Reports/RBPLCD-8835_20251118_151728_report.html`

---

### 4. Context Trace JSON (Debug)

**Purpose:** Full execution details for debugging

**Structure:**
```json
{
  "ticket_id": "RBPLCD-8835",
  "execution_time": "54.5s",
  "steps": [
    {
      "step_number": 1,
      "step_text": "Login",
      "context_before": {
        "url": "/client/login",
        "visible_elements": ["data-loginBtn", "data-userNameInput"],
        "timestamp": "2025-11-18T15:16:34"
      },
      "agent_chain": ["LearningAgent", "SelectorAgent_L1"],
      "selector_used": "[data-loginBtn='loginBtn']",
      "llm_reasoning": {
        "l1_pre_rag": "Login button on auth page",
        "l1_post_rag": "High confidence match, element visible"
      },
      "execution_result": "success",
      "context_after": {
        "url": "/client/dashboard",
        "visible_elements": ["data-dashboard", "data-sidebar"],
        "timestamp": "2025-11-18T15:16:50"
      }
    }
  ],
  "orchestrator_decisions": [...],
  "learning_stored": [...]
}
```

**Output:** `Logs/context_trace_RBPLCD-8835_20251118_151728.json`

---

## Integration Flow

### End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST EXECUTION (plcd_taseq.py)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        Execute test → Some steps fail → Generate artifacts:
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     HTML Report          Video              Script
     (with context)    (recording)        (partial)
          │
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              TESTER REVIEWS REPORT                               │
│  - Sees failed steps with context (URL, visible elements)       │
│  - Sees LLM reasoning (why selector was chosen)                 │
│  - Sees agent chain (L1→L2→L3 attempts)                         │
│  - Watches video to understand what happened                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        Tester decides: "I know the correct selector"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           FEEDBACK COLLECTION (feedback_taseq.py)               │
│                                                                  │
│  $ python feedback_taseq.py RBPLCD-8835_report.html             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  ReportParserAgent                                │
     │  - Extracts steps with full context              │
     │  - Shows tester: step, context, tried selector   │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  FeedbackCollectorAgent                          │
     │  - Interactive CLI:                              │
     │    "Enter correct selector: ___"                 │
     │    "Add context tags: ___"                       │
     │  - Validates input                               │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  ContextEnrichmentAgent                          │
     │  - Analyzes correction context                   │
     │  - LLM suggests patterns:                        │
     │    "This applies to all navigation from Dashboard"│
     │  - Generates context metadata                    │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  LearningStorageAgent                            │
     │  - Embeds correction with context                │
     │  - Stores to ChromaDB:                           │
     │    {                                             │
     │      selector: "[data-navitem='runs']",          │
     │      context: {url: "/dashboard", ...},          │
     │      confidence: 0.98,                           │
     │      source: "human_feedback"                    │
     │    }                                             │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
                  Correction stored to ChromaDB
                  with high priority (0.95+)
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              NEXT TEST RUN (plcd_taseq.py)                      │
│                                                                  │
│  Same ticket or similar context encountered...                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  OrchestratorAgent                               │
     │  - Routes to LearningAgent first                 │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  LearningAgent                                   │
     │  - Queries ChromaDB:                             │
     │    "Navigate to Runs + context: /dashboard"     │
     │  - FINDS human correction!                       │
     │  - Returns: [data-navitem='runs'], conf: 0.98   │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────┐
     │  OrchestratorAgent                               │
     │  - Sees confidence 0.98 (> 0.90 threshold)       │
     │  - Decision: Skip L1/L2/L3, use correction       │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
                    Execute directly → SUCCESS! ✓
                              │
                              ▼
                  Store success to LearningAgent
                  (increases usage_count for this pattern)
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `plcd_taseq.py` skeleton with LangGraph setup
- [ ] Implement `TestExecutionState` typed dict
- [ ] Create base agent classes
- [ ] Setup ChromaDB collections (learning_collection)
- [ ] Update `plcdtestassistant.yaml` with agent configurations

### Phase 2: Context Tracking
- [ ] Implement `ContextAgent`
- [ ] Create DOM extraction JavaScript
- [ ] Build context memory (sliding window)
- [ ] Test context capture on real page

### Phase 3: Jira Parsing
- [ ] Implement `JiraAgent` with LLM
- [ ] Add format examples to YAML
- [ ] Test with various ticket formats
- [ ] Validate output structure

### Phase 4: Selector Discovery Agents
- [ ] Implement `SelectorAgent_L1` (RAG + LLM)
  - [ ] Pre-RAG query enhancement
  - [ ] Post-RAG validation
- [ ] Implement `SelectorAgent_L2` (DOM + LLM)
  - [ ] DOM scraping
  - [ ] LLM element matching
- [ ] Implement `SelectorAgent_L3` (Vision)
  - [ ] Screenshot capture
  - [ ] LLM-Vision analysis

### Phase 5: Learning System
- [ ] Implement `LearningAgent`
- [ ] Create retrieval logic (semantic + context matching)
- [ ] Implement storage logic
- [ ] Test learning across runs

### Phase 6: Orchestrator
- [ ] Implement `OrchestratorAgent`
- [ ] Create routing decision logic
- [ ] Build LangGraph workflow
- [ ] Add conditional edges

### Phase 7: Artifacts Generation
- [ ] Video recording integration
- [ ] Python script generator
- [ ] Enhanced HTML report template
- [ ] Context trace JSON export

### Phase 8: Feedback Tool
- [ ] Create `feedback_taseq.py`
- [ ] Implement `ReportParserAgent`
- [ ] Implement `FeedbackCollectorAgent`
- [ ] Implement `ContextEnrichmentAgent`
- [ ] Implement `LearningStorageAgent`
- [ ] Test feedback → storage → retrieval loop

### Phase 9: Testing & Validation
- [ ] Test with RBPLCD-8835 (current failing ticket)
- [ ] Test with multiple tickets
- [ ] Validate learning loop
- [ ] Validate memory management
- [ ] Performance testing

### Phase 10: Documentation
- [ ] User guide
- [ ] Agent configuration guide
- [ ] Troubleshooting guide
- [ ] Example customizations

---

## Success Metrics

1. **Context Accuracy:** 95%+ correct context capture
2. **Selector Discovery:** 85%+ success rate with context tracking
3. **Learning Effectiveness:** 90%+ reuse of human corrections
4. **Execution Speed:** < 2 minutes per test (10 steps)
5. **Memory Efficiency:** < 500MB RAM usage
6. **Agent Routing:** < 3 agents called per step on average

---

## Future Enhancements

1. **Multi-language Support:** Jira tickets in different languages
2. **Parallel Execution:** Run multiple tests simultaneously
3. **Advanced Vision:** Object detection for complex UIs
4. **Natural Language Steps:** "Click the blue button on the right"
5. **Self-Healing:** Automatically fix broken selectors without human feedback
6. **CI/CD Integration:** GitHub Actions, Jenkins plugins
7. **Dashboard UI:** Web interface for viewing tests and corrections
8. **Collaborative Learning:** Share corrections across team/organization

---

## Appendix

### A. LangGraph State Schema

```python
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class TestExecutionState(TypedDict):
    # Test metadata
    ticket_id: str
    module: str
    current_step: int
    total_steps: int

    # Context tracking
    context_history: List[Dict[str, Any]]
    current_context: Dict[str, Any]

    # Selector discovery
    selector_attempts: List[Dict[str, Any]]
    successful_selectors: Dict[str, str]

    # Learning
    failure_patterns: List[Dict[str, Any]]
    corrections_used: List[Dict[str, Any]]

    # Agent routing
    agent_chain: List[str]
    next_agent: str
    orchestrator_reasoning: List[str]

    # Execution results
    step_results: List[Dict[str, Any]]
    overall_status: str

    # Artifacts
    video_path: Optional[str]
    script_path: Optional[str]
    report_path: Optional[str]

    # Playwright objects (not serializable, stored separately)
    # page: Page
    # browser: Browser
```

### B. ChromaDB Collections

```python
# Collection: learning_collection
{
  "name": "learning_collection",
  "metadata": {"description": "Human corrections and verified selectors"},
  "embedding_function": "text-embedding-3-small",
  "documents": [
    {
      "id": "learning_001",
      "embedding": [...],
      "document": "Navigate to Runs from Dashboard using sidebar navigation",
      "metadata": {
        "selector": "[data-navitem='runs']",
        "step_text": "Navigate to Runs",
        "context": {
          "url_pattern": "*/dashboard",
          "required_visible": ["data-sidebar", "data-navitem"],
          "tags": ["navigation", "sidebar"]
        },
        "confidence": 0.98,
        "source": "human_feedback",
        "ticket_id": "RBPLCD-8835",
        "created": "2025-11-18T15:30:00",
        "usage_count": 5,
        "verified": true
      }
    }
  ]
}

# Collection: context_snapshots (optional, for debugging)
{
  "name": "context_snapshots",
  "metadata": {"description": "Page context snapshots during execution"},
  "documents": [
    {
      "id": "context_snapshot_001",
      "document": "Dashboard page with sidebar navigation visible",
      "metadata": {
        "ticket_id": "RBPLCD-8835",
        "step_number": 2,
        "timestamp": "2025-11-18T15:16:53",
        "url": "/client/dashboard",
        "visible_elements": ["data-sidebar", "data-navitem='runs'", ...],
        "breadcrumb": ["Home", "Dashboard"]
      }
    }
  ]
}
```

---

**End of Design Document**

*Last Updated: 2025-11-18*
*Version: 2.0 - Sequential Context Tracking*
*Authors: PLCD Test Automation Team*
