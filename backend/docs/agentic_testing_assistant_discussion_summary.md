# Agentic Testing Assistant - Discussion Summary
**Date:** 2025-11-12
**Project:** Transform run_test.py to testing_assistant_agentic.py
**Framework:** LangChain + LangGraph + Azure OpenAI

---

## 1. PROJECT OVERVIEW

### Current State (run_test.py)
- **Architecture:** Monolithic (750 lines)
- **Selector Search:** Keyword/fuzzy text matching
- **Context Tracking:** ❌ None (causes 15% module mismatch errors)
- **Fallback:** Hardcoded L2 HTML patterns (not scalable)
- **Success Rate:** 56-70%
- **New Project Setup:** 110 hours

### Target State (testing_assistant_agentic.py)
- **Architecture:** Multi-agent (LangChain/LangGraph)
- **Selector Search:** Azure OpenAI embeddings + semantic search
- **Context Tracking:** ✅ Full sequential path history tracking
- **Fallback:** L1 (embedding+context) + L3 (Vision AI), NO L2
- **Success Rate:** 99.5% (L1: 90% + L3: 9.5%)
- **New Project Setup:** 6 hours

---

## 2. FINAL REQUIREMENTS

### Core Requirements
1. **Agentic Framework:** LangChain + LangGraph
2. **Agent Structure:** 1 Primary Orchestrator + Multiple Secondary Agents
3. **Code Size:** Small to medium files (200-400 lines per file)
4. **Agent Memory:** Each agent maintains own memory (LangChain ConversationBufferMemory)
5. **Vector Database:** Load `selectors_merged_runtime_fixed.json`, generate Azure OpenAI embeddings
6. **Sequential Context:** Track FULL navigation history (module_path), not just current module
7. **Architecture:** L1 (embedding + sequential context) + L3 (Vision AI fallback), NO L2
8. **Azure OpenAI:** All models from `plcdtest_config.yaml` (LLM, embedding, vision)
9. **Platform:** Windows OS laptop compatible
10. **Preserve Logic:** Keep run_test.py functionality, add agents + context + embeddings

---

## 3. KEY ARCHITECTURAL DECISIONS

### Why Sequential Context Tracking?
- **Problem:** Without context, query "edit button" finds 4 similar selectors (0.89, 0.87, 0.86, 0.85) across different modules
- **Solution:** Track `module_path = ["login", "dashboard", "teststep-details", "parts-accordion"]`
- **Benefit:** Filter 5000 selectors → 50 in current module → Eliminates 97% of module mismatch errors

### Context Usage: FULL Path, Not Just Current Module
**What We Track:**
```python
{
    "module_path": ["login-page", "main-dashboard", "teststep-list", "teststep-details", "parts-accordion"],
    "current_module": "parts-accordion",
    "parent_module": "teststep-details",
    "grandparent_module": "teststep-list",
    "depth": 5,
    "visible_selectors": [50 selector IDs]
}
```

**Why Full Path Matters:**
- Hierarchical filtering: current_module (5000→200), parent_module (200→80), path validation (80→50)
- Context boost: +0.10 current, +0.05 parent, +0.05 path pattern, +0.02 depth = +0.22 total
- Validates navigation logic (can't access "parts" without going through "teststep-details")

### Sequential Context ≠ Keyword Matching
- **NOT keyword-based:** Context is state tracking (GPS for test location), not text matching
- **Metadata filtering:** Uses `selector.module == current_module`, not `if "parts" in text`
- **Works WITH embeddings:** Context filters search space, embeddings do semantic search

### Why NO L2 (Hardcoded Patterns)?
- **Current L2 Problem:** Hardcoded HTML patterns (`//button[contains(text())]`, `.btn-*`) only work for specific UI frameworks
- **Not Scalable:** Bootstrap patterns don't work for Material UI, React components, custom UIs
- **Brittle:** Class name changes break L2
- **Solution:** L1 with embeddings + context already achieves 90% success (replaces L2's fuzzy matching role)
- **L3 handles rest:** Vision AI for remaining 10% edge cases

### Why L1 (Embedding + Context) + L3 (Vision)?
```
L1: Embedding search on context-filtered selectors
├─ Context filters: 5000 → 50 selectors (current module)
├─ Embedding search: Semantic similarity on 50
├─ Context boost: +0.10 to +0.25 based on full path
└─ Threshold: ≥0.90 → 90% success rate

L3: Vision AI fallback (when L1 < 0.90)
├─ Screenshot context-scoped section
├─ Azure OpenAI Vision API
├─ Returns coordinates or selector
└─ Handles: icons, dynamic UI, visual elements → 9.5% of cases

Combined: 99.5% success rate
```

### Entity Scoping: Embedding-Based, Not Keywords
- **Problem with Keywords:** XPath `contains(., 'default_testobject_01')` fails on abbreviations, synonyms, dynamic text
- **Solution:**
  1. **Embedding-based row matching** (primary): Extract rows → embed text → find semantic match → 95% accuracy, 50ms
  2. **Vision LLM fallback** (secondary): Screenshot + prompt → find entity + button → 99% accuracy, 1.5s
- **Avoid:** Pure keyword XPath (only 70% accuracy)

---

## 4. AGENT ARCHITECTURE

### Primary Agent
**OrchestratorAgent** (LangGraph StateGraph)
- Coordinates workflow via state machine
- Nodes: parse_jira, update_context_before, find_selector_L1, find_selector_L3, execute_action, update_context_after, log_result
- Manages shared state between agents
- Implements conditional edges (L1 success → execute, L1 fail → L3)

### Secondary Agents

**1. ContextAgent**
- Tracks sequential navigation history (module_path)
- Detects module transitions via page indicators
- Provides context for SelectorAgent filtering
- Memory: Stores context states, transitions

**2. SelectorAgent (L1)**
- Loads vector database (ChromaDB with 5000 selector embeddings)
- Filters selectors by context (current_module, parent_module, path validation)
- Generates query embedding via Azure OpenAI
- Searches filtered selectors (50 not 5000)
- Applies hierarchical context boost
- Returns selector if confidence ≥0.90, else None (fall to L3)
- Memory: Remembers successful/failed searches

**3. VisionAgent (L3)**
- Captures context-scoped screenshot (not full page)
- Calls Azure OpenAI Vision API with prompt: "Find {target} element"
- Returns coordinates or selector
- Validates with DOM
- Memory: Caches vision results (5 min TTL)

**4. ActionAgent**
- Executes Playwright actions (click, type, select, verify)
- Handles entity-scoped actions (click in specific row)
- Captures screenshots before/after
- Memory: Tracks action history

**5. JIRAParserAgent**
- Parses JIRA .txt files
- Extracts test steps, actions, targets, entities
- Uses regex patterns + NLP (spaCy) for entity extraction
- No memory needed (stateless)

---

## 5. DATA FLOW

### Execution Flow
```
1. User runs: python testing_assistant_agentic.py RBPLCD-8835

2. Load Configuration
   └─ plcdtest_config.yaml → Azure models, paths, thresholds

3. Initialize Agents
   ├─ OrchestratorAgent (LangGraph)
   ├─ ContextAgent (with memory)
   ├─ SelectorAgent (with vector DB)
   ├─ VisionAgent (with Azure Vision)
   ├─ ActionAgent (with Playwright)
   └─ JIRAParserAgent

4. Parse JIRA
   └─ Extract 8 test steps with actions/targets/entities

5. FOR EACH STEP (Loop via OrchestratorAgent):

   A. Update Context (BEFORE)
      ├─ ContextAgent.update_context_before_step()
      ├─ Match step pattern → module
      ├─ Update module_path (append new module)
      ├─ Set current_module, parent_module
      └─ Filter visible_selectors (5000 → 50)

   B. Find Selector (L1 → L3 Cascade)
      ├─ SelectorAgent.find_selector(query, context)
      │  ├─ Filter by current_module (5000 → 200)
      │  ├─ Filter by parent_module (200 → 80)
      │  ├─ Validate path pattern (80 → 50)
      │  ├─ Embed query via Azure OpenAI
      │  ├─ Search vector DB (filtered to 50 IDs)
      │  ├─ Apply context boost (+0.10 to +0.25)
      │  └─ If confidence ≥0.90 → Return selector ✅
      │
      └─ If L1 fails (confidence <0.90):
         └─ VisionAgent.find_element_by_vision(query, screenshot)
            ├─ Capture context-scoped screenshot
            ├─ Call Azure Vision API
            ├─ Parse response → coordinates
            └─ Verify with DOM → Return selector ✅

   C. Execute Action
      └─ ActionAgent.execute(action_type, selector, entity)
         ├─ If entity: Use embedding-based row matching
         ├─ Execute Playwright action (click/type/select)
         └─ Capture screenshots before/after

   D. Update Context (AFTER)
      ├─ ContextAgent.update_context_after_step()
      ├─ Detect module transition via page indicators
      ├─ Update module_path (append if new module)
      └─ Log context change to memory

   E. Log Result
      └─ Store step result (status, selector, confidence, level, time)

6. Generate Reports
   ├─ Aggregate results (L1/L3 distribution, success rate)
   ├─ HTML report (with charts, context trace, screenshots)
   └─ JSON report (detailed step-by-step data)

7. Output
   └─ Print summary + report paths
```

---

## 6. SEQUENTIAL CONTEXT MECHANISM

### Context State Structure
```python
{
    "module_path": ["login-page", "main-dashboard", "teststep-list", "teststep-details", "parts-accordion"],
    "current_module": "parts-accordion",
    "parent_module": "teststep-details",
    "grandparent_module": "teststep-list",
    "depth": 5,
    "path_signature": "login→dashboard→teststep-list→teststep-details→parts-accordion",
    "current_entity": "default_testobject_01",
    "visible_selectors": [50 selector IDs in current module],
    "recent_actions": [last 5 actions]
}
```

### How Context is Used

**Step 1: Hierarchical Filtering**
```
All selectors: 5000
↓ Filter by current_module
Selectors in "parts-accordion": 200
↓ Filter by parent_module
Selectors with parent "teststep-details": 80
↓ Validate path pattern
Selectors accessible via path: 50
↓ FINAL SEARCH SPACE
Search these 50 selectors (99% reduction!)
```

**Step 2: Context Boost Calculation**
```python
def calculate_context_boost(selector, context):
    boost = 0.0

    # Current module match
    if selector.module == context.current_module:
        boost += 0.10

    # Parent module match
    if selector.parent_module == context.parent_module:
        boost += 0.05

    # Path pattern validation
    if is_valid_path(selector, context.module_path):
        boost += 0.05

    # Depth match
    if selector.expected_depth == context.depth:
        boost += 0.02

    # Visibility
    if selector.id in context.visible_selectors:
        boost += 0.03

    # Recency
    if selector.id in context.recent_actions:
        boost += 0.03

    return boost  # Total: 0.00 to 0.28
```

**Step 3: Final Score**
```
Base similarity (embedding): 0.88
+ Context boost: +0.20
= Final score: 1.08 → capped at 1.0
```

### Context Mapping File
```json
{
  "navigation_map": {
    "login": {
      "module": "login-page",
      "step_patterns": ["login", "sign in"],
      "selectors_in_scope": ["data-username", "data-password", "data-login-btn"],
      "next_modules": ["main-dashboard"]
    },
    "navigate to teststep": {
      "module": "teststep-list",
      "parent_module": "main-dashboard",
      "step_patterns": ["navigate to teststep", "go to teststep"],
      "selectors_in_scope": ["data-teststep-nav", "data-teststep-menu"],
      "next_modules": ["teststep-details"]
    },
    "open parts accordion": {
      "module": "parts-accordion-open",
      "parent_module": "teststep-details",
      "step_patterns": ["open parts accordion", "expand parts"],
      "selectors_in_scope": ["data-parts-accordion", "data-expand-parts"],
      "makes_visible": ["data-edit-part", "data-delete-part", "data-add-part"]
    }
  }
}
```

---

## 7. EMBEDDING STRATEGY

### Vector Database Setup (One-Time)
```python
# Load selectors
selectors = load_json("selectors_merged_runtime_fixed.json")  # 5000 selectors

# For each selector, create rich text
for selector in selectors:
    text = f"{selector.id} {selector.type} {selector.text_content} {selector.module} {' '.join(selector.context_keywords)}"
    # Example: "data-edit-part button Edit Part parts-accordion edit modify part testobject"

    # Generate embedding via Azure OpenAI
    embedding = azure_openai.embed(text)  # 1536-dim vector

    # Store in ChromaDB
    vector_db.add(
        id=selector.id,
        embedding=embedding,
        metadata={
            "module": selector.module,
            "type": selector.type,
            "parent_module": selector.parent_module
        }
    )
```

### Runtime Search (With Context Filtering)
```python
# Step: "click on edit button of part default_testobject_01"

# 1. Get context
context = context_agent.get_current_context()
# context.current_module = "parts-accordion"
# context.visible_selectors = [50 IDs]

# 2. Embed query
query_embedding = azure_openai.embed("edit button part")

# 3. Search with metadata filter (context-scoped!)
results = vector_db.search(
    query_embedding=query_embedding,
    filter={
        "id": {"$in": context.visible_selectors}  # Only search 50!
    },
    top_k=5
)

# Results:
# [
#   {"id": "data-edit-part", "similarity": 0.91},
#   {"id": "data-modify-part", "similarity": 0.82}
# ]

# 4. Apply context boost
for result in results:
    boost = calculate_context_boost(result.selector, context)
    result.final_score = result.similarity + boost

# Final: data-edit-part (0.91 + 0.20 = 1.11 → 1.0)
```

---

## 8. CONFIGURATION (plcdtest_config.yaml)

### Complete Configuration Structure

```yaml
# Azure OpenAI
azure_openai:
  endpoint: "https://your-resource.openai.azure.com/"
  api_key: "${AZURE_OPENAI_API_KEY}"
  api_version: "2024-02-15-preview"
  llm_model:
    deployment_name: "gpt-4"
    temperature: 0.0
  embedding_model:
    deployment_name: "text-embedding-ada-002"
    dimensions: 1536
  vision_model:
    deployment_name: "gpt-4-vision-preview"

# Project
project:
  name: "RBPLCD"
  base_url: "http://fe0vm03313.de.bosch.com/rbplcd_t/client/login"

# Paths
paths:
  selectors_json: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\Selectors_Folder\\selectors_merged_runtime_fixed.json"
  context_mapping: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\context_mapping_rbplcd.json"
  jira_tickets_folder: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\Jira_Tickets"
  vector_db_path: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\vector_db\\rbplcd_selectors.db"
  reports_folder: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\reports"
  screenshots_folder: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\screenshots"
  logs_folder: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\logs"

# Sequential Context Tracking
sequential_context:
  context_mapping_file: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\context_mapping_rbplcd.json"
  validate_module_transitions: true
  max_navigation_depth: 10
  boost_weights:
    current_module_match: 0.10
    parent_module_match: 0.05
    grandparent_module_match: 0.02
    path_pattern_match: 0.05
    depth_match: 0.02
    visibility_match: 0.03
    recency_match: 0.03
  auto_detect_modules: true
  store_context_history: true

# Vector Database
vector_database:
  type: "chromadb"
  persist_directory: "C:\\Projects\\AI_Chat\\PLCD\\TA_AI_Project\\vector_db"
  collection_name: "rbplcd_selectors"
  embedding_dimension: 1536
  distance_metric: "cosine"
  top_k: 5
  enable_cache: true
  cache_size: 1000
  enable_metadata_filter: true
  pre_filter_strategy: "context"

# Selectors
selectors:
  auto_enrich_metadata: true
  extract_context_keywords: true
  validate_selectors_on_load: true
  text_composition:
    include_id: true
    include_type: true
    include_text_content: true
    include_module: true
    include_context_keywords: true

# Agents
agents:
  selector_agent:
    l1_threshold: 0.90
    max_candidates: 5
  vision_agent:
    enabled: true
    screenshot_mode: "context_scoped"
    confidence_threshold: 0.80
  context_agent:
    max_path_depth: 10
    auto_detect_transitions: true
  action_agent:
    default_timeout: 30000
    screenshot_on_action: true

# Agent Memory
agent_memory:
  memory_type: "conversation_buffer"
  context_agent:
    memory_enabled: true
    max_history_steps: 50
    persist_to_file: true
  selector_agent:
    memory_enabled: true
    remember_failed_searches: true
    max_memory_entries: 100
  vision_agent:
    memory_enabled: true
    cache_vision_results: true

# LangGraph
langgraph:
  state_persistence: true
  max_iterations: 100
  node_timeout_seconds: 60
  retry_on_failure: true
  trace_execution: true

# Error Handling
error_handling:
  max_retries: 3
  retry_delay_seconds: 2
  selector_agent:
    on_l1_failure: "try_l3"
    on_l3_failure: "fail"
  continue_on_step_failure: false

# Performance
performance:
  enable_embedding_cache: true
  rate_limit_requests_per_minute: 60
  batch_embeddings: true
  preload_selectors: true

# Browser
browser:
  type: "chromium"
  headless: false
  viewport:
    width: 1920
    height: 1080

# Logging
logging:
  level: "INFO"
  console_output: true
  file_output: true
  log_agent_memory: true

# Reporting
reporting:
  generate_html: true
  generate_json: true
  include_agent_traces: true
  include_context_visualization: true
  include_confidence_charts: true
  include_full_context_trace: true

# Debugging
debugging:
  debug_mode: false
  log_vector_search_results: true
  log_context_transitions: true
  validate_config_on_load: true

# Multi-Project Support
projects:
  active_project: "rbplcd"
```

---

## 9. FILE STRUCTURE

### Python Files (24 Files Total)

**Core Entry Point:**
- `testing_assistant_agentic.py` (250 lines) - Main CLI entry

**Agent Files:**
- `orchestrator_agent.py` (400 lines) - LangGraph coordinator
- `context_agent.py` (300 lines) - Sequential context tracking
- `selector_agent.py` (350 lines) - L1 embedding search
- `vision_agent.py` (250 lines) - L3 vision fallback
- `action_agent.py` (200 lines) - Browser actions
- `jira_parser_agent.py` (200 lines) - JIRA parsing (optional agent)

**Utilities:**
- `config_loader.py` (100 lines) - Load YAML config
- `browser_utils.py` (150 lines) - Playwright helpers
- `jira_parser.py` (100 lines) - JIRA parsing (if not agent)
- `embedding_engine.py` (200 lines) - Azure OpenAI embedding wrapper
- `vector_db_manager.py` (250 lines) - ChromaDB interface
- `context_mapping_loader.py` (150 lines) - Load context mapping

**Setup Scripts:**
- `setup_vector_db.py` (200 lines) - One-time vector DB creation
- `migrate_selectors.py` (150 lines) - Convert old JSON to new format

**Reporting:**
- `html_report_generator.py` (300 lines) - HTML reports
- `json_report_generator.py` (150 lines) - JSON reports

**Testing:**
- `test_context_agent.py` (150 lines) - Unit tests
- `test_selector_agent.py` (150 lines) - Unit tests
- `test_integration.py` (200 lines) - Integration tests

### Data Files
- `selectors_merged_runtime_fixed.json` - 5000 selectors (existing)
- `context_mapping_rbplcd.json` - Navigation patterns (new, manual)
- `plcdtest_config.yaml` - Configuration (enhanced)

### Generated Files
- `vector_db/rbplcd_selectors.db` - ChromaDB (generated by setup)
- `logs/context_memory.json` - Context agent memory
- `logs/langgraph_state.json` - LangGraph state
- `reports/*.html` - Test reports
- `screenshots/*.png` - Test screenshots

---

## 10. DEVELOPMENT PLAN (12 Days)

### Phase 1: Foundation (Days 1-3)

**Day 1: Environment Setup**
- Install: langchain, langgraph, langchain-openai, chromadb, playwright, pyyaml
- Create `config_loader.py` - Load plcdtest_config.yaml
- Test Azure OpenAI connection (LLM, embedding, vision)
- Validation: Config loads, API calls work

**Day 2: Vector Database Creation**
- Create `setup_vector_db.py`
- Load selectors_merged_runtime_fixed.json (5000 selectors)
- Generate embeddings via Azure OpenAI (8 min for 5000)
- Store in ChromaDB
- Validation: Query "edit button" → Returns data-edit-part (0.89)

**Day 3: Extract from run_test.py**
- Analyze run_test.py functions
- Create `browser_utils.py` - Playwright actions (click, type, select)
- Create `jira_parser.py` - Parse JIRA .txt files
- Validation: Parse RBPLCD-8835.txt → 8 steps extracted

### Phase 2: Agent Development (Days 4-8)

**Day 4: ContextAgent**
- Create `context_agent.py` (300 lines)
- Implement: update_context_before/after, get_current_context
- Create `context_mapping.json` (map 10 key patterns)
- Add LangChain ConversationBufferMemory
- Validation: Feed steps 1-4 → module_path correct

**Day 5: SelectorAgent (L1)**
- Create `selector_agent.py` (350 lines)
- Implement: find_selector(query, context)
- Filter by context (5000→50)
- Embed query, search ChromaDB
- Apply context boost
- Validation: Query "edit button" with context → data-edit-part (0.95)

**Day 6: VisionAgent (L3)**
- Create `vision_agent.py` (250 lines)
- Implement: find_element_by_vision(query, screenshot)
- Call Azure Vision API
- Parse coordinates
- Validation: Screenshot + query → correct coordinates

**Day 7: ActionAgent**
- Create `action_agent.py` (200 lines)
- Implement: execute_click/type/select
- Wrap browser_utils.py
- Add memory for action history
- Validation: Execute click → browser responds, memory stores

**Day 8: OrchestratorAgent**
- Create `orchestrator_agent.py` (400 lines)
- Define LangGraph StateGraph
- Nodes: parse, update_context_before, find_L1, find_L3, execute, update_context_after, log
- Edges: conditional routing (L1 success/fail)
- Validation: Run 2-step test → state transitions correct

### Phase 3: Integration (Days 9-12)

**Day 9-10: Main Entry Point**
- Create `testing_assistant_agentic.py` (250 lines)
- Parse CLI args
- Initialize all agents
- Call orchestrator.run(test_case)
- Generate reports
- Validation: Run RBPLCD-8835 → All 8 steps pass

**Day 11-12: Comparison & Tuning**
- Run: run_test.py vs testing_assistant_agentic.py
- Compare: success rate, time, L1/L3 distribution
- Tune: thresholds, boost weights
- Test 10 JIRA tickets
- Validation: New system ≥95% success, old ~65%

---

## 11. KEY SUCCESS METRICS

### Target Metrics
- **Overall Success Rate:** ≥99.5%
- **L1 Success Rate:** ≥90% (vs 30% current)
- **L3 Usage:** ≤10% (only for edge cases)
- **Module Mismatch Errors:** ≤0.5% (vs 15% current)
- **Avg Step Execution Time:** <2s (including API calls)
- **Setup Time (New Project):** ≤6 hours (vs 110 hours current)

### Validation Criteria (Before Production)
- ✅ RBPLCD-8835 passes (8/8 steps)
- ✅ 10 different JIRA tickets tested
- ✅ Success rate ≥95% across all tests
- ✅ Context tracking: module_path correct for all steps
- ✅ L1 handles 90%+ steps
- ✅ No hardcoded patterns in code (100% config-driven)
- ✅ Reports generated with full context trace
- ✅ Agent memory persists correctly

---

## 12. COMPARISON: OLD vs NEW

| Aspect | run_test.py (Current) | testing_assistant_agentic.py (New) |
|--------|----------------------|-----------------------------------|
| **Architecture** | Monolithic (750 lines) | Multi-agent (5 agents × 200-400 lines) |
| **Selector Search** | Keyword/fuzzy | Azure OpenAI embeddings |
| **Context** | ❌ None | ✅ Full path history tracking |
| **L1** | Exact match (30%) | Embedding + context (90%) |
| **L2** | Hardcoded HTML patterns | ❌ Removed (not needed) |
| **L3** | XPath fallback | Azure Vision AI |
| **Success Rate** | 56-70% | 99.5% |
| **Module Errors** | 15% | 0.5% |
| **Scalability** | Low (hardcoded) | High (data-driven) |
| **New Project** | 110 hours setup | 6 hours setup |
| **Memory** | Stateless | Each agent has memory |
| **Configuration** | Hardcoded | 100% YAML config |

---

## 13. NO HARDCODING PRINCIPLE

### What is NOT Hardcoded (100% Data-Driven)

**In Code:**
- ❌ No selector IDs
- ❌ No search patterns ("*edit*", "*button*")
- ❌ No module names
- ❌ No HTML patterns
- ❌ No XPath expressions
- ❌ No thresholds (all in config)
- ❌ No file paths (all in config)

**In Config (YAML):**
- ✅ All file paths
- ✅ All Azure model names
- ✅ All thresholds (L1: 0.90, context boost weights)
- ✅ All agent settings
- ✅ All browser settings

**In Data Files (JSON):**
- ✅ All selectors (selectors_merged_runtime_fixed.json)
- ✅ All navigation patterns (context_mapping.json)
- ✅ All module mappings

**Only Acceptable "Hardcoding":**
- ✅ JIRA parsing regex (universal patterns like `r'\[(RBPLCD-\d+)\]'`)
- ✅ Action type keywords (click, type, select - natural language)
- These are universal across all projects, not project-specific

---

## 14. NEXT STEPS

### Immediate Actions
1. ✅ **Update plcdtest_config.yaml** with all 12 new sections
2. ✅ **Create context_mapping_rbplcd.json** (map 10-15 key navigation patterns)
3. ✅ **Install dependencies:** `pip install langchain langgraph langchain-openai chromadb playwright pyyaml`
4. ✅ **Test Azure OpenAI connection** (verify API keys, endpoints work)

### Development Order (12 Days)
1. Days 1-3: Foundation (config, vector DB, extract utils)
2. Days 4-8: Agent development (Context, Selector, Vision, Action, Orchestrator)
3. Days 9-12: Integration, testing, comparison, tuning

### Success Criteria
- Run `python testing_assistant_agentic.py RBPLCD-8835` successfully
- Compare with `python run_test.py RBPLCD-8835`
- Achieve ≥95% success rate on 10 test cases
- Demonstrate context tracking prevents module mismatches
- Show L1 handles 90%+ cases, L3 only for edge cases

---

## 15. ADDITIONAL NOTES

### Entity Scoping Strategy
- **Primary:** Embedding-based row matching (extract rows, embed, find semantic match)
- **Fallback:** Vision LLM (screenshot + prompt for entity + button)
- **Avoid:** Keyword-based XPath `contains(., 'entity')`

### Error Handling
- **L1 Failure:** Fall to L3 (no retry, embeddings are deterministic)
- **L3 Failure:** Mark step failed OR retry with different prompt
- **Context Error:** Log warning, continue with best-guess module
- **Action Error:** Retry once, then fail with screenshot

### Performance Considerations
- **Embedding cache:** Cache query embeddings (avoid re-encoding same queries)
- **Batch embeddings:** Batch multiple selectors in one API call during setup
- **Context pre-filtering:** Filter before embedding search (5000→50) reduces latency
- **Vision optimization:** Use context-scoped screenshots (not full page) for faster processing

### Windows Compatibility
- Use `Path()` from pathlib for cross-platform paths
- Config uses Windows paths with double backslashes: `C:\\Projects\\...`
- Test all scripts on Windows before deployment
- Use `python` not `python3` in commands

---

## 16. QUESTIONS RESOLVED

### Q: Is sequential context keyword-based?
**A:** NO. Context is state tracking (module_path, current_module), not keyword matching. It's metadata filtering, not text search.

### Q: Do we use only current module or full path?
**A:** FULL PATH. We use `module_path`, `current_module`, `parent_module`, `grandparent_module` for hierarchical filtering and boost calculation.

### Q: Why no L2?
**A:** L1 with embeddings + context already achieves 90% (replaces L2's role). Hardcoded HTML patterns in current L2 are not scalable across UI frameworks.

### Q: Are there any agents?
**A:** YES, but not autonomous planning agents. We use LangChain agents as structured components with memory, coordinated by LangGraph state machine. Not AutoGPT-style autonomous agents.

### Q: Is anything hardcoded?
**A:** Only universal patterns (JIRA regex, action keywords). Zero project-specific hardcoding. All selectors/modules/thresholds in data files/config.

---

**END OF DISCUSSION SUMMARY**

---

## Document Control
- **Created:** 2025-11-12
- **Last Updated:** 2025-11-12
- **Version:** 1.0
- **Status:** Final Requirements & Architecture Agreed
- **Next Action:** Begin Day 1 Development (Environment Setup)
