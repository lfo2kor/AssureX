# PLCD Testing Assistant - File Dependencies & Execution Flow

## File Dependency Tree

```
┌─────────────────────────────────────────────────────────────────┐
│                     PLCD Testing Assistant                      │
│                    File Dependency Diagram                      │
└─────────────────────────────────────────────────────────────────┘

CONFIGURATION FILES (Must configure before running)
═══════════════════════════════════════════════════
plcdtestassistant.yaml ◄────────┐
    │                           │
    │  Loaded by all scripts   │
    └──────────────────────┐   │
                           ▼   │
                    config_loader.py
                           │
                           │ Provides:
                           │ - get_azure_client()
                           │ - get_chroma_client()
                           │ - get_embedding_model()
                           │
                           ▼
        ┌──────────────────┴────────────────────┐
        │                                        │
        ▼                                        ▼
   setup_vectordb.py                    plcd_taseq.py
   (ONE-TIME SETUP)                     (MAIN EXECUTION)


DATA FILES (Required inputs)
═══════════════════════════════════════════════════
Selectors_Folder/
└── selectors_merged_runtime_fixed.json
        │
        │ Contains 1340+ selectors with:
        │ - attr, value (CSS selectors)
        │ - module, textContent, context
        │ - priority, isDynamic flags
        │
        └───► Used by: setup_vectordb.py


Jira_Tickets/
└── RBPLCD-XXXX.txt
        │
        │ Contains:
        │ - Title, Module
        │ - Test steps (numbered)
        │ - Expected results
        │
        └───► Used by: plcd_taseq.py → JiraAgent


GENERATED DATA (Created during setup)
═══════════════════════════════════════════════════
data/chromadb_llm/
├── chroma.sqlite3              (Vector embeddings)
└── [UUID folders]              (Metadata & indexes)
        │
        │ Created by: setup_vectordb.py
        │ Contains: 1340+ embedded selectors
        │
        └───► Used by: plcd_taseq.py → SelectorAgent_L1


EXECUTION FILES (Python scripts)
═══════════════════════════════════════════════════

┌────────────────────┐
│  plcd_taseq.py     │  ◄─── MAIN ENTRY POINT
│  (Main Execution)  │
└─────────┬──────────┘
          │
          │ Imports & Uses:
          │
          ├──► config_loader.py
          │       ├─ load_config()
          │       ├─ get_azure_client()
          │       └─ get_chroma_client()
          │
          ├──► agent1_selector_discovery.py
          │       └─ Agent1SelectorDiscovery
          │           └─ discover_selector()
          │
          ├──► report_generator.py
          │       └─ generate_html_report()
          │
          └──► script_generator.py
                  └─ generate_playwright_script()


┌────────────────────┐
│ setup_vectordb.py  │  ◄─── ONE-TIME SETUP SCRIPT
│ (Vector DB Setup)  │
└─────────┬──────────┘
          │
          │ Imports & Uses:
          │
          ├──► config_loader.py
          │       ├─ load_config()
          │       ├─ get_azure_client()
          │       ├─ get_selector_file_path()
          │       ├─ get_chromadb_path()
          │       └─ get_embedding_model()
          │
          ├──► Azure OpenAI API
          │       └─ embeddings.create()
          │
          └──► ChromaDB
                  └─ create_collection()


AGENT FILES (Imported by plcd_taseq.py)
═══════════════════════════════════════════════════

agent1_selector_discovery.py
    │
    │ Class: Agent1SelectorDiscovery
    │
    │ Methods:
    │ - discover_selector(step_text, module, page_context)
    │ - query_selectors() → queries ChromaDB
    │ - generate_embedding() → calls Azure OpenAI
    │
    └───► Dependencies:
            - config_loader.py
            - ChromaDB
            - Azure OpenAI


report_generator.py
    │
    │ Function: generate_html_report()
    │
    │ Generates:
    │ - Reports/RBPLCD-XXXX_TIMESTAMP_report.html
    │
    └───► No external dependencies


script_generator.py
    │
    │ Functions:
    │ - generate_playwright_script()
    │ - generate_pytest_config()
    │ - generate_readme()
    │
    │ Generates:
    │ - Generated_Scripts/RBPLCD-XXXX_TIMESTAMP_test.py
    │ - Generated_Scripts/conftest.py
    │ - Generated_Scripts/README.md
    │
    └───► No external dependencies


OUTPUT FILES (Generated after execution)
═══════════════════════════════════════════════════

Reports/
└── RBPLCD-XXXX_YYYYMMDD_HHMMSS_report.html
        │
        └─ HTML report with:
           - Test results (PASSED/FAILED)
           - Step-by-step execution details
           - Agent chain (L1→L2→L3)
           - Confidence scores
           - Screenshots (embedded)

Generated_Scripts/
└── RBPLCD-XXXX_YYYYMMDD_HHMMSS_test.py
        │
        └─ Executable Playwright script with:
           - All selectors used
           - Login flow
           - Test steps
           - Assertions

Videos/
└── video_YYYYMMDD_HHMMSS.webm
        │
        └─ Screen recording of browser execution

Logs/
├── plcd_taseq.log              (Main execution log)
├── setup_vectordb.log          (Setup log)
├── agent1_selector_discovery.log
└── context_trace_RBPLCD-XXXX_TIMESTAMP.json
        │
        └─ Detailed context tracking:
           - URL history
           - Visible elements per step
           - Agent decisions
```

---

## Execution Flow with Dependencies

### Phase 1: ONE-TIME SETUP

```
USER RUNS:
    python setup_vectordb.py
        │
        ├─► Loads: plcdtestassistant.yaml
        │       └─ config_loader.py
        │
        ├─► Reads: Selectors_Folder/selectors_merged_runtime_fixed.json
        │       └─ 1340 selectors
        │
        ├─► Calls: Azure OpenAI API
        │       └─ Embedding endpoint (text-embedding-3-small)
        │       └─ Generates 1340 embeddings
        │
        └─► Creates: data/chromadb_llm/
                └─ Stores embeddings in ChromaDB
                └─ Creates collection: selectors_base_collection

OUTPUT: "Setup Complete! ChromaDB ready for Agent 1 queries."
```

### Phase 2: TEST EXECUTION

```
USER RUNS:
    python plcd_taseq.py RBPLCD-8835
        │
        ├─► Loads: plcdtestassistant.yaml
        │       └─ config_loader.py
        │
        ├─► Reads: Jira_Tickets/RBPLCD-8835.txt
        │       └─ JiraAgent (LLM parsing)
        │
        ├─► Connects: data/chromadb_llm/
        │       └─ Loads selectors_base_collection
        │
        ├─► Launches: Playwright browser
        │       └─ Microsoft Edge or Chromium
        │
        ├─► Executes Test Steps:
        │   │
        │   For each step:
        │   │
        │   ├─► ContextAgent
        │   │       └─ Captures visible elements, URL, breadcrumb
        │   │
        │   ├─► SelectorAgent_L1
        │   │       ├─ Queries ChromaDB (semantic search)
        │   │       ├─ Calls Azure OpenAI (embedding)
        │   │       └─ Returns selector with confidence
        │   │
        │   ├─► SelectorAgent_L2 (if L1 conf < 0.70)
        │   │       ├─ Scrapes live DOM
        │   │       ├─ Calls Azure OpenAI (chat)
        │   │       └─ Suggests selector
        │   │
        │   ├─► SelectorAgent_L3 (for text verification)
        │   │       ├─ Takes screenshot
        │   │       ├─ Calls Azure OpenAI (vision)
        │   │       └─ Verifies text presence
        │   │
        │   └─► Playwright Action
        │           └─ Executes click/type/verify
        │
        └─► Generates Artifacts:
            │
            ├─► report_generator.py
            │       └─ Reports/RBPLCD-8835_*_report.html
            │
            ├─► script_generator.py
            │       └─ Generated_Scripts/RBPLCD-8835_*_test.py
            │
            └─► Video & Logs
                    ├─ Videos/video_*.webm
                    └─ Logs/plcd_taseq.log

OUTPUT:
    - HTML Report
    - Playwright Script
    - Video Recording
    - Execution Logs
```

---

## Critical File Relationships

### MUST EXIST BEFORE SETUP:
```
plcdtestassistant.yaml              (User must configure)
Selectors_Folder/selectors_merged_runtime_fixed.json  (Provided)
requirements.txt                     (Provided)
config_loader.py                     (Provided)
setup_vectordb.py                    (Provided)
```

### MUST EXIST BEFORE EXECUTION:
```
data/chromadb_llm/                  (Created by setup_vectordb.py)
├── chroma.sqlite3
└── [UUID folders]

Jira_Tickets/RBPLCD-XXXX.txt       (User creates per test)

plcd_taseq.py                       (Provided)
agent1_selector_discovery.py        (Provided)
report_generator.py                 (Provided)
script_generator.py                 (Provided)
```

### CREATED DURING EXECUTION:
```
Reports/RBPLCD-XXXX_*_report.html
Generated_Scripts/RBPLCD-XXXX_*_test.py
Videos/video_*.webm
Logs/plcd_taseq.log
Logs/context_trace_*.json
```

---

## Python Package Dependencies

```
plcd_taseq.py
    ├── langchain, langgraph     (Agent framework)
    ├── openai                   (Azure OpenAI SDK)
    ├── chromadb                 (Vector database)
    ├── playwright.sync_api      (Browser automation)
    ├── json, pathlib, logging   (Standard library)
    └── config_loader, agent1_selector_discovery, etc.

setup_vectordb.py
    ├── chromadb                 (Vector database)
    ├── openai                   (Azure OpenAI SDK)
    ├── pyyaml                   (YAML parsing)
    ├── json, pathlib, logging   (Standard library)
    └── config_loader

config_loader.py
    ├── yaml                     (YAML parsing)
    ├── openai                   (Azure OpenAI client)
    ├── chromadb                 (ChromaDB client)
    └── pathlib                  (File path handling)
```

---

## Dependency Installation Order

```
1. Python 3.10+              (System requirement)
2. pip install -r requirements.txt
   │
   ├─ langchain, langgraph   (Installs with dependencies)
   ├─ openai                 (Azure OpenAI SDK)
   ├─ chromadb               (Vector DB)
   ├─ playwright             (Browser automation)
   ├─ pyyaml                 (Config parsing)
   └─ jinja2, requests, etc. (Utilities)

3. playwright install msedge  (Browser binaries)
```

---

## Configuration Dependency Chain

```
plcdtestassistant.yaml
    │
    ├─► azure_openai section
    │   │
    │   ├─ api_key           → Used by: get_azure_client()
    │   ├─ endpoint          → Used by: get_azure_client()
    │   └─ models            → Used by: all agents
    │
    ├─► vector_database section
    │   │
    │   ├─ persist_directory → Used by: setup_vectordb.py
    │   └─ collections       → Used by: agent1_selector_discovery.py
    │
    ├─► selectors section
    │   │
    │   └─ source_file       → Used by: setup_vectordb.py
    │
    ├─► login section
    │   │
    │   ├─ username          → Used by: plcd_taseq.py (login step)
    │   └─ password          → Used by: plcd_taseq.py (login step)
    │
    └─► web_url
        │
        └─ Application URL   → Used by: plcd_taseq.py (browser navigation)
```

---

## Quick Reference: What Depends on What

| File | Depends On | Imported By | Generates |
|------|------------|-------------|-----------|
| `plcdtestassistant.yaml` | None | All Python files | N/A |
| `config_loader.py` | `plcdtestassistant.yaml` | All Python files | Config objects |
| `setup_vectordb.py` | `config_loader.py`, selectors JSON | (standalone) | `data/chromadb_llm/` |
| `plcd_taseq.py` | All agents, config_loader, ChromaDB | (standalone) | Reports, scripts, videos |
| `agent1_selector_discovery.py` | `config_loader.py`, ChromaDB | `plcd_taseq.py` | Selector results |
| `report_generator.py` | None | `plcd_taseq.py` | HTML reports |
| `script_generator.py` | None | `plcd_taseq.py` | Playwright scripts |

---

**KEY TAKEAWAY:**

1. **Setup Once:** `python setup_vectordb.py` (creates ChromaDB)
2. **Run Many Times:** `python plcd_taseq.py TICKET_ID` (executes tests)
3. **All depends on:** `plcdtestassistant.yaml` (central configuration)
