# PLCD Testing Assistant - Folder Structure

**Version:** 2.0  
**Date:** 2025-11-17  
**Purpose:** Complete directory layout and file organization

---

## Complete Project Structure

```
C:\Projects\AI_Chat\PLCD\TA_AI_Project\
│
├── plcdtestassistant.yaml              # Main configuration file
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
├── .env                                # Environment variables (optional)
├── .gitignore                          # Git ignore file
│
├── config_loader.py                    # YAML configuration loader
├── setup_vectordb.py                   # Phase 1: Vector database setup
├── test_setup.py                       # Phase 1: Validation script
├── plcd_ta.py                          # Main orchestrator (entry point)
│
├── agents/                             # Agent implementations
│   ├── __init__.py
│   ├── agent1_selector_discovery.py    # Agent 1 (L1): Semantic search
│   ├── agent2_dom_discovery.py         # Agent 2 (L2): DOM analysis
│   └── agent3_vision.py                # Agent 3 (L3): Vision fallback
│
├── utils/                              # Utility modules
│   ├── __init__.py
│   ├── embedding_utils.py              # Azure OpenAI embedding functions
│   ├── context_tracker.py              # Module context tracking
│   ├── report_generator.py             # HTML report generation
│   ├── script_generator.py             # Playwright script generation
│   └── db_manager.py                   # SQLite database operations
│
├── data/                               # Runtime data storage
│   ├── chromadb/                       # ChromaDB persistence
│   │   ├── chroma.sqlite3              # ChromaDB SQLite database
│   │   └── collections/                # Collection data
│   │       ├── selectors_base_collection/
│   │       └── runtime_learned_collection/
│   │
│   ├── project_config.db               # SQLite: Project configurations
│   ├── agent_memory.json               # Agent learning memory
│   └── feedback_learning.json          # Feedback loop data
│
├── Selectors_Folder/                   # Selector library
│   └── selectors_merged_runtime_fixed.json  # 1,340 selectors
│
├── Jira_Tickets/                       # Jira ticket files
│   ├── RBPLCD-8835.txt                 # Example ticket
│   ├── RBPLCD-8836.txt
│   └── ...
│
├── Logs/                               # Execution logs
│   ├── testing_assistant.log           # Main execution log
│   ├── agent_execution.log             # Agent decision log
│   ├── setup_vectordb.log              # Setup log
│   └── errors.log                      # Error-only log
│
├── Reports/                            # Generated HTML reports
│   ├── RBPLCD-8835_20251117_102345_report.html
│   ├── RBPLCD-8836_20251117_143022_report.html
│   └── ...
│
├── Videos/                             # Screen recordings
│   ├── RBPLCD-8835_20251117_102345.webm
│   └── ...
│
├── Screenshots/                        # Step screenshots
│   ├── step_1_success.png
│   ├── step_2_success.png
│   ├── step_3_vision.png
│   └── ...
│
├── Generated_Scripts/                  # Output Playwright scripts
│   ├── RBPLCD-8835_Teststep_20251117_102345.py
│   ├── RBPLCD-8836_Equipment_20251117_143022.py
│   └── ...
│
├── templates/                          # Jinja2 templates
│   ├── report_template.html            # HTML report template
│   └── script_template.py              # Playwright script template
│
└── tests/                              # Unit and integration tests
    ├── __init__.py
    ├── test_config_loader.py
    ├── test_agent1.py
    ├── test_agent2.py
    ├── test_context_tracker.py
    ├── test_embedding_utils.py
    └── test_integration.py
```

---

## Detailed File Descriptions

### Root Level Files

| File | Purpose | Lines of Code | Phase |
|------|---------|---------------|-------|
| `plcdtestassistant.yaml` | Main configuration with all settings | ~370 | All |
| `requirements.txt` | Python dependencies | ~80 | All |
| `config_loader.py` | Load and validate YAML config | ~100-150 | Phase 1 |
| `setup_vectordb.py` | Embed selectors into ChromaDB | ~250-300 | Phase 1 |
| `test_setup.py` | Validate Phase 1 setup | ~100-150 | Phase 1 |
| `plcd_ta.py` | Main orchestrator with LangGraph | ~500-600 | Phase 2-3 |

---

### `/agents/` Directory

**Purpose:** Agent implementations for L1, L2, L3

| File | Purpose | Lines of Code | Dependencies |
|------|---------|---------------|--------------|
| `agent1_selector_discovery.py` | Semantic search in ChromaDB | ~200-250 | ChromaDB, Azure OpenAI |
| `agent2_dom_discovery.py` | Live DOM extraction and matching | ~200-250 | Playwright, ChromaDB |
| `agent3_vision.py` | GPT-4o vision analysis | ~150-200 | Azure OpenAI Vision |

**Key Classes:**
- `SelectorDiscoveryAgent`: Implements L1 semantic search
- `DOMDiscoveryAgent`: Implements L2 DOM analysis
- `VisionAgent`: Implements L3 vision fallback

---

### `/utils/` Directory

**Purpose:** Shared utility modules

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `embedding_utils.py` | Azure OpenAI embedding functions | ~50-80 |
| `context_tracker.py` | Track module context across steps | ~100-150 |
| `report_generator.py` | Generate HTML reports from results | ~150-200 |
| `script_generator.py` | Generate Playwright scripts | ~150-200 |
| `db_manager.py` | SQLite database operations | ~200-250 |

**Key Functions:**
- `generate_embedding()`: Single text embedding
- `batch_generate_embeddings()`: Batch embedding for efficiency
- `cosine_similarity()`: Calculate similarity scores
- `build_stable_selector()`: Generate stable CSS/XPath selectors

---

### `/data/` Directory

**Purpose:** Runtime data storage and persistence

#### `/data/chromadb/`
- **ChromaDB persistence** for vector database
- Contains SQLite database and collection data
- Auto-created by ChromaDB during setup

#### `project_config.db` (SQLite)
Tables:
- `projects`: Project configurations
- `login_credentials`: Encrypted login info
- `agent_memory`: Historical step execution data
- `feedback_learning`: User feedback and corrections
- `execution_history`: Ticket execution summaries

#### `agent_memory.json` (JSON)
```json
{
  "step_memories": [
    {
      "step_text": "click save button",
      "selector_used": "[data-SaveBtn='AddBtn']",
      "confidence": 0.87,
      "agent_used": "L1",
      "module": "Teststep",
      "success": true,
      "timestamp": "2025-11-17T10:23:45"
    }
  ]
}
```

#### `feedback_learning.json` (JSON)
```json
{
  "corrections": [
    {
      "step_text": "edit part name",
      "wrong_selector": "[data-Edit='Btn']",
      "correct_selector": "[data-EditBtn='EditPartBtn']",
      "feedback_type": "manual_correction",
      "timestamp": "2025-11-17T10:25:12"
    }
  ]
}
```

---

### `/Selectors_Folder/` Directory

**Purpose:** Selector library storage

#### `selectors_merged_runtime_fixed.json`
- **1,340 selectors** across 30 modules
- Source of truth for base selector collection
- Format:
```json
{
  "metadata": {"total_count": 1340},
  "selectors": [
    {
      "id": "selector_0001",
      "attr": "data-SaveBtn",
      "value": "AddBtn",
      "module": "AddExisting",
      "elementType": "button",
      "label": "Save button for adding items",
      "context": ["save", "button", "add"],
      "priority": 25,
      "isDynamic": false,
      "runtimeVerified": false
    }
  ]
}
```

---

### `/Jira_Tickets/` Directory

**Purpose:** Store Jira ticket descriptions

**File Format:** `.txt` or `.docx`

**Example:** `RBPLCD-8835.txt`
```
Title: Edit Part Name in Runs Module

Steps:
Step 1: Navigate to Runs module
Step 2: Select first run from table
Step 3: Click edit button
Step 4: Update part name to "Test Part XYZ"
Step 5: Select status "Completed"
Step 6: Click save button
Step 7: Verify success message
Step 8: Verify part updated in table
```

---

### `/Logs/` Directory

**Purpose:** Store execution logs

| File | Content | Rotation |
|------|---------|----------|
| `testing_assistant.log` | Main execution flow | Daily |
| `agent_execution.log` | Agent decisions and confidence scores | Daily |
| `setup_vectordb.log` | Setup and embedding logs | Once |
| `errors.log` | Error-only logs | Daily |

**Log Format:**
```
2025-11-17 10:23:45 - Agent1 - INFO - Query: "click edit button"
2025-11-17 10:23:45 - Agent1 - INFO - Module filter: ["Teststep", "Common"]
2025-11-17 10:23:45 - Agent1 - INFO - Top candidate: selector_0057 (conf: 0.87)
2025-11-17 10:23:45 - Executor - INFO - Action: click, Selector: [data-EditBtn='EditPartBtn']
2025-11-17 10:23:46 - Executor - INFO - Step 5 PASSED
```

---

### `/Reports/` Directory

**Purpose:** Generated HTML reports

**Naming Convention:** `{ticket_id}_{timestamp}_report.html`

**Content:**
- Execution summary (passed/failed steps)
- Step-by-step details with screenshots
- Confidence scores and agent used
- Links to generated script and video

---

### `/Videos/` Directory

**Purpose:** Screen recordings of test executions

**Format:** `.webm` (Playwright default)

**Naming Convention:** `{ticket_id}_{timestamp}.webm`

**Configuration:**
```yaml
execution:
  record_video: true
```

---

### `/Screenshots/` Directory

**Purpose:** Step-by-step screenshots

**Naming Convention:**
- Success: `step_{step_number}_success.png`
- Vision: `step_{step_number}_vision.png`
- Failure: `step_{step_number}_failed.png`

**Configuration:**
```yaml
execution:
  screenshot_on_every_step: true
```

---

### `/Generated_Scripts/` Directory

**Purpose:** Output Playwright test scripts

**Naming Convention:** `{ticket_id}_{module}_{timestamp}.py`

**Organization:**
```
Generated_Scripts/
├── Teststep/
│   ├── RBPLCD-8835_Teststep_20251117_102345.py
│   └── RBPLCD-8839_Teststep_20251117_145022.py
├── Equipment/
│   └── RBPLCD-8836_Equipment_20251117_110134.py
└── ...
```

---

### `/templates/` Directory

**Purpose:** Jinja2 templates for generation

#### `report_template.html`
- HTML structure for reports
- Includes CSS styling
- Dynamic content placeholders

#### `script_template.py`
- Playwright script structure
- Pytest format
- Configurable steps

---

### `/tests/` Directory

**Purpose:** Unit and integration tests

| File | Purpose |
|------|---------|
| `test_config_loader.py` | Test YAML loading and validation |
| `test_agent1.py` | Test Agent 1 semantic search |
| `test_agent2.py` | Test Agent 2 DOM discovery |
| `test_context_tracker.py` | Test module tracking |
| `test_embedding_utils.py` | Test embedding generation |
| `test_integration.py` | End-to-end workflow test |

---

## File Size Estimates

| Directory | Estimated Size |
|-----------|---------------|
| `/data/chromadb/` | ~50-100 MB (1,340 embeddings) |
| `/Selectors_Folder/` | ~500 KB (JSON file) |
| `/Jira_Tickets/` | ~10-50 KB per ticket |
| `/Logs/` | ~1-5 MB per day |
| `/Reports/` | ~100-500 KB per report |
| `/Videos/` | ~5-20 MB per video |
| `/Screenshots/` | ~50-200 KB per screenshot |
| `/Generated_Scripts/` | ~5-15 KB per script |

**Total Project Size:** ~200-500 MB (with logs and videos)

---

## File Creation Order (Development Sequence)

### Phase 1: Foundation (Week 1)
1. `requirements.txt`
2. `plcdtestassistant.yaml`
3. `config_loader.py`
4. `utils/embedding_utils.py`
5. `setup_vectordb.py`
6. `test_setup.py`

### Phase 2: Agent 1 (Week 2)
7. `agents/__init__.py`
8. `agents/agent1_selector_discovery.py`
9. `utils/context_tracker.py`
10. `plcd_ta.py` (basic version)

### Phase 3: Complete System (Week 3-4)
11. `agents/agent2_dom_discovery.py`
12. `agents/agent3_vision.py`
13. `utils/report_generator.py`
14. `utils/script_generator.py`
15. `utils/db_manager.py`
16. `plcd_ta.py` (complete version)
17. Test files

---

## Folder Permissions & Access

### Read-Only Folders
- `/Selectors_Folder/` - Source selector library (don't modify)
- `/Jira_Tickets/` - Input tickets (read-only during execution)

### Read-Write Folders
- `/data/` - ChromaDB and memory files
- `/Logs/` - Log files
- `/Reports/` - Generated reports
- `/Videos/` - Recorded videos
- `/Screenshots/` - Step screenshots
- `/Generated_Scripts/` - Output scripts

### Auto-Created Folders
The following folders are automatically created on first run:
- `/data/chromadb/`
- `/data/chromadb/collections/`
- `/Logs/`
- `/Reports/`
- `/Videos/`
- `/Screenshots/`
- `/Generated_Scripts/`

---

## .gitignore Configuration

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data & Logs
/data/chromadb/
/data/*.db
/data/*.json
/Logs/*.log

# Generated Outputs
/Reports/*.html
/Videos/*.webm
/Screenshots/*.png
/Generated_Scripts/*.py

# Sensitive
.env
plcdtestassistant.yaml  # Contains API keys

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## Backup Strategy

**What to Backup:**
1. `/Selectors_Folder/` - Selector library
2. `/Jira_Tickets/` - Test tickets
3. `plcdtestassistant.yaml` - Configuration
4. `/data/project_config.db` - Project settings
5. Source code files (`.py`)

**What NOT to Backup:**
- `/data/chromadb/` - Can be regenerated
- `/Logs/` - Temporary logs
- `/Videos/` - Large, can be regenerated
- `/Screenshots/` - Temporary
- `/Generated_Scripts/` - Can be regenerated

---

## Environment Variables (.env)

Optional `.env` file for sensitive data:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://ai2ets.openai.azure.com/

# Application
APP_USERNAME=mechanic
APP_PASSWORD=avalon

# Paths
BASE_FOLDER=C:/Projects/AI_Chat/PLCD/TA_AI_Project
```

---

**END OF FOLDER STRUCTURE DOCUMENTATION**

This document provides complete directory layout and file organization for the PLCD Testing Assistant project.
