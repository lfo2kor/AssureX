# PLCD Testing Assistant - Setup and Execution Guide

**Version:** 2.0
**Date:** November 2025
**Multi-Agent Architecture with Sequential Context Tracking**

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Configuration Files](#configuration-files)
6. [Vector Database Setup](#vector-database-setup)
7. [Running Tests](#running-tests)
8. [Troubleshooting](#troubleshooting)
9. [Dependencies Reference](#dependencies-reference)

---

## Overview

The PLCD Testing Assistant is an AI-powered test automation framework that uses:
- **LangGraph Multi-Agent Architecture** for intelligent test execution
- **Azure OpenAI** for LLM-based decision making
- **ChromaDB** for semantic selector search
- **Playwright** for browser automation

### Key Agents:
- **JiraAgent**: Parses Jira tickets using LLM
- **ContextAgent**: Tracks execution context sequentially
- **LearningAgent**: Learns from past executions
- **SelectorAgent L1**: RAG-based selector discovery
- **SelectorAgent L2**: Live DOM analysis
- **SelectorAgent L3**: Vision-based verification
- **OrchestratorAgent**: Routes between agents

---

## System Requirements

### Software Requirements:
- **Python**: 3.10 or higher
- **Operating System**: Windows 10/11, Linux, or macOS
- **Browser**: Microsoft Edge or Chromium
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 2GB free space

### Network Requirements:
- Access to Azure OpenAI endpoint
- Access to test application URL
- Internet connection for package installation

---

## Project Structure

```
TA_AI_Project/
│
├── plcd_taseq.py                    # MAIN EXECUTION FILE
├── config_loader.py                 # Configuration loader
├── setup_vectordb.py                # Vector database setup script
├── agent1_selector_discovery.py    # Selector discovery agent
├── report_generator.py              # HTML report generator
├── script_generator.py              # Playwright script generator
│
├── plcdtestassistant.yaml           # MAIN CONFIGURATION FILE
├── requirements.txt                 # Python dependencies
│
├── Jira_Tickets/                    # Jira ticket files (.txt)
├── Selectors_Folder/                # Selector JSON files
│   └── selectors_merged_runtime_fixed.json
│
├── data/
│   └── chromadb_llm/                # ChromaDB storage (created during setup)
│
├── Logs/                            # Execution logs
├── Reports/                         # HTML test reports
├── Videos/                          # Recorded test videos
├── Generated_Scripts/               # Generated Playwright scripts
└── Screenshots/                     # Test screenshots
```

---

## Setup Instructions

### Step 1: Extract Project Folder

Extract the `TA_AI_Project` folder to your desired location:

```
Example: C:\Projects\TA_AI_Project
         /home/user/projects/TA_AI_Project
```

### Step 2: Create Python Virtual Environment

Open a terminal/command prompt in the project folder:

**Windows:**
```bash
cd C:\Projects\TA_AI_Project
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
cd /path/to/TA_AI_Project
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed langchain-X.X.X langgraph-X.X.X openai-X.X.X chromadb-X.X.X playwright-X.X.X ...
```

### Step 4: Install Playwright Browsers

```bash
playwright install msedge
playwright install chromium
```

**Expected output:**
```
Downloading Microsoft Edge...
✓ Browser installed successfully
```

### Step 5: Verify Installation

```bash
python -c "import langchain, chromadb, playwright; print('All packages installed successfully!')"
```

**Expected output:**
```
All packages installed successfully!
```

---

## Configuration Files

### 1. Main Configuration: `plcdtestassistant.yaml`

**Key sections to update:**

#### A. Base Configuration
```yaml
base_folder: "C:/Projects/AI_Chat/PLCD/TA_AI_Project"  # UPDATE THIS PATH
web_url: "http://your-app-url/login"                   # UPDATE APPLICATION URL
browser: "edge"                                        # edge or chromium
```

#### B. Login Credentials
```yaml
login:
  username: "your_username"  # UPDATE
  password: "your_password"  # UPDATE
```

#### C. Azure OpenAI Configuration
```yaml
azure_openai:
  api_key: "YOUR_AZURE_OPENAI_API_KEY"           # UPDATE
  endpoint: "https://your-endpoint.openai.azure.com/"  # UPDATE
  api_version: "2024-02-15-preview"

  models:
    chat: "gpt-4o"                          # Your deployment name
    embedding: "text-embedding-3-small"     # Your deployment name
    vision: "gpt-4o"                        # Your deployment name
```

#### D. Selectors Source File
```yaml
selectors:
  source_file: "Selectors_Folder/selectors_merged_runtime_fixed.json"
  total_count: 1340  # Update based on your selector count
```

### 2. Selectors JSON File

**Location:** `Selectors_Folder/selectors_merged_runtime_fixed.json`

**Format:**
```json
{
  "selectors": [
    {
      "id": "selector_0001",
      "attr": "data-test",
      "value": "sidebar-nav-item-nav_item_teststeps",
      "module": "Teststep",
      "textContent": "Runs",
      "context": ["sidebar", "navigation"],
      "pageUrl": "http://example.com/dashboard",
      "priority": 100,
      "isDynamic": false
    }
  ]
}
```

**This file should already be included in your project folder.**

### 3. Jira Ticket Files

**Location:** `Jira_Tickets/`

**Format:** Plain text files named `RBPLCD-XXXX.txt`

**Example:** `Jira_Tickets/RBPLCD-8835.txt`
```
Title: Create new teststep
Module: Teststep

Steps:
1. Login
2. Navigate to Teststep from sidebar
3. Click Create New button
4. Enter name "test_measurement_01"
5. Click Save
6. Verify success message "Successfully created"
```

---

## Vector Database Setup

### CRITICAL: Must run BEFORE first execution

The vector database (ChromaDB) stores embedded selectors for semantic search. This must be set up once before running any tests.

### Step 1: Verify Configuration

Ensure `plcdtestassistant.yaml` is configured (see Configuration Files section).

### Step 2: Run Vector Database Setup Script

```bash
python setup_vectordb.py
```

### Expected Output:

```
================================================================================
PLCD Testing Assistant - Vector Database Setup
================================================================================

[OK] Configuration loaded: plcdtestassistant.yaml
[OK] Azure OpenAI client initialized
[OK] Selectors loaded: 1340 selectors from JSON
[OK] ChromaDB client initialized: ./data/chromadb_llm
[OK] Collection: selectors_base_collection

Embedding selectors...
[1/27] Batch 1: Embedding selectors 1-50... [OK] (50 embeddings)
[2/27] Batch 2: Embedding selectors 51-100... [OK] (50 embeddings)
...
[27/27] Batch 27: Embedding selectors 1301-1340... [OK] (40 embeddings)

Storing in ChromaDB...
[OK] Stored 1340 selectors in collection: selectors_base_collection

Verification...
[OK] Collection count: 1340 selectors
[OK] Test query successful

Statistics:
- Total selectors: 1340
- Embedding dimension: 1536
- ChromaDB location: ./data/chromadb_llm
- Time taken: 180.5 seconds

Top modules by selector count:
  1. Teststep: 245 selectors (18.3%)
  2. Common: 189 selectors (14.1%)
  3. EntityAttribute: 156 selectors (11.6%)
  4. CreateNew: 123 selectors (9.2%)
  5. Tests: 98 selectors (7.3%)

================================================================================
Setup Complete! ChromaDB ready for Agent 1 queries.
================================================================================

[OK] Vector database setup completed successfully!
```

### What This Does:

1. **Loads selectors** from JSON file
2. **Converts each selector** to natural language using LLM
3. **Generates embeddings** using Azure OpenAI (text-embedding-3-small)
4. **Stores in ChromaDB** with metadata for filtering
5. **Verifies** collection was created successfully

### Troubleshooting Setup:

#### Error: "Configuration file not found"
```bash
# Ensure you're in the project root directory
cd C:\Projects\TA_AI_Project
ls plcdtestassistant.yaml  # Should show the file
```

#### Error: "Azure OpenAI authentication failed"
- Check `api_key` in `plcdtestassistant.yaml`
- Verify `endpoint` URL is correct
- Test connection:
  ```bash
  python config_loader.py
  ```

#### Error: "Selectors file not found"
- Verify file exists: `Selectors_Folder/selectors_merged_runtime_fixed.json`
- Check `source_file` path in YAML is correct

#### Error: "Embedding failed"
- Check Azure OpenAI quota/rate limits
- Verify embedding model deployment name matches YAML config
- Retry with smaller batch size (edit `batch_size` in YAML)

---

## Running Tests

### Execution Sequence

Once setup is complete, follow this sequence to run tests:

### Step 1: Prepare Jira Ticket

Create or ensure Jira ticket file exists:

**File:** `Jira_Tickets/RBPLCD-8835.txt`

### Step 2: Activate Virtual Environment

**Windows:**
```bash
cd C:\Projects\TA_AI_Project
venv\Scripts\activate
```

**Linux/macOS:**
```bash
cd /path/to/TA_AI_Project
source venv/bin/activate
```

### Step 3: Run Test Execution

```bash
python plcd_taseq.py RBPLCD-8835
```

Replace `RBPLCD-8835` with your ticket ID.

### Expected Output:

```
================================================================================
PLCD Testing Assistant (Sequential) - Executing: RBPLCD-8835
================================================================================

[1/6] Parsing Jira ticket with LLM...
[OK] Ticket: Create new teststep
[OK] Module: Teststep
[OK] Steps: 6

[2/6] Initializing browser...
[OK] Browser: edge

[3/6] Capturing initial context...
[OK] Initial context captured

[4/6] Logging in...
[OK] Logged in as: mechanic

[5/6] Executing test steps with agents...
--------------------------------------------------------------------------------

Step 1/6: Login
  [SKIPPED] Already logged in

Step 2/6: Navigate to Teststep from sidebar
  [L1] [data-test='sidebar-nav-item-nav_item_teststeps'] (conf: 0.92)
[OK] [data-test='sidebar-nav-item-nav_item_teststeps'] (agent: L1)

Step 3/6: Click Create New button
  [L1] [data-test='create-new-btn'] (conf: 0.88)
[OK] [data-test='create-new-btn'] (agent: L1)

Step 4/6: Enter name "test_measurement_01"
  [L1] [data-test='name-input-field'] (conf: 0.85)
[OK] [data-test='name-input-field'] (agent: L1)

Step 5/6: Click Save
  [L1] [data-test='btn-save'] (conf: 0.91)
[OK] [data-test='btn-save'] (agent: L1)

Step 6/6: Verify success message "Successfully created"
  [L3 Vision] Verifying text: 'Successfully created'
  [L3] Text found: Successfully created TestObject: test_measurement_01
       Location: green banner
       Confidence: 0.95
[OK] Text verified (agent: L3)

--------------------------------------------------------------------------------

[6/6] Generating artifacts...
[OK] HTML Report: Reports/RBPLCD-8835_20251128_143025_report.html
[OK] Playwright Script: Generated_Scripts/RBPLCD-8835_20251128_143025_test.py
[OK] Context Trace: Logs/context_trace_RBPLCD-8835_20251128_143025.json

[OK] Summary:
     Status: PASSED
     Total Steps: 6
     Passed: 6
     Failed: 0
     Execution Time: 45.2s
     Agent Chain: JiraAgent -> ContextAgent -> SelectorAgent_L1 -> SelectorAgent_L3

================================================================================
Test Execution Complete!
================================================================================
```

### Step 4: Review Results

#### A. HTML Report
```bash
# Open in browser (Windows)
start Reports/RBPLCD-8835_20251128_143025_report.html

# Open in browser (Linux/macOS)
xdg-open Reports/RBPLCD-8835_20251128_143025_report.html
```

#### B. Generated Playwright Script
```bash
# View generated script
cat Generated_Scripts/RBPLCD-8835_20251128_143025_test.py

# Run generated script independently
pytest Generated_Scripts/RBPLCD-8835_20251128_143025_test.py
```

#### C. Video Recording
```bash
# Videos are saved automatically if enabled
ls Videos/
```

#### D. Logs
```bash
# View execution logs
tail -f Logs/plcd_taseq.log
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    START: python plcd_taseq.py TICKET_ID    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  1. JiraAgent: Parse Jira Ticket (LLM-based)                │
│     - Load ticket file from Jira_Tickets/                   │
│     - Extract steps, module, expected results               │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Initialize Browser (Playwright)                         │
│     - Launch Edge/Chromium                                  │
│     - Navigate to web_url                                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ContextAgent: Capture Initial Context                   │
│     - Extract visible data-* attributes                     │
│     - Record URL, breadcrumb, page title                    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Login to Application                                    │
│     - Fill username & password                              │
│     - Click login button                                    │
│     - Wait for dashboard                                    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Execute Test Steps (Multi-Agent Loop)                   │
│                                                              │
│  For each step:                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │ A. ContextAgent: Capture current page context  │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ B. Text Verification Step?                     │        │
│  │    YES → SelectorAgent_L3 (Vision)             │        │
│  │    NO  → Continue to selector discovery        │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ C. LearningAgent: Check learned selectors      │        │
│  │    (Currently disabled to avoid confusion)     │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ D. SelectorAgent_L1: RAG + LLM Validation      │        │
│  │    - Query ChromaDB with step text             │        │
│  │    - Boost by name matching, action keywords   │        │
│  │    - Validate against visible DOM              │        │
│  │    - Confidence >= 0.70? → Execute             │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ E. Confidence < 0.70?                          │        │
│  │    → SelectorAgent_L2: DOM + LLM Analysis      │        │
│  │      - Scrape live DOM elements                │        │
│  │      - LLM analyzes and suggests selector      │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ F. Execute Action (click/type/verify)          │        │
│  │    - Playwright performs action                │        │
│  │    - Wait for page response                    │        │
│  │    - Capture screenshot                        │        │
│  └────────────────────────────────────────────────┘        │
│                     │                                       │
│                     ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ G. Store successful selector to Learning DB    │        │
│  └────────────────────────────────────────────────┘        │
│                                                              │
│  Repeat for next step...                                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Generate Artifacts                                      │
│     - HTML Report (with screenshots)                        │
│     - Playwright Script (reusable)                          │
│     - Context Trace (debug)                                 │
│     - Video Recording (webm)                                │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    END: Test Complete                       │
│              Exit Code: 0 (PASSED) or 1 (FAILED)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

#### 1. "Vector database not initialized"

**Error:**
```
[ERROR] Collection 'selectors_base_collection' does not exist
```

**Solution:**
```bash
python setup_vectordb.py
```

#### 2. "Jira ticket file not found"

**Error:**
```
FileNotFoundError: Ticket file not found: Jira_Tickets/RBPLCD-8835.txt
```

**Solution:**
- Verify file exists in `Jira_Tickets/` folder
- Check filename matches: `RBPLCD-XXXX.txt`
- Ensure ticket ID is correct

#### 3. "Playwright browser not installed"

**Error:**
```
playwright._impl._api_types.Error: Executable doesn't exist
```

**Solution:**
```bash
playwright install msedge
playwright install chromium
```

#### 4. "Azure OpenAI API rate limit"

**Error:**
```
openai.RateLimitError: Rate limit exceeded
```

**Solution:**
- Wait 1 minute and retry
- Check Azure OpenAI quota in Azure Portal
- Reduce `batch_size` in `plcdtestassistant.yaml`

#### 5. "Selector not found / Low confidence"

**Behavior:**
- L1 returns confidence < 0.70
- L2 agent activated but still fails

**Solution:**
1. Check if selector exists in JSON:
   ```bash
   grep "sidebar-nav-item" Selectors_Folder/selectors_merged_runtime_fixed.json
   ```
2. Verify selector is visible on page
3. Add missing selector to JSON manually
4. Re-run vector database setup:
   ```bash
   python setup_vectordb.py
   ```

#### 6. "Login failed"

**Error:**
```
[ERROR] Login failed: Timeout waiting for selector
```

**Solution:**
- Verify `web_url` in YAML is correct
- Check `username` and `password` are correct
- Ensure application is accessible from your network

#### 7. "Context capture failed"

**Error:**
```
ContextAgent: Failed to capture context: ...
```

**Solution:**
- Check page loaded completely
- Verify JavaScript is enabled
- Increase `page_load_timeout` in YAML

---

## Dependencies Reference

### Core Python Packages

```
langchain>=0.1.0              # Agent orchestration framework
langgraph>=0.0.40             # Graph-based agent workflows
openai>=1.12.0                # Azure OpenAI SDK
chromadb>=0.4.22              # Vector database
playwright>=1.40.0            # Browser automation
pyyaml>=6.0                   # YAML configuration
requests>=2.31.0              # HTTP requests
jinja2>=3.1.2                 # Template engine
pytest>=7.4.0                 # Testing framework
```

### Installation Verification

```bash
# Check all packages
pip list | grep -E "langchain|chromadb|openai|playwright"

# Expected output:
chromadb                 0.4.22
langchain                0.1.0
langgraph                0.0.40
openai                   1.12.0
playwright               1.40.0
```

---

## Quick Reference Commands

### Setup Phase (ONE TIME)
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install browsers
playwright install msedge

# 4. Setup vector database
python setup_vectordb.py
```

### Execution Phase (EVERY TEST)
```bash
# 1. Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Run test
python plcd_taseq.py RBPLCD-XXXX

# 3. View report
start Reports/RBPLCD-XXXX_*_report.html  # Windows
# or
xdg-open Reports/RBPLCD-XXXX_*_report.html  # Linux
```

### Maintenance Commands
```bash
# Re-initialize vector database
python setup_vectordb.py

# Test configuration
python config_loader.py

# View logs
tail -f Logs/plcd_taseq.log

# Clean old artifacts (optional)
rm -rf Videos/*.webm
rm -rf Reports/*.html
```

---

## File Dependencies Checklist

Before running tests, ensure these files exist:

- [ ] `plcdtestassistant.yaml` (configured with your settings)
- [ ] `Selectors_Folder/selectors_merged_runtime_fixed.json`
- [ ] `Jira_Tickets/RBPLCD-XXXX.txt` (your ticket file)
- [ ] `data/chromadb_llm/` (created by setup_vectordb.py)
- [ ] `requirements.txt`

Required Python files:
- [ ] `plcd_taseq.py`
- [ ] `config_loader.py`
- [ ] `setup_vectordb.py`
- [ ] `agent1_selector_discovery.py`
- [ ] `report_generator.py`
- [ ] `script_generator.py`

---

## Support and Troubleshooting

If you encounter issues not covered in this guide:

1. **Check Logs:**
   ```bash
   cat Logs/plcd_taseq.log
   cat Logs/setup_vectordb.log
   ```

2. **Verify Configuration:**
   ```bash
   python config_loader.py
   ```

3. **Test Azure OpenAI Connection:**
   ```bash
   python -c "from config_loader import load_config, get_azure_client; config = load_config(); client = get_azure_client(config); print('Connection OK')"
   ```

4. **Check Vector Database:**
   ```bash
   python -c "from config_loader import load_config, get_chroma_client; config = load_config(); client = get_chroma_client(config); coll = client.get_collection('selectors_base_collection'); print(f'Selectors: {coll.count()}')"
   ```

---

## Version History

- **v2.0** (Nov 2025): Multi-agent architecture with sequential context tracking
- **v1.0** (Oct 2025): Initial release with single-agent RAG

---

**END OF SETUP GUIDE**
