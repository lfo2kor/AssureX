# PLCD Testing Assistant - Quick Start Guide

## 🚀 For First-Time Setup

### Step 1: Install Dependencies (5 minutes)
```bash
cd TA_AI_Project
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
playwright install msedge
```

### Step 2: Configure (2 minutes)

Edit `plcdtestassistant.yaml`:

```yaml
# Update these 3 sections:

1. Azure OpenAI:
   api_key: "YOUR_KEY_HERE"
   endpoint: "https://your-endpoint.openai.azure.com/"

2. Application URL:
   web_url: "http://your-app-url/login"

3. Login Credentials:
   login:
     username: "your_username"
     password: "your_password"
```

### Step 3: Setup Vector Database (3-5 minutes)
```bash
python setup_vectordb.py
```

**Expected:** "Setup Complete! ChromaDB ready for Agent 1 queries."

---

## ✅ For Running Tests (Every Time)

### Quick Commands:
```bash
# 1. Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# 2. Run test
python plcd_taseq.py RBPLCD-8835

# 3. View report
start Reports\RBPLCD-8835_*_report.html  # Windows
xdg-open Reports/RBPLCD-8835_*_report.html  # Linux
```

---

## 📋 Files You Need

### Must Have:
- ✅ `plcdtestassistant.yaml` (configured)
- ✅ `Selectors_Folder/selectors_merged_runtime_fixed.json`
- ✅ `Jira_Tickets/RBPLCD-XXXX.txt` (your ticket)
- ✅ `data/chromadb_llm/` (created by setup_vectordb.py)

### Main Execution Files:
- `plcd_taseq.py` - **Main script to run**
- `config_loader.py` - Configuration loader
- `setup_vectordb.py` - Vector DB setup
- `agent1_selector_discovery.py` - Selector agent
- `report_generator.py` - Report generator
- `script_generator.py` - Script generator

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Collection does not exist" | Run `python setup_vectordb.py` |
| "Ticket file not found" | Check `Jira_Tickets/RBPLCD-XXXX.txt` exists |
| "Browser not found" | Run `playwright install msedge` |
| "Azure OpenAI auth failed" | Check `api_key` in YAML |
| Low selector confidence | Verify selector in JSON, re-run setup |

---

## 📁 Folder Structure

```
TA_AI_Project/
├── plcd_taseq.py              ← RUN THIS
├── plcdtestassistant.yaml     ← CONFIGURE THIS
├── setup_vectordb.py          ← RUN ONCE (setup)
├── requirements.txt
│
├── Jira_Tickets/              ← PUT TICKETS HERE
│   └── RBPLCD-XXXX.txt
│
├── Selectors_Folder/
│   └── selectors_merged_runtime_fixed.json
│
├── data/chromadb_llm/         ← Created by setup
│
└── Reports/                   ← Results appear here
    └── RBPLCD-XXXX_*_report.html
```

---

## 🎯 Execution Sequence

```
1. Setup (ONE TIME):
   python setup_vectordb.py

2. Run Test (EVERY TIME):
   python plcd_taseq.py RBPLCD-XXXX

3. View Results:
   - HTML Report: Reports/
   - Generated Script: Generated_Scripts/
   - Video: Videos/
   - Logs: Logs/
```

---

## 💡 Key Points

- **Vector Database Setup**: Must run BEFORE first test
- **Virtual Environment**: Must activate EVERY TIME before running
- **Configuration**: Update YAML with your Azure keys and app URL
- **Jira Tickets**: Place in `Jira_Tickets/` folder as `.txt` files
- **Reports**: Auto-generated in `Reports/` folder after each run

---

## 📞 Quick Health Check

```bash
# Test configuration
python config_loader.py

# Verify vector database
python -c "from config_loader import load_config, get_chroma_client; config = load_config(); client = get_chroma_client(config); coll = client.get_collection('selectors_base_collection'); print(f'Vector DB: {coll.count()} selectors loaded')"

# Test Azure OpenAI
python -c "from config_loader import load_config, get_azure_client; config = load_config(); client = get_azure_client(config); print('Azure OpenAI: Connected')"
```

**All should print "OK" or success messages.**

---

## 📖 Full Documentation

See `SETUP_GUIDE.md` for detailed documentation.

---

**Questions? Check logs:** `Logs/plcd_taseq.log`
