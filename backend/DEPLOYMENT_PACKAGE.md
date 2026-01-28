# PLCD Test Assistant - Deployment Package

## Files to Share with Your Team

### **1. Core Python Scripts** (Required)
```
plcd_taseq.py                      # Main test execution engine
setup_vectordb.py                   # ChromaDB vector database setup
script_generator.py                 # Playwright test script generator
report_generator.py                 # HTML report generator
```

### **2. Configuration Files** (Required - Update Before Sharing)
```
plcdtestassistant.yaml             # Main configuration file
requirements.txt                    # Python package dependencies
```

**⚠️ IMPORTANT**: Remove or redact Azure OpenAI credentials from `plcdtestassistant.yaml` before sharing!

### **3. Selector Database** (Required)
```
Selectors_Folder/
  └── selectors_merged_runtime_fixed.json    # 1341 UI selectors with embeddings
```

### **4. Input/Output Folders** (Create Empty Folders)
```
Jira_Tickets/                      # Place Jira ticket .txt files here
Logs/                              # Execution logs (auto-created)
Reports/                           # HTML test reports (auto-created)
Videos/                            # Test execution recordings (auto-created)
Generated_Scripts/                 # Playwright .py scripts (auto-created)
```

### **5. ChromaDB Database** (Optional - Can Regenerate)
```
data/
  └── chromadb_llm/                # Vector database (20-30 MB)
      ├── chroma.sqlite3
      └── [multiple UUID folders]
```

**Note**: Team can regenerate this by running `python setup_vectordb.py` (~20 minutes)

### **6. Sample Jira Tickets** (Recommended)
```
Jira_Tickets/RBPLCD-8002.txt       # Working example: View teststep details
Jira_Tickets/RBPLCD-8004.txt       # Working example: Navigate and verify
Jira_Tickets/RBPLCD-8005.txt       # Working example: Copy teststep
```

---

## Setup Instructions for Team Members

### **Step 1: Prerequisites**
- Windows OS
- Python 3.11 or higher
- Microsoft Edge browser
- Azure OpenAI API access

### **Step 2: Extract Files**
Extract all files to: `C:\Projects\PLCD_TestAssistant\`

### **Step 3: Create Virtual Environment**
```bash
cd C:\Projects\PLCD_TestAssistant
python -m venv venv
.\venv\Scripts\activate
```

### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 5: Install Playwright Browsers**
```bash
.\venv\Scripts\playwright.exe install msedge
.\venv\Scripts\playwright.exe install --with-deps
.\venv\Scripts\playwright.exe install ffmpeg
```

### **Step 6: Configure Azure OpenAI**
Edit `plcdtestassistant.yaml`:
```yaml
azure_openai:
  api_key: "YOUR_API_KEY_HERE"
  endpoint: "https://YOUR_ENDPOINT.openai.azure.com/"
  gpt_deployment: "gpt-4o"
  embedding_deployment: "text-embedding-3-small"
  api_version: "2024-02-15-preview"
```

### **Step 7: Configure Application URL**
Edit `plcdtestassistant.yaml`:
```yaml
login:
  url: "http://YOUR_APPLICATION_URL/login"
  username: "YOUR_USERNAME"
  password: "YOUR_PASSWORD"
```

### **Step 8: Setup ChromaDB**
If `data/chromadb_llm/` folder is NOT included:
```bash
python setup_vectordb.py
```
Wait ~20 minutes for completion.

### **Step 9: Run Sample Test**
```bash
python plcd_taseq.py RBPLCD-8002
```

Expected output:
- HTML Report: `Reports/RBPLCD-8002_YYYYMMDD_HHMMSS_report.html`
- Video: `Videos/XXXXX.webm`
- Script: `Generated_Scripts/RBPLCD-8002_YYYYMMDD_HHMMSS_test.py`

---

## Package Size Estimates

| Component | Size | Notes |
|-----------|------|-------|
| Python scripts | < 1 MB | Core engine files |
| Configuration | < 100 KB | YAML + requirements |
| Selector JSON | ~5 MB | 1341 selectors |
| ChromaDB | ~30 MB | Can be regenerated |
| Sample tickets | < 50 KB | Example inputs |
| **Total (without venv)** | **~36 MB** | Compact deployment |
| Virtual environment | ~500 MB | Optional, can be recreated |

---

## Required Python Packages

Create `requirements.txt` if not present:
```
playwright==1.48.0
chromadb==1.3.5
openai==1.57.4
pyyaml==6.0.2
Pillow==11.0.0
```

---

## Folder Structure After Setup

```
PLCD_TestAssistant/
├── plcd_taseq.py
├── setup_vectordb.py
├── script_generator.py
├── report_generator.py
├── plcdtestassistant.yaml
├── requirements.txt
├── DEPLOYMENT_PACKAGE.md
│
├── Selectors_Folder/
│   └── selectors_merged_runtime_fixed.json
│
├── data/
│   └── chromadb_llm/
│       ├── chroma.sqlite3
│       └── [UUID folders...]
│
├── Jira_Tickets/
│   ├── RBPLCD-8002.txt
│   ├── RBPLCD-8004.txt
│   └── RBPLCD-8005.txt
│
├── Logs/                          (auto-created)
├── Reports/                       (auto-created)
├── Videos/                        (auto-created)
├── Generated_Scripts/             (auto-created)
└── venv/                          (created by team)
```

---

## Key Features

### Multi-Agent Architecture
- **L1 Agent**: Semantic search using ChromaDB embeddings (0.7-0.9 confidence)
- **L2 Agent**: Live DOM scraping fallback
- **L3 Agent**: GPT-4 Vision for text verification
- **Learning Agent**: Stores successful selectors for reuse

### Supported Actions
- Navigation (click links, buttons)
- Form interaction (type, select dropdowns)
- Accordion/panel expansion
- Text verification (success messages, displayed text)
- Table row selection
- Button clicks (with menu support)

### Output Artifacts
- HTML reports with step details and confidence scores
- Context traces (JSON) for debugging
- Video recordings (.webm format)
- Generated Playwright scripts (.py)

---

## Troubleshooting

### Issue: "ChromaDB collection empty"
**Solution**: Run `python setup_vectordb.py`

### Issue: "Selector not found"
**Solution**:
1. Check selector exists in `selectors_merged_runtime_fixed.json`
2. Add missing selector with proper `step_text`
3. Rebuild ChromaDB: `python setup_vectordb.py`

### Issue: Login fails
**Solution**: Update credentials in `plcdtestassistant.yaml`

### Issue: Video not recording
**Solution**: Install ffmpeg: `.\venv\Scripts\playwright.exe install ffmpeg`

---

## Security Notes

⚠️ **Before sharing with team:**
1. Remove Azure OpenAI API keys from `plcdtestassistant.yaml`
2. Remove application credentials (username/password)
3. Review logs for sensitive data
4. Consider using environment variables for credentials

---

## Support Contacts

For setup issues or questions:
- Project Lead: [Your Name]
- Email: [Your Email]
- Documentation: This file

---

## Version Info

- **Project**: PLCD Test Assistant
- **Version**: 1.0
- **Python**: 3.11+
- **Playwright**: 1.48.0
- **ChromaDB**: 1.3.5
- **LLM**: Azure OpenAI GPT-4o
- **Embeddings**: text-embedding-3-small (1536 dimensions)
- **Selectors**: 1341 UI elements
- **Last Updated**: 2025-11-21
