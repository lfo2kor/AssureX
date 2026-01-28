# PLCD Testing Assistant - Team Handover Checklist

## 📦 What You Received

This folder contains the complete PLCD Testing Assistant v2.0 with multi-agent architecture.

---

## ✅ Pre-Setup Checklist

Before starting, verify you have:

- [ ] **Python 3.10+** installed
  ```bash
  python --version  # Should show 3.10 or higher
  ```

- [ ] **Azure OpenAI credentials**
  - API Key
  - Endpoint URL
  - Deployment names (gpt-4o, text-embedding-3-small)

- [ ] **Access to test application**
  - Application URL
  - Login credentials (username/password)

- [ ] **Network connectivity**
  - Can reach Azure OpenAI endpoint
  - Can reach test application

---

## 🚀 Setup Steps (DO ONCE)

### Step 1: Extract and Navigate
```bash
# Extract TA_AI_Project folder to your desired location
cd C:\Projects\TA_AI_Project  # or your chosen path
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
playwright install msedge
```

**Expected time:** 5-10 minutes

### Step 4: Configure Application

Edit `plcdtestassistant.yaml`:

```yaml
# Section 1: Update Base Folder
base_folder: "C:/Projects/TA_AI_Project"  # YOUR ACTUAL PATH

# Section 2: Update Application URL
web_url: "http://your-app-url/login"

# Section 3: Update Login Credentials
login:
  username: "your_username"
  password: "your_password"

# Section 4: Update Azure OpenAI (MOST IMPORTANT)
azure_openai:
  api_key: "YOUR_AZURE_OPENAI_API_KEY"
  endpoint: "https://your-endpoint.openai.azure.com/"
  api_version: "2024-02-15-preview"

  models:
    chat: "gpt-4o"                      # Your deployment name
    embedding: "text-embedding-3-small" # Your deployment name
    vision: "gpt-4o"                    # Your deployment name
```

**Expected time:** 2-3 minutes

### Step 5: Setup Vector Database
```bash
python setup_vectordb.py
```

**Expected output:**
```
================================================================================
PLCD Testing Assistant - Vector Database Setup
================================================================================

[OK] Configuration loaded: plcdtestassistant.yaml
[OK] Azure OpenAI client initialized
[OK] Selectors loaded: 1340 selectors from JSON
[OK] ChromaDB client initialized: ./data/chromadb_llm

Embedding selectors...
[1/27] Batch 1: Embedding selectors 1-50... [OK] (50 embeddings)
...
[27/27] Batch 27: Embedding selectors 1301-1340... [OK] (40 embeddings)

[OK] Stored 1340 selectors in collection: selectors_base_collection

Statistics:
- Total selectors: 1340
- Time taken: 180.5 seconds

================================================================================
Setup Complete! ChromaDB ready for Agent 1 queries.
================================================================================
```

**Expected time:** 3-5 minutes (depending on API speed)

**✅ SETUP COMPLETE!** You're ready to run tests.

---

## 🎯 Running Tests (DO EVERY TIME)

### Step 1: Prepare Jira Ticket

Create a text file in `Jira_Tickets/` folder:

**File:** `Jira_Tickets/RBPLCD-8835.txt`

**Content format:**
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

### Step 2: Activate Virtual Environment

```bash
cd C:\Projects\TA_AI_Project

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 3: Run Test

```bash
python plcd_taseq.py RBPLCD-8835
```

Replace `RBPLCD-8835` with your actual ticket ID.

### Step 4: Review Results

Results are generated in these folders:

- **HTML Report:** `Reports/RBPLCD-8835_YYYYMMDD_HHMMSS_report.html`
- **Playwright Script:** `Generated_Scripts/RBPLCD-8835_YYYYMMDD_HHMMSS_test.py`
- **Video Recording:** `Videos/video_YYYYMMDD_HHMMSS.webm`
- **Execution Log:** `Logs/plcd_taseq.log`

**Open report:**
```bash
# Windows
start Reports\RBPLCD-8835_*_report.html

# Linux/macOS
xdg-open Reports/RBPLCD-8835_*_report.html
```

---

## 📁 Key Files Reference

### Files You Must Configure:
| File | Purpose | Action Required |
|------|---------|-----------------|
| `plcdtestassistant.yaml` | Main configuration | ✏️ EDIT: Azure keys, app URL, credentials |

### Files You Must Run:
| File | When to Run | Purpose |
|------|-------------|---------|
| `setup_vectordb.py` | **Once** (initial setup) | Creates vector database |
| `plcd_taseq.py TICKET_ID` | **Every test** | Executes test automation |

### Files You Must Create:
| File | Format | Location |
|------|--------|----------|
| Jira ticket files | `.txt` | `Jira_Tickets/RBPLCD-XXXX.txt` |

### Files Already Provided (Don't Touch):
| File | Purpose |
|------|---------|
| `selectors_merged_runtime_fixed.json` | Pre-built selector database (1340 selectors) |
| `config_loader.py` | Configuration utilities |
| `agent1_selector_discovery.py` | Selector discovery agent |
| `report_generator.py` | Report generation |
| `script_generator.py` | Script generation |
| `requirements.txt` | Python dependencies |

---

## 🔍 Verification Checklist

After setup, verify everything is working:

### 1. Test Configuration
```bash
python config_loader.py
```
**Expected:** `[OK] All configuration tests passed!`

### 2. Test Vector Database
```bash
python -c "from config_loader import load_config, get_chroma_client; config = load_config(); client = get_chroma_client(config); coll = client.get_collection('selectors_base_collection'); print(f'Vector DB: {coll.count()} selectors loaded')"
```
**Expected:** `Vector DB: 1340 selectors loaded`

### 3. Test Azure OpenAI Connection
```bash
python -c "from config_loader import load_config, get_azure_client; config = load_config(); client = get_azure_client(config); print('Azure OpenAI: Connected')"
```
**Expected:** `Azure OpenAI: Connected`

### 4. Test Playwright
```bash
python -c "from playwright.sync_api import sync_playwright; playwright = sync_playwright().start(); print('Playwright: OK'); playwright.stop()"
```
**Expected:** `Playwright: OK`

**✅ If all 4 tests pass, you're ready to run tests!**

---

## 📊 Expected Outputs

### During Setup (`setup_vectordb.py`):
- ✅ ChromaDB folder created: `data/chromadb_llm/`
- ✅ Collection created with 1340 selectors
- ✅ Embeddings generated via Azure OpenAI
- ✅ Setup log: `Logs/setup_vectordb.log`

### During Test Execution (`plcd_taseq.py`):
- ✅ Browser window opens (unless headless mode)
- ✅ Test steps execute with real-time console output
- ✅ HTML report generated with screenshots
- ✅ Playwright script generated for reuse
- ✅ Video recording saved (if enabled)
- ✅ Logs saved: `Logs/plcd_taseq.log`

---

## ⚠️ Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Vector DB not initialized** | `Collection 'selectors_base_collection' does not exist` | Run `python setup_vectordb.py` |
| **Azure auth failed** | `openai.AuthenticationError` | Check `api_key` in YAML |
| **Ticket file not found** | `FileNotFoundError: Jira_Tickets/...` | Create ticket file in correct location |
| **Browser not found** | `Executable doesn't exist` | Run `playwright install msedge` |
| **Low selector confidence** | `No selector found (L1: 0.XX)` | Check selector exists in JSON, re-run setup |
| **Rate limit error** | `RateLimitError` | Wait 1 minute, check Azure quota |

---

## 🔄 Re-running Setup

If you need to re-initialize the vector database (e.g., after updating selectors):

```bash
python setup_vectordb.py
```

This will:
- Delete existing ChromaDB collection
- Re-embed all selectors
- Create fresh collection

**Warning:** This takes 3-5 minutes. Only do this if selectors have changed.

---

## 📚 Documentation Files

Your team has access to these guides:

| File | Purpose | Read When |
|------|---------|-----------|
| `QUICK_START.md` | Quick reference for setup & execution | First time setup |
| `SETUP_GUIDE.md` | Complete detailed documentation | Troubleshooting issues |
| `FILE_DEPENDENCIES.md` | File relationships and dependencies | Understanding architecture |
| `TEAM_HANDOVER_CHECKLIST.md` | This file - checklist for team setup | Right now! |

---

## 🎓 Training Steps for New Team Members

### Day 1: Setup
1. [ ] Read `QUICK_START.md`
2. [ ] Complete setup steps (install dependencies)
3. [ ] Configure `plcdtestassistant.yaml`
4. [ ] Run `setup_vectordb.py`
5. [ ] Verify all health checks pass

### Day 2: First Test
1. [ ] Create sample Jira ticket in `Jira_Tickets/`
2. [ ] Run first test: `python plcd_taseq.py TICKET_ID`
3. [ ] Review HTML report
4. [ ] Check generated Playwright script
5. [ ] Watch video recording

### Day 3: Advanced
1. [ ] Read `SETUP_GUIDE.md` for details
2. [ ] Review `FILE_DEPENDENCIES.md` to understand architecture
3. [ ] Experiment with different tickets
4. [ ] Review logs to understand agent decisions
5. [ ] Troubleshoot any issues

---

## 🛠️ Maintenance Tasks

### Weekly:
- [ ] Clean old reports: `rm Reports/*.html` (optional)
- [ ] Review logs for errors: `tail Logs/plcd_taseq.log`

### Monthly:
- [ ] Update selectors JSON if application UI changed
- [ ] Re-run `setup_vectordb.py` to refresh embeddings

### As Needed:
- [ ] Update `plcdtestassistant.yaml` if Azure keys change
- [ ] Update application URL/credentials if changed

---

## 📞 Support Checklist

If something doesn't work, follow this checklist:

1. **Check Logs:**
   ```bash
   cat Logs/plcd_taseq.log
   cat Logs/setup_vectordb.log
   ```

2. **Verify Configuration:**
   ```bash
   python config_loader.py
   ```

3. **Check Vector Database:**
   ```bash
   ls -la data/chromadb_llm/  # Should show files
   ```

4. **Re-run Setup (if needed):**
   ```bash
   python setup_vectordb.py
   ```

5. **Check Azure OpenAI Quota:**
   - Log into Azure Portal
   - Check OpenAI resource usage
   - Verify no rate limits hit

6. **Review Documentation:**
   - `SETUP_GUIDE.md` - Detailed troubleshooting
   - `FILE_DEPENDENCIES.md` - Understanding file relationships

---

## 🎯 Success Criteria

You know the setup is successful when:

- [x] `python setup_vectordb.py` completes with "Setup Complete!"
- [x] `python plcd_taseq.py TICKET_ID` runs without errors
- [x] HTML report is generated in `Reports/` folder
- [x] Playwright script is generated in `Generated_Scripts/` folder
- [x] Test executes in browser (you see it happening)
- [x] All verification commands pass (see Verification Checklist)

---

## 📦 What to Do If You're Stuck

1. **Re-read QUICK_START.md** - Most issues are covered there
2. **Check SETUP_GUIDE.md** - Detailed troubleshooting section
3. **Verify plcdtestassistant.yaml** - 90% of issues are config-related
4. **Check logs** - Error messages are usually clear
5. **Re-run setup** - When in doubt: `python setup_vectordb.py`

---

## 🎉 You're Ready!

If you've completed all setup steps and verifications pass, you're ready to start using the PLCD Testing Assistant!

**Next steps:**
1. Create your first Jira ticket file
2. Run your first test
3. Review the results
4. Start automating!

**Good luck! 🚀**

---

**Questions or Issues?**
- Review documentation files provided
- Check logs in `Logs/` folder
- Verify configuration in `plcdtestassistant.yaml`
