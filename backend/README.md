# AI-Powered Vision-Based Test Automation

## Overview

**Version:** 1.0 PoC
**Purpose:** Execute functional tests using GPT-4o vision without test selectors, codebase access, or historical tests.

### Key Innovation
- **Vision-only approach**: AI "sees" the screen like a human
- **99%+ accuracy** with 3-retry logic
- **60-90 second** execution time
- **Zero manual intervention** required

### Target Users
Testers can submit a Jira ticket number and receive automated test results with HTML report, video recording, and Playwright script.

---

## Technology Stack

- **Python 3.11+**
- **LangChain + LangGraph** - Multi-agent orchestration
- **Azure OpenAI GPT-4o** - Vision model for element detection
- **Playwright** - Browser automation (Edge support)
- **Pydantic** - State validation
- **Jinja2** - HTML report templating

---

## Prerequisites

1. **Python 3.11 or higher**
2. **Microsoft Edge browser** (installed)
3. **Azure OpenAI API access** with GPT-4o deployment
4. **Windows OS** (current PoC setup)

---

## Installation

### Step 1: Clone or Navigate to Project

```bash
cd C:\Projects\AI_Chat\PLCD\TA_AI_Project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Playwright Browsers

```bash
playwright install msedge
```

---

## Configuration

### 1. Edit `plcdtest_config.yaml`

Open `plcdtest_config.yaml` and configure:

```yaml
# Azure OpenAI Configuration
azure_openai:
  api_key: "YOUR_ACTUAL_API_KEY_HERE"  # ⚠️ REQUIRED: Replace with real API key
  endpoint: "https://ai2ets.openai.azure.com/"
  api_version: "2024-02-15-preview"
  deployment_gpt4o: "gpt-4o"
```

### 2. Verify Other Settings

- **web_url**: Target application URL
- **login credentials**: Username and password
- **wait_times**: Adjust if needed (milliseconds)
- **folder paths**: Relative paths will be auto-resolved

---

## Usage

### Run the Tool

```bash
python main.py
```

### What Happens:

1. **Banner displays**
2. **Prompts for ticket number**: Enter Jira ticket ID (e.g., `RBPLCD-8835`)
3. **Workflow executes**:
   - Loads configuration
   - Parses Jira ticket from `Jira_Tickets/{TICKET-ID}.txt`
   - Launches browser and performs auto-login
   - Executes each test step using GPT-4o vision
   - Records video of execution
   - Generates HTML report and Playwright script
4. **Summary displays** with status and output file paths

### Example

```bash
python main.py

================================================================================
               AI-Powered Vision-Based Test Automation
                              Version 1.0 PoC
================================================================================

Enter Jira ticket number (e.g., RBPLCD-8835): RBPLCD-8835

Starting test automation workflow...

[Execution logs...]

================================================================================
                            EXECUTION SUMMARY
================================================================================

📋 Ticket ID:       RBPLCD-8835
📦 Module:          Teststep
📝 Title:           edit part details

✓ Overall Status:  PASSED
📊 Steps:           8 passed, 0 failed (total: 8)
⏱️  Execution Time:  87.3 seconds

📄 HTML Report:     C:\Projects\...\Reports\RBPLCD-8835_report_20251007_143052.html
🎬 Video:           C:\Projects\...\Videos\video.mp4
📜 Script:          C:\Projects\...\Generated_Scripts\RBPLCD-8835_script_20251007_143052.py

================================================================================
                         🎉 Test Execution Successful!
================================================================================
```

---

## Project Structure

```
TA_AI_Project/
├── plcdtest_config.yaml          # Configuration file
├── main.py                        # Entry point
├── requirements.txt               # Dependencies
├── README.md                      # This file
│
├── agents/                        # Agent modules
│   ├── jira_parser_agent.py      # Parse Jira tickets
│   ├── vision_executor_agent.py  # Execute tests with vision
│   └── report_generator_agent.py # Generate reports
│
├── workflows/                     # LangGraph workflows
│   └── test_workflow.py          # Main workflow orchestration
│
├── models/                        # Data models
│   └── state.py                  # TestAutomationState schema
│
├── utils/                         # Utility modules
│   ├── config_loader.py          # YAML config loader
│   ├── logger.py                 # Logging with sensitive data masking
│   └── vision_helper.py          # Azure OpenAI GPT-4o client
│
├── templates/                     # HTML templates
│   └── report_template.html      # Report template (optional)
│
├── Jira_Tickets/                  # Input: Jira ticket files
│   └── RBPLCD-8835.txt           # Example ticket
│
├── Reports/                       # Output: HTML reports
│   └── screenshots/              # Step screenshots
│
├── Videos/                        # Output: Execution videos
├── Generated_Scripts/             # Output: Playwright scripts
└── Logs/                          # Execution logs
```

---

## Output Files

### 1. HTML Report
- **Location**: `Reports/{TICKET-ID}_report_{TIMESTAMP}.html`
- **Contents**:
  - Executive summary (status, time, date)
  - Test information (ticket details, steps, acceptance criteria)
  - Step results table with embedded screenshots
  - Links to video and script

### 2. Execution Video
- **Location**: `Videos/*.mp4`
- **Format**: 1920x1080, 30fps, H.264
- **Contents**: Full browser automation recording

### 3. Playwright Script
- **Location**: `Generated_Scripts/{TICKET-ID}_script_{TIMESTAMP}.py`
- **Contents**: Executable Python script with coordinate-based clicks
- **Usage**: Can be run independently to reproduce test

### 4. Logs
- **Location**: `Logs/{TICKET-ID}_{TIMESTAMP}.log`
- **Contents**: Detailed DEBUG-level logs with sensitive data masked

---

## Jira Ticket Format

Place Jira ticket files in `Jira_Tickets/` folder.

### Filename Format
`{TICKET-ID}.txt` (e.g., `RBPLCD-8835.txt`)

### File Content Format

```
[RBPLCD-8835] edit part details
Status: Open
Project: RB-PLCD
Component/s: Teststep

Steps to Reproduce:
1. Login
2. navigate to teststep
3. click on teststep named as default_Measurement01
4. open parts accordion
5. click on edit button of part default_testobject_01
6. Click on Type and select "Type 5" from drop down
7. click on save
8. "Successfully edited" message should be displayed

Acceptance Criteria:
"Successfully edited: 'TestObject' default_testobject_01" message should be displayed
```

**Required Fields:**
- Title line with `[TICKET-ID]` format
- `Component/s:` field for module name
- `Steps to Reproduce:` section with numbered steps
- `Acceptance Criteria:` section

---

## Performance Targets (PoC)

| Metric | Target |
|--------|--------|
| **Accuracy** | 99%+ (with retries) |
| **Execution Time** | 60-90 seconds |
| **Cost per Test** | $0.02 |
| **Setup Time** | 0 minutes |
| **Tester Effort** | 10 seconds |

---

## Success Criteria

### Must-Have (PoC)
- ✅ Execute RBPLCD-8835 successfully
- ✅ Execute RBPLCD-8862 successfully
- ✅ Achieve 99%+ accuracy
- ✅ Generate HTML report with screenshots
- ✅ Generate execution video
- ✅ Generate Playwright script
- ✅ Complete in under 90 seconds
- ✅ Zero manual intervention

---

## Troubleshooting

### Issue: "Config file not found"
**Solution**: Ensure `plcdtest_config.yaml` exists in project root.

### Issue: "API key not configured"
**Solution**: Edit `plcdtest_config.yaml` and replace `YOUR_API_KEY_HERE` with actual Azure OpenAI API key.

### Issue: "Jira ticket file not found"
**Solution**: Place ticket file in `Jira_Tickets/` folder with correct naming: `{TICKET-ID}.txt`.

### Issue: "Browser launch failed"
**Solution**:
1. Run `playwright install msedge`
2. Ensure Microsoft Edge is installed

### Issue: "Low vision accuracy"
**Solution**:
- Check screenshot quality
- Retry logic automatically handles this (max 3 attempts)
- Check logs for detailed error messages

### Issue: "Login failed"
**Solution**:
- Verify credentials in `plcdtest_config.yaml`
- Check web application is accessible
- Review screenshots in `Reports/screenshots/`

---

## Out of Scope (PoC)

The following features are NOT included in this PoC:

❌ Multiple simultaneous tests
❌ Test scheduling
❌ Cloud deployment
❌ Multi-user support
❌ Historical pattern learning
❌ Selector-based fallback
❌ Multi-framework support
❌ Multi-browser testing
❌ Mobile testing

---

## Post-PoC Enhancements

### Recommended Next Steps:
1. **Cross-Test Learning**: Learn from successful executions, reuse patterns
   - 2x faster (90s → 45s)
   - 50% cheaper ($0.02 → $0.01)
   - Better accuracy (93% → 97%)

2. **Dashboard UI**: Web interface for test management

3. **Parallel Execution**: Run multiple tests simultaneously

4. **CI/CD Integration**: Jenkins/GitHub Actions integration

---

## Testing the Installation

### 1. Test Phase 1 (Foundation)

```bash
python test_phase1.py
```

Expected: All 6 tests pass (structure, imports, config, logger, state, vision).

### 2. Test Jira Parser

```bash
python test_jira_parser.py
```

Expected: Successfully parses `RBPLCD-8835.txt` and extracts 8 steps.

### 3. Full End-to-End Test

```bash
python main.py
```

Enter: `RBPLCD-8835` when prompted.

Expected: Complete execution with HTML report, video, and script generated.

---

## License

Internal PoC - Bosch Internal Use Only

---

## Support

For issues or questions:
- Check logs in `Logs/` folder
- Review execution screenshots in `Reports/screenshots/`
- Contact development team

---

**Generated**: 2025-10-07
**Version**: 1.0 PoC
**Status**: Ready for Testing
