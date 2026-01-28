# How to Use the Streamlit Admin Portal

## Starting the Application

### Option 1: Using the Batch File (Windows)
Double-click `START_STREAMLIT.bat`

### Option 2: Command Line
```bash
streamlit run web_app.py
```

### Option 3: Using Virtual Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Run streamlit
streamlit run web_app.py
```

## Step-by-Step Guide

### Step 1: Access the Application

After starting, your browser will automatically open to:
```
http://localhost:8501
```

You'll see the admin portal homepage with the title:
**📤 Upload Project Configuration**

### Step 2: Upload Configuration File

1. Click the **"Browse files"** button or drag & drop a YAML file
2. You can use `sample_config.yaml` for testing
3. Only `.yaml` and `.yml` files are accepted

### Step 3: Review Configuration Preview

Once uploaded, you'll see a preview showing:
- **Project Name** (e.g., "sample_test_project")
- **Base Folder** (e.g., "Projects")
- **Web URL** (e.g., "https://www.google.com")
- **Number of Testers** (e.g., 2)
- **Tester Usernames** (e.g., "alice_tester, bob_tester")
- **Module Mappings** count

You can expand **"View Full Configuration"** to see the complete YAML.

### Step 4: Validate & Create Project

Click the **"🔍 Validate & Create Project"** button.

The system will:
1. Validate config structure ✅
2. Check base folder ✅
3. Verify project name is unique ✅
4. Test web URL accessibility ⚠️ (warning if not reachable)
5. Create folder structure ✅
6. Create initial files ✅
7. Save config.yaml ✅
8. Register in database ✅

### Step 5: View Results

#### If Successful ✅

You'll see:
```
✅ Project Created Successfully!

📊 Project Details
Project Name: sample_test_project
Project ID: 1
Testers Created: 2
📁 Project Location: C:/Projects/AI_Chat/PLCD/TA_AI_Project/Projects/sample_test_project
```

And a table showing:
```
👥 Tester Accounts

Username         Password        Login URL
alice_tester     alice123        http://localhost:8501/tester (TODO)
bob_tester       bob456          http://localhost:8501/tester (TODO)
```

Click **"➕ Create Another Project"** to reset and upload another config.

#### If Failed ❌

You'll see:
```
❌ Project Creation Failed

🚫 Errors:
Error: Project 'sample_test_project' already exists
```

Click **"🔄 Try Again"** to clear and start over.

## What Gets Created?

When a project is successfully created:

### 1. Folder Structure
```
Projects/
└── sample_test_project/
    ├── config.yaml               ← Your uploaded config
    ├── feedback_log.json         ← Empty JSON array
    ├── Jira_Tickets/            ← For Jira ticket files
    ├── Selectors_Folder/        ← UI selectors
    │   └── selectors.json       ← Empty JSON array
    ├── Reports/                 ← Test reports
    ├── Videos/                  ← Test recordings
    ├── Generated_Scripts/       ← Playwright scripts
    └── Logs/                    ← Log files
```

### 2. Database Records

**Projects Table:**
- ID: 1
- Name: "sample_test_project"
- Base Folder: "Projects"
- Created At: "2025-10-21T15:30:00"

**Testers Table:**
- ID: 1, Username: "alice_tester", Password: (hashed), Project ID: 1
- ID: 2, Username: "bob_tester", Password: (hashed), Project ID: 2

## Common Use Cases

### Use Case 1: Create Your First Project

1. Copy `sample_config.yaml` to `my_config.yaml`
2. Edit `my_config.yaml`:
   ```yaml
   project:
     name: "my_first_project"  # Change this to unique name

   testers:
     - username: "my_username"
       password: "my_password"
   ```
3. Upload `my_config.yaml` through the web UI
4. Click validate
5. Done!

### Use Case 2: Create Multiple Projects

1. Upload first config → Validate → Success
2. Click "Create Another Project"
3. Upload second config → Validate → Success
4. Repeat as needed

### Use Case 3: Fix Configuration Errors

1. Upload config → Validate → Failed
2. Read error messages (e.g., "Missing required field: 'testers'")
3. Edit your config file to fix the error
4. Click "Try Again"
5. Upload the corrected file
6. Validate again

## Validation Details

Click **"🔍 Validation Steps Details"** to see exactly what was checked:

```
✅ validate_structure: All required fields present and valid
✅ check_base_folder: Base folder validated: C:/Projects/...
✅ check_project_uniqueness: Project name 'my_project' is unique
⚠️ check_web_url: Cannot connect to web URL: https://...
○ validate_login: Login validation not yet implemented (TODO)
✅ create_folders: Created 7 project folders
✅ create_initial_files: Created 2 initial files
✅ save_config: Config saved to: C:/Projects/.../config.yaml
✅ create_database_records: Created project (ID: 1) and 2 tester(s)
```

## Tips & Tricks

### Tip 1: Test with Sample Config
Always test with `sample_config.yaml` first to ensure the system works.

### Tip 2: Unique Project Names
Each project needs a unique name. If you get "already exists" error, change the project name.

### Tip 3: Web URL Warnings
If the web URL is not accessible, you'll get a warning but the project will still be created. This is normal if the URL is internal or requires VPN.

### Tip 4: Database Persistence
Projects are stored in `database/testers.db`. Once created, they persist across sessions.

### Tip 5: View Created Projects
Check the `Projects/` folder to see all created project folders.

## Troubleshooting

### Problem: "Project already exists"
**Solution:** Change the project name in your config.yaml or delete the existing project from the database.

### Problem: "Base folder not writable"
**Solution:** Check folder permissions or run as administrator.

### Problem: Port 8501 in use
**Solution:** Run on different port:
```bash
streamlit run web_app.py --server.port 8502
```

### Problem: Invalid YAML
**Solution:** Check your YAML syntax. Use an online YAML validator if needed.

### Problem: Import errors
**Solution:** Run `python test_streamlit_imports.py` to diagnose.

## Stopping the Server

Press **Ctrl + C** in the terminal to stop the Streamlit server.

## Next Steps

After creating projects:
1. Add Jira ticket files to `Jira_Tickets/` folder
2. Testers can log in (once tester portal is implemented)
3. Run test automation using `run_test.py`

## Security Notes

⚠️ **Important:**
- The admin portal has no authentication (TODO)
- Keep config files secure (contain API keys)
- Passwords are shown only during creation
- Passwords are hashed in database using bcrypt

## Need Help?

1. Check validation error messages
2. Review logs
3. Run test scripts
4. Check `STREAMLIT_README.md` for detailed documentation
