# Test Automation - Admin Portal

A Streamlit web application for managing test automation projects.

## Overview

This admin portal allows administrators to:
- Upload `config.yaml` files
- Validate project configurations
- Create new testing projects
- Set up project folder structures
- Register projects and testers in the database

## Prerequisites

- Python 3.11+
- Virtual environment activated
- All dependencies installed (see `requirements.txt`)

## Quick Start

### 1. Start the Streamlit Application

```bash
streamlit run web_app.py
```

The application will open in your default browser at: `http://localhost:8501`

### 2. Upload Configuration File

1. Click "Browse files" or drag and drop a `config.yaml` file
2. Review the configuration preview
3. Click "Validate & Create Project"
4. View the results

### 3. Sample Configuration

A sample configuration file is provided: `sample_config.yaml`

You can use this as a template for creating new projects.

## Configuration File Structure

A valid `config.yaml` must contain:

```yaml
base_folder: "Projects"              # Base folder for projects

project:
  name: "your_project_name"          # Unique project name

testers:                             # At least 1 tester required
  - username: "tester1"
    password: "password123"

web_application:
  url: "https://your-app-url.com"    # Web application URL
  test_credentials:
    username: "test@example.com"
    password: "testpass"

wait_times:                          # Wait time configurations
  default: 5
  long: 10

folders:                             # Folder structure
  jira_tickets: "Jira_Tickets"

azure_openai:                        # Azure OpenAI config
  api_key: "your_key"
  endpoint: "https://your-endpoint"

module_mapping:                      # Module mappings
  login: "login_module"

execution:                           # Execution settings
  headless: true
  timeout: 30000
```

## Features

### Configuration Validation

The system validates:
- ✅ Required fields are present
- ✅ Base folder exists and is writable
- ✅ Project name is unique
- ✅ Web URL is accessible (warning if not)
- ✅ Tester credentials are valid

### Project Setup

On successful validation, the system:
1. Creates project folder structure
2. Creates initial files (`selectors.json`, `feedback_log.json`)
3. Saves `config.yaml` to project folder
4. Registers project in database
5. Creates tester accounts

### Folder Structure Created

```
Projects/
└── your_project_name/
    ├── config.yaml
    ├── feedback_log.json
    ├── Jira_Tickets/
    ├── Selectors_Folder/
    │   └── selectors.json
    ├── Reports/
    ├── Videos/
    ├── Generated_Scripts/
    └── Logs/
```

## Validation Results

### Success

When a project is created successfully, you'll see:
- ✅ Success message
- Project name and ID
- Project location path
- Number of testers created
- Tester account credentials (username/password)

### Failure

If validation fails, you'll see:
- ❌ Error message
- List of validation errors
- Validation warnings (if any)
- Which validation steps passed/failed

## Error Handling

The system handles:
- Invalid YAML syntax
- Missing required fields
- Duplicate project names
- Duplicate usernames
- Inaccessible folders
- Network errors (web URL checks)

If an error occurs, all changes are rolled back (folders deleted).

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run web_app.py --server.port 8502
```

### Import Errors

Run the test script to verify all imports:

```bash
python test_streamlit_imports.py
```

### Database Issues

The database is located at: `database/testers.db`

To view database contents, you can use SQLite tools or the test scripts.

### Logs

Check the application logs for detailed error information.

## Testing

### Test the Application

Run the import test:

```bash
python test_streamlit_imports.py
```

Run the project manager agent test:

```bash
python test_project_manager.py
```

### Create a Test Project

1. Use the provided `sample_config.yaml`
2. Modify the `project.name` to something unique
3. Upload through the web interface
4. Verify project is created in `Projects/` folder

## Security Notes

- Passwords are stored hashed in the database (bcrypt)
- Passwords are shown in the UI only during initial setup
- Config files should be kept secure (contain API keys)
- The admin portal should be restricted to administrators only

## Future Enhancements

- [ ] Tester login portal (currently TODO)
- [ ] Project management (edit, delete projects)
- [ ] User management (edit, delete testers)
- [ ] Login validation using Playwright
- [ ] Multi-page application (admin, tester, reports)
- [ ] Authentication for admin portal

## File Structure

```
TA_AI_Project/
├── web_app.py                    # Main Streamlit entry point
├── web/
│   ├── __init__.py
│   └── admin_upload.py           # Admin upload page
├── agents/
│   └── project_manager_agent.py  # Project validation & setup
├── database/
│   ├── db.py                     # Database operations
│   ├── models.py                 # Pydantic models
│   └── testers.db                # SQLite database
├── sample_config.yaml            # Sample configuration
└── STREAMLIT_README.md           # This file
```

## Support

For issues or questions:
1. Check the logs
2. Run test scripts
3. Review validation error messages
4. Check database state

## License

Internal use only.
