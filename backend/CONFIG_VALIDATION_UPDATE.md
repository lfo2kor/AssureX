# Config Validation Update Summary

## What Was Changed

Updated `agents/project_manager_agent.py` to validate the **new config format** instead of the old format.

## Updated Function

**File:** `agents/project_manager_agent.py`
**Function:** `validate_config_structure()` (lines 284-454)

## New Config Format Requirements

### Required Top-Level Fields

1. **base_folder** (string) - Base folder path
2. **project** (dict) - Project information
   - `name` (string) - Unique project name
3. **testers** (list) - At least 1 tester
   - `username` (string, min 3 chars)
   - `password` (string, min 6 chars)
4. **web_application** (dict) - Web app configuration
   - `url` (string) - Application URL
   - `browser` (string) - Browser type (edge, chromium, firefox, webkit)
   - `environment` (string) - Environment (test, staging, production)
   - `test_credentials` (dict)
     - `username` (string)
     - `password` (string)
5. **wait_times** (dict) - All timing configurations
   - `after_login`
   - `after_navigation`
   - `after_click`
   - `after_type`
   - `after_dropdown`
   - `page_load`
6. **folders** (dict) - Folder structure
   - `jira`
   - `reports`
   - `videos`
   - `scripts`
   - `logs`
   - `selectors`
7. **azure_openai** (dict) - Azure OpenAI configuration
   - `api_key`
   - `endpoint`
   - `api_version`
   - `deployment_gpt4o`
8. **module_mapping** (list) - Jira to web app mappings
   - Each item must have:
     - `jira_name` (string)
     - `web_app_name` (string)
9. **execution** (dict) - Execution settings
   - `max_retries`
   - `screenshot_on_every_step`
   - `record_video`
   - `generate_script`
   - `headless`

## Validation Results

### Test 1: plcdtest_config.yaml ✅
```
✅ Config validation PASSED!
   All required fields are present and valid
```

### Test 2: sample_config.yaml ✅
```
✅ Config validation PASSED!
   (Updated to new format)
```

### Test 3: Minimal Valid Config ✅
```
✅ Minimal config validation PASSED!
```

### Test 4: Missing Fields ✅
```
✓ Correctly identified missing fields
   Found 7 error(s)
```

### Test 5: Invalid Credentials ✅
```
✓ Correctly identified invalid tester credentials
   Found 2 error(s)
```

## What Was Removed

The validation **no longer checks** for these old fields:
- `login` (old field name)
- Old `module_mapping` as dict (now expects list)
- Generic dictionary checks without nested field validation

## What Was Added

New detailed validation for:
- `web_application.browser`
- `web_application.environment`
- All 6 wait_times fields
- All 6 folders fields
- All 4 azure_openai fields
- All 5 execution fields
- `module_mapping` as list with `jira_name` and `web_app_name`
- Tester credential length validation (min 3 chars username, min 6 chars password)

## Error Messages

The validation now provides detailed error messages:

**Example - Missing nested field:**
```
Missing required field: 'web_application.browser'
Missing required field: 'wait_times.after_login'
Missing required field: 'folders.jira'
```

**Example - Invalid tester:**
```
Tester 0 'username' must be at least 3 characters
Tester 0 'password' must be at least 6 characters
```

**Example - Wrong type:**
```
'module_mapping' must be a list
'web_application.test_credentials' must be a dictionary
```

## Files Updated

1. ✅ `agents/project_manager_agent.py` - Updated validation function
2. ✅ `sample_config.yaml` - Updated to new format
3. ✅ `test_new_config_validation.py` - New test script

## Files Validated

- ✅ `plcdtest_config.yaml` - Production config (PASSES)
- ✅ `sample_config.yaml` - Template config (PASSES)

## How to Test

Run the validation test:
```bash
python test_new_config_validation.py
```

Expected output: All 5 tests pass

## Backward Compatibility

⚠️ **Breaking Change:** Old config format will **NOT** pass validation.

Configs need to be updated to include:
- `web_application.browser`
- `web_application.environment`
- All required wait_times fields
- All required folders fields
- All required azure_openai fields
- All required execution fields
- `module_mapping` as list (not dict)

## Using the Streamlit App

The Streamlit admin portal will now:
1. Accept new config format ✅
2. Show detailed error messages for missing fields ✅
3. Validate all nested fields ✅
4. Provide clear guidance on what's missing ✅

Start the app:
```bash
streamlit run web_app.py
```

Upload `plcdtest_config.yaml` or `sample_config.yaml` - both will pass validation!

## Next Steps

1. Update any existing config files to new format
2. Use `plcdtest_config.yaml` or `sample_config.yaml` as template
3. Test project creation via Streamlit UI
4. Verify all validation steps pass

## Support

If validation fails:
1. Check the error messages - they're very specific
2. Compare your config to `plcdtest_config.yaml` or `sample_config.yaml`
3. Ensure all required fields are present
4. Check field types (list vs dict)
5. Verify nested fields exist
