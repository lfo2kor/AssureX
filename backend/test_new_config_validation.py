"""
Test script to verify the updated config validation works with the new format.
"""
import sys
import io
import yaml

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agents.project_manager_agent import validate_config_structure

print("="*80)
print("Testing Updated Config Validation")
print("="*80)

# Test 1: Load and validate plcdtest_config.yaml
print("\nTest 1: Validating plcdtest_config.yaml")
print("-"*80)

try:
    with open('plcdtest_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("✓ Config file loaded successfully")
    print(f"  Project name: {config.get('project', {}).get('name', 'N/A')}")
    print(f"  Testers: {len(config.get('testers', []))}")
    print(f"  Module mappings: {len(config.get('module_mapping', []))}")

    # Validate structure
    is_valid, errors = validate_config_structure(config)

    if is_valid:
        print("\n✅ Config validation PASSED!")
        print("   All required fields are present and valid")
    else:
        print("\n❌ Config validation FAILED!")
        print(f"   Found {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")

except FileNotFoundError:
    print("✗ plcdtest_config.yaml not found")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Test with sample_config.yaml (old format)
print("\n" + "="*80)
print("\nTest 2: Validating sample_config.yaml (old format)")
print("-"*80)

try:
    with open('sample_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("✓ Config file loaded successfully")

    # Validate structure
    is_valid, errors = validate_config_structure(config)

    if is_valid:
        print("\n✅ Config validation PASSED!")
        print("   (Unexpected - this should fail with old format)")
    else:
        print("\n✓ Config validation correctly identified issues with old format")
        print(f"   Found {len(errors)} error(s):")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

except FileNotFoundError:
    print("○ sample_config.yaml not found (skipping)")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Test with minimal valid config
print("\n" + "="*80)
print("\nTest 3: Validating minimal valid config (programmatic)")
print("-"*80)

minimal_config = {
    "base_folder": "Projects",
    "project": {
        "name": "test_project"
    },
    "testers": [
        {
            "username": "testuser",
            "password": "password123"
        }
    ],
    "web_application": {
        "url": "http://example.com",
        "browser": "edge",
        "environment": "test",
        "test_credentials": {
            "username": "test",
            "password": "test123"
        }
    },
    "wait_times": {
        "after_login": 3000,
        "after_navigation": 2000,
        "after_click": 1000,
        "after_type": 500,
        "after_dropdown": 1000,
        "page_load": 5000
    },
    "folders": {
        "jira": "Jira_Tickets",
        "reports": "Reports",
        "videos": "Videos",
        "scripts": "Generated_Scripts",
        "logs": "Logs",
        "selectors": "Selectors_Folder"
    },
    "azure_openai": {
        "api_key": "test_key",
        "endpoint": "https://test.openai.azure.com",
        "api_version": "2024-02-15-preview",
        "deployment_gpt4o": "gpt-4o"
    },
    "module_mapping": [
        {
            "jira_name": "test",
            "web_app_name": "Test"
        }
    ],
    "execution": {
        "max_retries": 3,
        "screenshot_on_every_step": True,
        "record_video": True,
        "generate_script": True,
        "headless": False
    }
}

is_valid, errors = validate_config_structure(minimal_config)

if is_valid:
    print("✅ Minimal config validation PASSED!")
else:
    print("❌ Minimal config validation FAILED!")
    print(f"   Found {len(errors)} error(s):")
    for error in errors:
        print(f"   - {error}")

# Test 4: Test with missing required fields
print("\n" + "="*80)
print("\nTest 4: Validating config with missing fields")
print("-"*80)

incomplete_config = {
    "base_folder": "Projects",
    "project": {
        "name": "test"
    },
    "testers": [
        {
            "username": "test",
            "password": "pass"
        }
    ]
    # Missing: web_application, wait_times, folders, azure_openai, module_mapping, execution
}

is_valid, errors = validate_config_structure(incomplete_config)

if not is_valid:
    print("✓ Correctly identified missing fields")
    print(f"   Found {len(errors)} error(s):")
    for error in errors[:5]:
        print(f"   - {error}")
    if len(errors) > 5:
        print(f"   ... and {len(errors) - 5} more errors")
else:
    print("✗ Should have failed (missing required fields)")

# Test 5: Test with invalid tester credentials
print("\n" + "="*80)
print("\nTest 5: Validating testers with short username/password")
print("-"*80)

invalid_tester_config = minimal_config.copy()
invalid_tester_config['testers'] = [
    {
        "username": "ab",  # Too short (< 3 chars)
        "password": "12345"  # Too short (< 6 chars)
    }
]

is_valid, errors = validate_config_structure(invalid_tester_config)

if not is_valid:
    print("✓ Correctly identified invalid tester credentials")
    print(f"   Found {len(errors)} error(s):")
    for error in errors:
        print(f"   - {error}")
else:
    print("✗ Should have failed (invalid credentials)")

print("\n" + "="*80)
print("All tests completed!")
print("="*80 + "\n")
