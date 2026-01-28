"""
Test script for Project Manager Agent.
This script tests the project manager agent functionality.
"""
import sys
import io
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agents.project_manager_agent import project_manager_agent

# Sample config for testing
TEST_CONFIG = {
    "base_folder": "Projects",
    "project": {
        "name": "test_automation_project"
    },
    "testers": [
        {
            "username": "tester1",
            "password": "password123"
        },
        {
            "username": "tester2",
            "password": "secure456"
        }
    ],
    "web_application": {
        "url": "https://www.google.com",  # Using Google for testing accessibility
        "test_credentials": {
            "username": "test@example.com",
            "password": "testpass"
        }
    },
    "wait_times": {
        "default": 5,
        "long": 10
    },
    "folders": {
        "jira_tickets": "Jira_Tickets",
        "selectors": "Selectors_Folder"
    },
    "azure_openai": {
        "api_key": "dummy_key",
        "endpoint": "https://api.openai.com"
    },
    "module_mapping": {
        "login": "login_module",
        "navigation": "nav_module"
    },
    "execution": {
        "headless": True,
        "timeout": 30
    }
}


def print_validation_results(state: dict):
    """Print formatted validation results."""
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"\nStatus: {state.get('validation_status', 'UNKNOWN')}")

    if state.get('project_created'):
        print(f"✓ Project Created: {state.get('project_name')}")
        print(f"  Project ID: {state.get('project_id')}")
        print(f"  Project Path: {state.get('project_path')}")
        print(f"  Testers Created: {state.get('testers_created')}")

    # Print validation steps
    print("\nValidation Steps:")
    for step in state.get('validation_steps', []):
        step_name = step.get('step', 'unknown')
        status = step.get('status', 'unknown')
        message = step.get('message', '')

        if status == 'passed':
            symbol = '✓'
        elif status == 'failed':
            symbol = '✗'
        elif status == 'warning':
            symbol = '⚠'
        elif status == 'skipped':
            symbol = '○'
        else:
            symbol = '-'

        print(f"  {symbol} {step_name}: {message}")

    # Print errors
    if state.get('validation_errors'):
        print("\nErrors:")
        for error in state['validation_errors']:
            print(f"  ✗ {error}")

    # Print warnings
    if state.get('validation_warnings'):
        print("\nWarnings:")
        for warning in state['validation_warnings']:
            print(f"  ⚠ {warning}")

    print("="*80 + "\n")


def test_valid_config():
    """Test with valid config."""
    print("\n" + "="*80)
    print("TEST 1: Valid Configuration")
    print("="*80)

    app_root = os.getcwd()
    state = {
        'uploaded_config': TEST_CONFIG,
        'app_root': app_root
    }

    result = project_manager_agent(state)
    print_validation_results(result)

    return result.get('validation_status') == 'success'


def test_missing_fields():
    """Test with missing required fields."""
    print("\n" + "="*80)
    print("TEST 2: Missing Required Fields")
    print("="*80)

    # Config missing 'testers' field
    invalid_config = {
        "base_folder": "Projects",
        "project": {
            "name": "invalid_project"
        },
        # Missing 'testers' field
        "web_application": {
            "url": "https://example.com"
        },
        "wait_times": {},
        "folders": {},
        "azure_openai": {},
        "module_mapping": {},
        "execution": {}
    }

    app_root = os.getcwd()
    state = {
        'uploaded_config': invalid_config,
        'app_root': app_root
    }

    result = project_manager_agent(state)
    print_validation_results(result)

    return result.get('validation_status') == 'failed'


def test_duplicate_project():
    """Test creating project with duplicate name."""
    print("\n" + "="*80)
    print("TEST 3: Duplicate Project Name")
    print("="*80)

    # First, ensure the project exists (from test 1)
    # Try to create it again
    app_root = os.getcwd()
    state = {
        'uploaded_config': TEST_CONFIG,
        'app_root': app_root
    }

    result = project_manager_agent(state)
    print_validation_results(result)

    return result.get('validation_status') == 'failed'


def test_invalid_url():
    """Test with invalid web URL."""
    print("\n" + "="*80)
    print("TEST 4: Invalid Web URL (should warn but continue)")
    print("="*80)

    invalid_url_config = TEST_CONFIG.copy()
    invalid_url_config['project'] = {"name": "url_test_project"}
    invalid_url_config['web_application'] = {
        "url": "https://this-url-definitely-does-not-exist-12345.com",
        "test_credentials": {"username": "test", "password": "test"}
    }

    app_root = os.getcwd()
    state = {
        'uploaded_config': invalid_url_config,
        'app_root': app_root
    }

    result = project_manager_agent(state)
    print_validation_results(result)

    # Should succeed with warnings
    return result.get('validation_status') == 'success' and len(result.get('validation_warnings', [])) > 0


def cleanup_test_projects():
    """Cleanup test projects created during testing."""
    print("\n" + "="*80)
    print("CLEANUP: Removing test project folders")
    print("="*80)

    import shutil

    projects_to_clean = [
        "Projects/test_automation_project",
        "Projects/url_test_project"
    ]

    for project_path in projects_to_clean:
        full_path = Path(project_path)
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                print(f"✓ Removed: {full_path}")
            except Exception as e:
                print(f"✗ Failed to remove {full_path}: {e}")
        else:
            print(f"○ Not found: {full_path}")

    print("\nNote: Database records remain (no delete function implemented yet)")
    print("="*80 + "\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PROJECT MANAGER AGENT TEST SUITE")
    print("="*80)

    results = []

    # Test 1: Valid config
    try:
        results.append(("Valid Configuration", test_valid_config()))
    except Exception as e:
        print(f"Test 1 crashed: {e}")
        results.append(("Valid Configuration", False))

    # Test 2: Missing fields
    try:
        results.append(("Missing Required Fields", test_missing_fields()))
    except Exception as e:
        print(f"Test 2 crashed: {e}")
        results.append(("Missing Required Fields", False))

    # Test 3: Duplicate project
    try:
        results.append(("Duplicate Project", test_duplicate_project()))
    except Exception as e:
        print(f"Test 3 crashed: {e}")
        results.append(("Duplicate Project", False))

    # Test 4: Invalid URL
    try:
        results.append(("Invalid URL Warning", test_invalid_url()))
    except Exception as e:
        print(f"Test 4 crashed: {e}")
        results.append(("Invalid URL Warning", False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, passed_test in results:
        if passed_test:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    # Cleanup
    cleanup_test_projects()


if __name__ == "__main__":
    main()
