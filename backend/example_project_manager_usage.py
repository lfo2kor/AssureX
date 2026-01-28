"""
Example usage of the Project Manager Agent.

This script demonstrates how to use the project_manager_agent to validate
and set up a new project.
"""
import os
import yaml
from agents.project_manager_agent import project_manager_agent


def example_usage():
    """Example of using the project manager agent."""

    # Example 1: Load config from YAML file
    print("Example 1: Loading config from YAML file")
    print("="*80)

    # You would typically load this from an uploaded file
    config_path = "config.yaml"  # Replace with actual config path

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            uploaded_config = yaml.safe_load(f)

        # Get current directory as app root
        app_root = os.getcwd()

        # Create state
        state = {
            'uploaded_config': uploaded_config,
            'app_root': app_root
        }

        # Run project manager agent
        result = project_manager_agent(state)

        # Check results
        if result['validation_status'] == 'success':
            print(f"✓ Project created successfully!")
            print(f"  Project: {result['project_name']}")
            print(f"  Path: {result['project_path']}")
            print(f"  Project ID: {result['project_id']}")
            print(f"  Testers created: {result['testers_created']}")
        else:
            print(f"✗ Project creation failed!")
            print(f"  Errors: {result['validation_errors']}")

        print("\nValidation Steps:")
        for step in result['validation_steps']:
            print(f"  - {step['step']}: {step['status']} - {step['message']}")

    else:
        print(f"Config file not found: {config_path}")

    print("="*80 + "\n")


    # Example 2: Create config programmatically
    print("Example 2: Creating config programmatically")
    print("="*80)

    # Build config dictionary
    config = {
        "base_folder": "Projects",
        "project": {
            "name": "my_new_project"
        },
        "testers": [
            {
                "username": "alice",
                "password": "alice_password"
            },
            {
                "username": "bob",
                "password": "bob_password"
            }
        ],
        "web_application": {
            "url": "https://example.com",
            "test_credentials": {
                "username": "test_user",
                "password": "test_pass"
            }
        },
        "wait_times": {
            "default": 5,
            "long": 10,
            "short": 2
        },
        "folders": {
            "jira_tickets": "Jira_Tickets",
            "selectors": "Selectors_Folder",
            "reports": "Reports"
        },
        "azure_openai": {
            "api_key": "your_api_key_here",
            "endpoint": "https://your-endpoint.openai.azure.com",
            "deployment_name": "gpt-4"
        },
        "module_mapping": {
            "login": "login_module",
            "navigation": "navigation_module",
            "forms": "forms_module"
        },
        "execution": {
            "headless": True,
            "timeout": 30000,
            "slow_mo": 0
        }
    }

    app_root = os.getcwd()

    state = {
        'uploaded_config': config,
        'app_root': app_root
    }

    # Run project manager agent
    result = project_manager_agent(state)

    # Display results
    print(f"Status: {result['validation_status']}")

    if result.get('validation_warnings'):
        print("\nWarnings:")
        for warning in result['validation_warnings']:
            print(f"  ⚠ {warning}")

    if result.get('validation_errors'):
        print("\nErrors:")
        for error in result['validation_errors']:
            print(f"  ✗ {error}")

    print("="*80 + "\n")


    # Example 3: Handling validation errors
    print("Example 3: Handling validation errors")
    print("="*80)

    # Invalid config (missing required fields)
    invalid_config = {
        "base_folder": "Projects",
        "project": {
            "name": "invalid_project"
        }
        # Missing testers, web_application, etc.
    }

    state = {
        'uploaded_config': invalid_config,
        'app_root': app_root
    }

    result = project_manager_agent(state)

    print(f"Status: {result['validation_status']}")
    print(f"Errors: {len(result.get('validation_errors', []))}")

    for error in result.get('validation_errors', []):
        print(f"  ✗ {error}")

    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nPROJECT MANAGER AGENT - USAGE EXAMPLES")
    print("="*80 + "\n")

    print("NOTE: These examples show how to use the project_manager_agent.")
    print("      Uncomment the example_usage() call below to run them.\n")

    # Uncomment to run examples (will create actual projects)
    # example_usage()

    print("To use the agent:")
    print("1. Prepare your config dictionary or load from YAML")
    print("2. Create state with 'uploaded_config' and 'app_root'")
    print("3. Call project_manager_agent(state)")
    print("4. Check result['validation_status'] for success/failure")
    print("5. Review validation_steps, errors, and warnings")
    print("\n")
