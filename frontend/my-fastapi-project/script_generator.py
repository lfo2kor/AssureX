"""
PLCD Testing Assistant - Playwright Script Generator
Generates executable Playwright test scripts from successful executions
"""

import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_playwright_script(
    ticket_id: str,
    ticket_data: Dict,
    step_results: List[Dict],
    config: Dict
) -> str:
    """
    Generate executable Playwright Python test script

    Args:
        ticket_id: Jira ticket ID
        ticket_data: Parsed ticket data
        step_results: List of step execution results
        config: Configuration dictionary

    Returns:
        Path to generated script file
    """
    # Create Generated_Scripts folder
    scripts_folder = Path(config['folders']['generated_scripts'])
    scripts_folder.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"{ticket_id}_{timestamp}_test.py"
    script_path = scripts_folder / script_filename

    # Extract configuration
    app_config = config.get('application', {})
    base_url = app_config.get('base_url', 'http://localhost')
    login_url = app_config.get('login_url', f"{base_url}/login")
    username = app_config.get('credentials', {}).get('username', 'testuser')
    password = app_config.get('credentials', {}).get('password', 'password')

    # Filter only successful steps
    successful_steps = [s for s in step_results if s.get('status') == 'PASSED']

    # Generate imports
    script_content = f'''"""
Auto-generated Playwright Test Script
Generated from: {ticket_id}
Title: {ticket_data.get('title', 'N/A')}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pytest
from playwright.sync_api import Page, expect


def test_{ticket_id.lower().replace('-', '_')}(page: Page):
    """
    Test: {ticket_data.get('title', 'N/A')}
    Module: {ticket_data.get('module', 'N/A')}
    """

    # Step 0: Login
    print("Step 0: Login")
    page.goto("{login_url}")
    page.wait_for_load_state('networkidle')

    # Enter credentials
    page.fill("input[name='userId']", "{username}")
    page.fill("input[name='password']", "{password}")

    # Click login button
    page.click("[data-loginBtn='loginBtn']")
    page.wait_for_load_state('networkidle')

    print("✓ Login successful")
'''

    # Generate test steps
    for step_result in successful_steps:
        step_num = step_result.get('step_number', '?')
        step_text = step_result.get('step_text', 'N/A')
        selector = step_result.get('selector', 'N/A')
        agent_used = step_result.get('agent_used', 'N/A')
        confidence = step_result.get('confidence', 0)

        # Escape quotes in step_text
        step_text_escaped = step_text.replace('"', '\\"')

        script_content += f'''
    # Step {step_num}: {step_text_escaped}
    # Selector discovered by: Agent {agent_used} (confidence: {confidence:.2f})
    print("Step {step_num}: {step_text_escaped}")

    # Wait for element to be available
    page.wait_for_selector("{selector}", timeout=10000)

'''

        # Determine action based on step text
        step_lower = step_text.lower()

        if 'navigate' in step_lower or 'click' in step_lower or 'open' in step_lower:
            script_content += f'''    # Click action
    page.click("{selector}")
    page.wait_for_load_state('networkidle')
    print("✓ Clicked element")
'''

        elif 'enter' in step_lower or 'input' in step_lower or 'type' in step_lower:
            # Extract value if mentioned in step text
            script_content += f'''    # Input action (modify value as needed)
    page.fill("{selector}", "test_value")
    print("✓ Entered value")
'''

        elif 'select' in step_lower and 'dropdown' in step_lower:
            script_content += f'''    # Select from dropdown (modify option as needed)
    page.select_option("{selector}", label="Option 1")
    print("✓ Selected option")
'''

        elif 'save' in step_lower or 'submit' in step_lower:
            script_content += f'''    # Save/Submit action
    page.click("{selector}")
    page.wait_for_load_state('networkidle')
    print("✓ Saved/Submitted")
'''

        elif 'verify' in step_lower or 'check' in step_lower or 'should' in step_lower:
            script_content += f'''    # Verification
    expect(page.locator("{selector}")).to_be_visible()
    print("✓ Verified element visible")
'''

        else:
            # Generic click
            script_content += f'''    # Generic action
    page.click("{selector}")
    page.wait_for_timeout(1000)
    print("✓ Action completed")
'''

    # Add final verification if expected result exists
    expected_result = ticket_data.get('expected_result', '')
    if expected_result:
        expected_escaped = expected_result[:100].replace('"', '\\"').replace('\n', ' ')
        script_content += f'''
    # Expected Result Verification
    # {expected_escaped}
    print("✓ Test completed successfully")


if __name__ == "__main__":
    """Run test with pytest"""
    pytest.main([__file__, "-v", "-s"])
'''
    else:
        script_content += '''
    print("✓ Test completed successfully")


if __name__ == "__main__":
    """Run test with pytest"""
    pytest.main([__file__, "-v", "-s"])
'''

    # Write to file
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return str(script_path)


def generate_pytest_config(config: Dict):
    """
    Generate pytest configuration file for generated scripts

    Args:
        config: Configuration dictionary
    """
    scripts_folder = Path(config['folders']['generated_scripts'])
    scripts_folder.mkdir(parents=True, exist_ok=True)

    conftest_path = scripts_folder / "conftest.py"

    # Extract browser config
    browser_config = config.get('browser', {})
    browser_type = browser_config.get('type', 'chromium')
    headless = browser_config.get('headless', False)

    conftest_content = f'''"""
Pytest configuration for generated Playwright tests
"""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="function")
def page():
    """
    Playwright page fixture - creates browser and page for each test
    """
    with sync_playwright() as p:
        # Launch browser
        browser = p.{browser_type}.launch(
            headless={headless},
            args=['--start-maximized']
        )

        # Create context without viewport to allow maximized window
        context = browser.new_context(no_viewport=True)

        # Create page
        page = context.new_page()

        # Set default timeout
        page.set_default_timeout(30000)

        yield page

        # Cleanup
        context.close()
        browser.close()
'''

    # Write conftest.py
    with open(conftest_path, 'w', encoding='utf-8') as f:
        f.write(conftest_content)

    return str(conftest_path)


def generate_readme(config: Dict):
    """
    Generate README for generated scripts folder

    Args:
        config: Configuration dictionary
    """
    scripts_folder = Path(config['folders']['generated_scripts'])
    scripts_folder.mkdir(parents=True, exist_ok=True)

    readme_path = scripts_folder / "README.md"

    readme_content = '''# Generated Playwright Test Scripts

This folder contains auto-generated Playwright test scripts from successful PLCD test executions.

## Prerequisites

```bash
pip install playwright pytest
playwright install
```

## Running Tests

### Run a specific test:
```bash
pytest RBPLCD-8835_20250117_120000_test.py -v -s
```

### Run all tests in folder:
```bash
pytest . -v -s
```

### Run with HTML report:
```bash
pytest . --html=report.html --self-contained-html
```

## Script Structure

Each generated script contains:
- **Auto-discovered selectors**: From Agent 1 (Semantic Search) or Agent 2 (DOM Discovery)
- **Confidence scores**: Indicating selector reliability
- **Comments**: Original test step descriptions
- **Actions**: Click, fill, select, verify operations

## Customization

Generated scripts are templates. You may need to:
1. Adjust input values (currently set to "test_value")
2. Add custom assertions for expected results
3. Modify timeouts for slower operations
4. Add additional verification steps

## Maintenance

- Scripts are timestamped to track generation time
- Each script is standalone and can be executed independently
- Update `conftest.py` to modify global browser settings
'''

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    return str(readme_path)


if __name__ == "__main__":
    print("Playwright Script Generator")
    print("This module is used by plcd_ta.py to generate test scripts.")
    print("Run plcd_ta.py to execute tests and generate scripts.")
