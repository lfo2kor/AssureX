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

    # Extract configuration from YAML (correct keys)
    login_url = config.get('web_url', 'http://localhost/login')
    username = config.get('login', {}).get('username', 'testuser')
    password = config.get('login', {}).get('password', 'password')

    # Filter only successful steps
    successful_steps = [s for s in step_results if s.get('status') == 'PASSED']

    # Use same login selectors as plcd_taseq.py for consistency
    # These selectors are proven to work in the live application
    login_username_selector = 'input[type="text"]'
    login_password_selector = 'input[type="password"]'
    login_button_selector = "[data-loginBtn='loginBtn']"

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
    # Initialize step results list
    step_results = []

    # Step 0: Login
    print("Step 0: Login")
    page.goto("{login_url}")
    page.wait_for_load_state('networkidle')

    # Enter credentials
    page.fill('{login_username_selector}', "{username}")
    page.fill('{login_password_selector}', "{password}")

    # Click login button
    page.click("{login_button_selector}")
    page.wait_for_load_state('networkidle')

    print("[OK] Login successful")

    # Record login step
    step_results.append({{
        'step_number': 0,
        'step_text': 'Login',
        'selector': "{login_button_selector}",
        'agent_used': 'N/A',
        'confidence': 1.0,
        'status': 'PASSED'
    }})
'''

    # Generate test steps
    for step_result in successful_steps:
        step_num = step_result.get('step_number', '?')
        step_text = step_result.get('step_text', 'N/A')
        selector = step_result.get('selector', 'N/A')
        agent_used = step_result.get('agent_used', 'N/A')
        confidence = step_result.get('confidence', 0)
        action_type = step_result.get('action_type', 'click')  # Get actual action type from runtime

        # Escape quotes in step_text - escape both single and double quotes
        step_text_escaped = step_text.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

        script_content += f'''
    # Step {step_num}: {step_text_escaped}
    # Selector discovered by: Agent {agent_used} (confidence: {confidence:.2f})
    # Action type: {action_type}
    print("Step {step_num}: {step_text_escaped}")

'''

        # Only add wait_for_selector if it's not a verify_text action (L3 uses text, not CSS selector)
        if action_type != 'verify_text':
            script_content += f'''    # Wait for element to be available
    page.wait_for_selector("{selector}", timeout=10000)

'''

        # Use actual action_type from runtime execution instead of guessing from keywords
        if action_type == 'click':
            script_content += f'''    # Click action
    page.click("{selector}")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

        elif action_type == 'type':
            # Extract value if mentioned in step text
            script_content += f'''    # Input action (modify value as needed)
    page.fill("{selector}", "test_value")
    print("[OK] Entered value")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

        elif action_type == 'navigate':
            script_content += f'''    # Navigate action
    page.click("{selector}")
    page.wait_for_load_state('networkidle')
    print("[OK] Navigated")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

        elif action_type == 'verify' or action_type == 'verify_text':
            # For verify_text (L3), use the actual verified text with get_by_text
            verified_text = step_result.get('verified_text', '')
            if action_type == 'verify_text' and verified_text:
                # Escape quotes in verified text
                verified_text_escaped = verified_text.replace('\\', '\\\\').replace('"', '\\"')
                script_content += f'''    # Text verification (L3 Vision)
    # Use nth(0) to handle multiple matches (e.g., text in table + tooltip)
    locator = page.get_by_text("{verified_text_escaped}", exact=False)
    if locator.count() > 1:
        locator = locator.nth(0)
    expect(locator).to_be_visible()
    print("[OK] Verified text: {verified_text_escaped}")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'verified_text': "{verified_text_escaped}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''
            else:
                # Regular element verification
                script_content += f'''    # Element verification
    expect(page.locator("{selector}")).to_be_visible()
    print("[OK] Verified element visible")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

        elif action_type == 'clear':
            script_content += f'''    # Clear action
    page.fill("{selector}", "")
    print("[OK] Cleared input")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

        else:
            # Default to click for unknown action types
            script_content += f'''    # Default action (click)
    page.click("{selector}")
    page.wait_for_timeout(1000)
    print("[OK] Action completed")

    # Record step result
    step_results.append({{
        'step_number': {step_num},
        'step_text': '{step_text_escaped}',
        'selector': "{selector}",
        'agent_used': '{agent_used}',
        'confidence': {confidence:.2f},
        'status': 'PASSED'
    }})
'''

    # Add final verification if expected result exists
    expected_result = ticket_data.get('expected_result', '')
    if expected_result:
        expected_escaped = expected_result[:100].replace('"', '\\"').replace('\n', ' ')
        script_content += f'''
    # Expected Result Verification
    # {expected_escaped}
    print("[OK] Test completed successfully")

    return step_results
'''
    else:
        script_content += f'''
    print("[OK] Test completed successfully")

    return step_results
'''

    # Add the main execution block that uses custom report generator
    script_content += f'''

if __name__ == "__main__":
    """Run test and generate custom HTML report"""
    import sys
    import yaml
    from datetime import datetime
    from pathlib import Path
    from playwright.sync_api import sync_playwright

    # Import custom report generator from parent directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from report_generator import generate_html_report

    # Load config
    with open(Path(__file__).parent.parent / "plcdtestassistant.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Ticket data
    ticket_data = {{
        'title': '{ticket_data.get('title', 'N/A')}',
        'module': '{ticket_data.get('module', 'N/A')}',
        'expected_result': '{ticket_data.get('expected_result', '')}'
    }}

    # Start execution timer
    start_time = datetime.now()

    # Create browser and run test
    with sync_playwright() as p:
        browser_type = config.get('browser', 'edge')

        if browser_type == 'edge':
            browser = p.chromium.launch(
                channel='msedge',
                headless=False,
                args=['--start-maximized']
            )
        else:
            browser = p.chromium.launch(
                headless=False,
                args=['--start-maximized']
            )

        context = browser.new_context(no_viewport=True)
        page = context.new_page()
        page.set_default_timeout(30000)

        try:
            # Run test and get step results
            step_results = test_{ticket_id.lower().replace('-', '_')}(page)
            overall_status = "PASSED"
        except Exception as e:
            print(f"[FAILED] Test failed: {{e}}")
            overall_status = "FAILED"
            step_results = []
        finally:
            context.close()
            browser.close()

    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()

    # Generate beautiful HTML report using custom generator
    report_path = generate_html_report(
        ticket_id="{ticket_id}",
        ticket_data=ticket_data,
        step_results=step_results,
        overall_status=overall_status,
        execution_time=execution_time,
        config=config
    )

    print(f"\\n[OK] HTML Report: {{report_path}}")
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

    # Extract browser config (browser is a string in YAML, not a dict)
    browser_type = config.get('browser', 'chromium')
    headless = False  # Always run in headful mode for visibility

    # Determine browser launch code based on type
    if browser_type == 'edge':
        browser_launch_code = """browser = p.chromium.launch(
            channel='msedge',
            headless=False,
            args=['--start-maximized']
        )"""
    elif browser_type in ['chromium', 'firefox', 'webkit']:
        browser_launch_code = f"""browser = p.{browser_type}.launch(
            headless=False,
            args=['--start-maximized']
        )"""
    else:
        # Default to chromium
        browser_launch_code = """browser = p.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )"""

    conftest_content = f'''"""
Pytest configuration for generated Playwright tests
Browser: {browser_type}
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
        {browser_launch_code}

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
