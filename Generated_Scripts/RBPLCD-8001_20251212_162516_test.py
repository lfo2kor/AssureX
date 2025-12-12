"""
Auto-generated Playwright Test Script
Generated from: RBPLCD-8001
Title: Edit teststep measurement and change Type attribute
Generated: 2025-12-12 16:25:16
"""

import pytest
from playwright.sync_api import Page, expect


def test_rbplcd_8001(page: Page):
    """
    Test: Edit teststep measurement and change Type attribute
    Module: Teststep
    """
    # Initialize step results list
    step_results = []

    # Step 0: Login
    print("Step 0: Login")
    page.goto("http://fe0vm03313.de.bosch.com/rbplcd_t/client/login")
    page.wait_for_load_state('networkidle')

    # Enter credentials
    page.fill('input[type="text"]', "mechanic")
    page.fill('input[type="password"]', "avalon")

    # Click login button
    page.click("[data-loginBtn='loginBtn']")
    page.wait_for_load_state('networkidle')

    print("[OK] Login successful")

    # Record login step
    step_results.append({
        'step_number': 0,
        'step_text': 'Login',
        'selector': "[data-loginBtn='loginBtn']",
        'agent_used': 'N/A',
        'confidence': 1.0,
        'status': 'PASSED'
    })

    # Step 2: Click Runs link in sidebar to navigate to Teststep module
    # Selector discovered by: Agent L1 (confidence: 0.80)
    # Action type: click
    print("Step 2: Click Runs link in sidebar to navigate to Teststep module")

    # Wait for element to be available
    page.wait_for_selector("[data-test='sidebar-nav-item-nav_item_teststeps']", timeout=10000)

    # Click action
    page.click("[data-test='sidebar-nav-item-nav_item_teststeps']")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 2,
        'step_text': 'Click Runs link in sidebar to navigate to Teststep module',
        'selector': "[data-test='sidebar-nav-item-nav_item_teststeps']",
        'agent_used': 'L1',
        'confidence': 0.80,
        'status': 'PASSED'
    })

    # Step 3: Click on table row for Teststep named default_Measurement01
    # Selector discovered by: Agent L1 (confidence: 0.79)
    # Action type: click
    print("Step 3: Click on table row for Teststep named default_Measurement01")

    # Wait for element to be available
    page.wait_for_selector("[data-attribute='default_Measurement01']", timeout=10000)

    # Click action
    page.click("[data-attribute='default_Measurement01']")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 3,
        'step_text': 'Click on table row for Teststep named default_Measurement01',
        'selector': "[data-attribute='default_Measurement01']",
        'agent_used': 'L1',
        'confidence': 0.79,
        'status': 'PASSED'
    })

    # Step 4: Click to expand Parts accordion section
    # Selector discovered by: Agent L1 (confidence: 0.97)
    # Action type: click
    print("Step 4: Click to expand Parts accordion section")

    # Wait for element to be available
    page.wait_for_selector("[data-expensionpanelheader="aeName.UnitUnderTest.names"]", timeout=10000)

    # Click action
    page.click("[data-expensionpanelheader="aeName.UnitUnderTest.names"]")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 4,
        'step_text': 'Click to expand Parts accordion section',
        'selector': "[data-expensionpanelheader="aeName.UnitUnderTest.names"]",
        'agent_used': 'L1',
        'confidence': 0.97,
        'status': 'PASSED'
    })

    # Step 5: Click edit button for test object default_testobject_01
    # Selector discovered by: Agent L1 (confidence: 0.90)
    # Action type: click
    print("Step 5: Click edit button for test object default_testobject_01")

    # Wait for element to be available
    page.wait_for_selector("[data-editnode='default_testobject_01']", timeout=10000)

    # Click action
    page.click("[data-editnode='default_testobject_01']")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 5,
        'step_text': 'Click edit button for test object default_testobject_01',
        'selector': "[data-editnode='default_testobject_01']",
        'agent_used': 'L1',
        'confidence': 0.90,
        'status': 'PASSED'
    })

    # Step 6: Click on Type dropdown field
    # Selector discovered by: Agent L1 (confidence: 1.00)
    # Action type: click
    print("Step 6: Click on Type dropdown field")

    # Wait for element to be available
    page.wait_for_selector("[data-attribute="Type"]", timeout=10000)

    # Click action
    page.click("[data-attribute="Type"]")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 6,
        'step_text': 'Click on Type dropdown field',
        'selector': "[data-attribute="Type"]",
        'agent_used': 'L1',
        'confidence': 1.00,
        'status': 'PASSED'
    })

    # Step 7: Choose Type 4 in dropdown menu
    # Selector discovered by: Agent L1 (confidence: 1.00)
    # Action type: click
    print("Step 7: Choose Type 4 in dropdown menu")

    # Wait for element to be available
    page.wait_for_selector("[data-autocompleteitem="Type 3"]", timeout=10000)

    # Click action
    page.click("[data-autocompleteitem="Type 3"]")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 7,
        'step_text': 'Choose Type 4 in dropdown menu',
        'selector': "[data-autocompleteitem="Type 3"]",
        'agent_used': 'L1',
        'confidence': 1.00,
        'status': 'PASSED'
    })

    # Step 8: Click Save button
    # Selector discovered by: Agent L1 (confidence: 0.75)
    # Action type: click
    print("Step 8: Click Save button")

    # Wait for element to be available
    page.wait_for_selector("[data-savebtn='SaveBtn']", timeout=10000)

    # Click action
    page.click("[data-savebtn='SaveBtn']")
    page.wait_for_load_state('networkidle')
    print("[OK] Clicked element")

    # Record step result
    step_results.append({
        'step_number': 8,
        'step_text': 'Click Save button',
        'selector': "[data-savebtn='SaveBtn']",
        'agent_used': 'L1',
        'confidence': 0.75,
        'status': 'PASSED'
    })

    # Step 9: Verify success message Successfully edited: \'TestObject\' default_testobject_01 is displayed
    # Selector discovered by: Agent L3 (confidence: 1.00)
    # Action type: verify_text
    print("Step 9: Verify success message Successfully edited: \'TestObject\' default_testobject_01 is displayed")

    # Text verification (L3 Vision)
    # Use nth(0) to handle multiple matches (e.g., text in table + tooltip)
    locator = page.get_by_text("Successfully edited: 'TestObject' default_testobject_01", exact=False)
    if locator.count() > 1:
        locator = locator.nth(0)
    expect(locator).to_be_visible()
    print("[OK] Verified text: Successfully edited: 'TestObject' default_testobject_01")

    # Record step result
    step_results.append({
        'step_number': 9,
        'step_text': 'Verify success message Successfully edited: \'TestObject\' default_testobject_01 is displayed',
        'verified_text': "Successfully edited: 'TestObject' default_testobject_01",
        'agent_used': 'L3',
        'confidence': 1.00,
        'status': 'PASSED'
    })

    print("[OK] Test completed successfully")

    return step_results


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
    ticket_data = {
        'title': 'Edit teststep measurement and change Type attribute',
        'module': 'Teststep',
        'expected_result': ''
    }

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
            step_results = test_rbplcd_8001(page)
            overall_status = "PASSED"
        except Exception as e:
            print(f"[FAILED] Test failed: {e}")
            overall_status = "FAILED"
            step_results = []
        finally:
            context.close()
            browser.close()

    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()

    # Generate beautiful HTML report using custom generator
    report_path = generate_html_report(
        ticket_id="RBPLCD-8001",
        ticket_data=ticket_data,
        step_results=step_results,
        overall_status=overall_status,
        execution_time=execution_time,
        config=config
    )

    print(f"\n[OK] HTML Report: {report_path}")
