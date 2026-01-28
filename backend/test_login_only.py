"""
Test Login Process - CV-Guided Selector-Based Approach

This script tests login functionality using:
1. Standard HTML selectors (fast, no CV)
2. CV-guided specific selectors (if standard fails)
3. NO coordinate-based clicking
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.vision_helper import AzureVisionClient
from playwright.sync_api import sync_playwright
import time

print("=" * 80)
print("LOGIN TEST - CV-Guided Selector-Based Approach")
print("=" * 80)

# Load config
config = load_config("plcdtest_config.yaml")
logger = setup_logger(config['folders']['logs'], "LOGIN-TEST")

print(f"\nWeb URL: {config['web_url']}")
print(f"Username: {config['login']['username']}")
print(f"Password: {'*' * len(config['login']['password'])}")

# Initialize vision client
vision_client = AzureVisionClient(config, logger)
print("Vision client initialized")

# Launch browser
print("\nLaunching browser...")
playwright = sync_playwright().start()
browser = playwright.chromium.launch(channel='msedge', headless=False)
context = browser.new_context(viewport={'width': 1920, 'height': 1080})
page = context.new_page()

try:
    # Navigate to login page
    print(f"Navigating to: {config['web_url']}")
    page.goto(config['web_url'], wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(config['wait_times']['page_load'])

    # Take screenshot
    print("\nTaking screenshot of login page...")
    screenshot = page.screenshot(path='login_page_selector_test.png')
    print("Screenshot saved: login_page_selector_test.png")

    # Get credentials
    username = config['login']['username']
    password = config['login']['password']

    # METHOD 1: Try standard selectors first
    print("\n" + "=" * 80)
    print("METHOD 1: Trying Standard HTML Selectors")
    print("=" * 80)

    standard_worked = False
    try:
        username_count = page.locator('input[type="text"]').count()
        password_count = page.locator('input[type="password"]').count()
        button_count = page.locator('button[type="submit"]').count()

        print(f"Found elements:")
        print(f"  - Text inputs: {username_count}")
        print(f"  - Password inputs: {password_count}")
        print(f"  - Submit buttons: {button_count}")

        if username_count > 0 and password_count > 0 and button_count > 0:
            print("\nAttempting login with standard selectors...")

            # Fill username
            page.locator('input[type="text"]').first.fill(username)
            print(f"  Username filled: {username}")

            # Fill password
            page.locator('input[type="password"]').first.fill(password)
            print(f"  Password filled: {'*' * len(password)}")

            # Take screenshot after filling
            page.screenshot(path='form_filled_standard.png')
            print("  Screenshot: form_filled_standard.png")

            # Click submit button
            page.locator('button[type="submit"]').first.click()
            print("  Login button clicked")

            standard_worked = True
        else:
            print("Standard selectors incomplete - will try CV-guided approach")

    except Exception as e:
        print(f"Standard selectors failed: {e}")
        print("Will try CV-guided approach...")

    # METHOD 2: CV-guided selectors (if standard didn't work)
    if not standard_worked:
        print("\n" + "=" * 80)
        print("METHOD 2: Using CV to Identify Specific Selectors")
        print("=" * 80)

        print("\nCalling GPT-4o to analyze login page...")
        selector_result = vision_client.identify_login_selectors(screenshot)

        print(f"\nCV Analysis Results:")
        print(f"  Confidence: {selector_result.get('confidence', 0)}")
        print(f"  Reasoning: {selector_result.get('reasoning', 'N/A')}")
        print(f"\n  Username selector: {selector_result.get('username_selector', 'N/A')}")
        print(f"  Password selector: {selector_result.get('password_selector', 'N/A')}")
        print(f"  Button selector: {selector_result.get('button_selector', 'N/A')}")

        # Extract selectors
        username_selector = selector_result.get('username_selector', '')
        password_selector = selector_result.get('password_selector', '')
        button_selector = selector_result.get('button_selector', '')

        print("\nAttempting login with CV-identified selectors...")

        try:
            # Fill username
            if page.locator(username_selector).count() > 0:
                page.locator(username_selector).first.fill(username)
                print(f"  Username filled using: {username_selector}")
            else:
                print(f"  WARNING: Username selector not found: {username_selector}")
                print(f"  Trying fallback: input[type='text']")
                page.locator('input[type="text"]').first.fill(username)

            # Fill password
            if page.locator(password_selector).count() > 0:
                page.locator(password_selector).first.fill(password)
                print(f"  Password filled using: {password_selector}")
            else:
                print(f"  WARNING: Password selector not found: {password_selector}")
                print(f"  Trying fallback: input[type='password']")
                page.locator('input[type="password"]').first.fill(password)

            # Take screenshot after filling
            page.screenshot(path='form_filled_cv.png')
            print("  Screenshot: form_filled_cv.png")

            # Click button
            if page.locator(button_selector).count() > 0:
                page.locator(button_selector).first.click()
                print(f"  Login button clicked using: {button_selector}")
            else:
                print(f"  WARNING: Button selector not found: {button_selector}")
                print(f"  Trying fallback: button[type='submit']")
                page.locator('button[type="submit"]').first.click()

        except Exception as e:
            print(f"  ERROR during CV-guided login: {e}")

    # Wait for navigation
    print("\nWaiting for login to complete...")
    page.wait_for_timeout(config['wait_times']['after_login'])

    # Take final screenshot
    page.screenshot(path='after_login_selector_test.png')
    print("Screenshot: after_login_selector_test.png")

    # Check result
    print("\n" + "=" * 80)
    print("LOGIN VERIFICATION")
    print("=" * 80)

    original_url = config['web_url']
    current_url = page.url

    print(f"\nOriginal URL: {original_url}")
    print(f"Current URL:  {current_url}")

    if current_url != original_url and 'login' not in current_url.lower():
        print("\n*** LOGIN SUCCESSFUL! ***")
        print("URL changed and no longer on login page")
    else:
        print("\n*** LOGIN FAILED ***")
        print("Still on login page")

    print("\nScreenshots saved:")
    print("  - login_page_selector_test.png (initial)")
    if standard_worked:
        print("  - form_filled_standard.png (after filling with standard selectors)")
    else:
        print("  - form_filled_cv.png (after filling with CV selectors)")
    print("  - after_login_selector_test.png (after login attempt)")

    print("\nKeeping browser open for inspection...")
    input("\nPress Enter to close browser...")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    context.close()
    browser.close()
    playwright.stop()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
