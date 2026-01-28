"""
Simple test to check if web application is accessible and capture login page
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.config_loader import load_config
from playwright.sync_api import sync_playwright
import time

print("=" * 80)
print("WEB APPLICATION ACCESS TEST")
print("=" * 80)

# Load config
config = load_config("plcdtest_config.yaml")
print(f"\nTarget URL: {config['web_url']}")

# Launch browser
print("\nLaunching browser...")
playwright = sync_playwright().start()
browser = playwright.chromium.launch(channel='msedge', headless=False)
context = browser.new_context(viewport={'width': 1920, 'height': 1080})
page = context.new_page()

try:
    # Navigate
    print(f"Navigating to: {config['web_url']}")
    page.goto(config['web_url'], wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(3000)

    # Take screenshot
    page.screenshot(path='current_login_page.png')
    print("\nScreenshot saved: current_login_page.png")

    # Get page title
    title = page.title()
    print(f"Page title: {title}")

    # Get current URL
    url = page.url
    print(f"Current URL: {url}")

    # Try to find login form elements using selectors
    print("\nLooking for login form elements...")

    # Check for username field
    username_selectors = [
        'input[type="text"]',
        'input[name="username"]',
        'input[id="username"]',
        'input[placeholder*="user" i]',
        'input[placeholder*="name" i]',
    ]

    for selector in username_selectors:
        count = page.locator(selector).count()
        if count > 0:
            print(f"  Found username field: {selector} (count: {count})")

    # Check for password field
    password_selectors = [
        'input[type="password"]',
        'input[name="password"]',
        'input[id="password"]',
    ]

    for selector in password_selectors:
        count = page.locator(selector).count()
        if count > 0:
            print(f"  Found password field: {selector} (count: {count})")

    # Check for button
    button_selectors = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button',
    ]

    for selector in button_selectors:
        count = page.locator(selector).count()
        if count > 0:
            print(f"  Found button: {selector} (count: {count})")
            # Try to get button text
            if count > 0:
                try:
                    text = page.locator(selector).first.inner_text()
                    print(f"    Button text: {text}")
                except:
                    pass

    # Try to fill the form using selectors
    print("\nAttempting to fill login form using selectors...")

    try:
        # Fill username
        username_filled = False
        for selector in username_selectors:
            if page.locator(selector).count() > 0:
                page.locator(selector).first.fill(config['login']['username'])
                print(f"  Username filled using: {selector}")
                username_filled = True
                break

        # Fill password
        password_filled = False
        for selector in password_selectors:
            if page.locator(selector).count() > 0:
                page.locator(selector).first.fill(config['login']['password'])
                print(f"  Password filled using: {selector}")
                password_filled = True
                break

        if username_filled and password_filled:
            page.wait_for_timeout(1000)
            page.screenshot(path='form_filled.png')
            print("\nScreenshot after filling form: form_filled.png")

            # Try to click submit button
            print("\nAttempting to click login button...")
            for selector in button_selectors:
                if page.locator(selector).count() > 0:
                    page.locator(selector).first.click()
                    print(f"  Clicked button: {selector}")
                    break

            # Wait for navigation
            page.wait_for_timeout(5000)

            # Check if URL changed
            new_url = page.url
            print(f"\nURL after login attempt: {new_url}")

            if new_url != url:
                print("SUCCESS: Page navigated after login!")
            else:
                print("FAILED: Still on same page")

            page.screenshot(path='after_login_attempt.png')
            print("Screenshot after login: after_login_attempt.png")

    except Exception as e:
        print(f"\nError during form fill: {e}")

    print("\nKeeping browser open for manual inspection...")
    print("Check the screenshots and browser window.")
    input("\nPress Enter to close browser...")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    context.close()
    browser.close()
    playwright.stop()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
