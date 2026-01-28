"""
Diagnose Login Page - Find out WHY typing doesn't work

This script will:
1. Check if login is in an iframe
2. Try using Playwright selectors as backup
3. Check if JavaScript is blocking input
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.config_loader import load_config
from playwright.sync_api import sync_playwright
import json

print("=" * 80)
print("LOGIN PAGE DIAGNOSIS")
print("=" * 80)

# Load config
config = load_config("plcdtest_config.yaml")

# Launch browser
print("\n📱 Launching browser...")
playwright = sync_playwright().start()
browser = playwright.chromium.launch(channel='msedge', headless=False)
context = browser.new_context(viewport={'width': 1920, 'height': 1080})
page = context.new_page()

# Navigate to login page
print(f"🌐 Navigating to: {config['web_url']}")
page.goto(config['web_url'], wait_until='networkidle', timeout=30000)
page.wait_for_timeout(3000)

print("\n" + "=" * 80)
print("DIAGNOSIS 1: Check for iframes")
print("=" * 80)

# Check if there are iframes
frames = page.frames
print(f"Number of frames on page: {len(frames)}")
for i, frame in enumerate(frames):
    print(f"  Frame {i}: {frame.url}")

if len(frames) > 1:
    print("\n⚠️  FOUND IFRAMES! Login might be inside an iframe.")
    print("   This changes how we need to interact with elements.")

print("\n" + "=" * 80)
print("DIAGNOSIS 2: Find input fields using Playwright selectors")
print("=" * 80)

# Try to find input fields using common selectors
selectors_to_try = [
    'input[type="text"]',
    'input[type="username"]',
    'input[type="email"]',
    'input[name*="user"]',
    'input[name*="User"]',
    'input[id*="user"]',
    'input[id*="User"]',
    'input[placeholder*="user"]',
    'input[placeholder*="User"]',
    'input:not([type="password"]):not([type="hidden"])',
]

username_field = None
print("\n🔍 Looking for username field...")
for selector in selectors_to_try:
    try:
        elements = page.locator(selector).all()
        if elements:
            print(f"  ✓ Found {len(elements)} element(s) with selector: {selector}")
            if not username_field:
                username_field = page.locator(selector).first
    except:
        pass

password_selectors = [
    'input[type="password"]',
]

password_field = None
print("\n🔍 Looking for password field...")
for selector in password_selectors:
    try:
        elements = page.locator(selector).all()
        if elements:
            print(f"  ✓ Found {len(elements)} element(s) with selector: {selector}")
            if not password_field:
                password_field = page.locator(selector).first
    except:
        pass

# Try to get all input fields
all_inputs = page.locator('input').all()
print(f"\n📊 Total input fields found: {len(all_inputs)}")

if len(all_inputs) > 0:
    print("\nInput field details:")
    for i, inp in enumerate(all_inputs[:10]):  # Show first 10
        try:
            attrs = page.evaluate("""
                (element) => {
                    return {
                        type: element.type,
                        name: element.name,
                        id: element.id,
                        placeholder: element.placeholder,
                        visible: element.offsetParent !== null
                    };
                }
            """, inp.element_handle())
            print(f"  Input {i}: type={attrs['type']}, name={attrs.get('name', 'N/A')}, id={attrs.get('id', 'N/A')}, visible={attrs['visible']}")
        except:
            print(f"  Input {i}: Could not read attributes")

print("\n" + "=" * 80)
print("DIAGNOSIS 3: Try using Playwright's fill() method")
print("=" * 80)

if username_field and password_field:
    print("\n✅ Found username and password fields using selectors!")
    print("   Attempting to fill using Playwright's fill() method...")

    try:
        # Method 1: Using fill()
        print("\n   Trying Method 1: fill()")
        username_field.fill(config['login']['username'])
        print(f"   ✓ Filled username: {config['login']['username']}")

        password_field.fill(config['login']['password'])
        print(f"   ✓ Filled password: {'*' * len(config['login']['password'])}")

        page.screenshot(path='diagnosis_after_fill.png')
        print("   ✓ Screenshot: diagnosis_after_fill.png")

        # Try to find and click login button
        button_selectors = [
            'button[type="submit"]',
            'button:has-text("Login")',
            'button:has-text("Sign in")',
            'input[type="submit"]',
            'button',
        ]

        login_button = None
        for selector in button_selectors:
            try:
                btn = page.locator(selector).first
                if btn:
                    print(f"   ✓ Found button with selector: {selector}")
                    login_button = btn
                    break
            except:
                pass

        if login_button:
            print("   Clicking login button...")
            login_button.click()
            page.wait_for_timeout(3000)

            page.screenshot(path='diagnosis_after_login.png')
            print("   ✓ Screenshot: diagnosis_after_login.png")

            current_url = page.url
            print(f"\n   Current URL: {current_url}")

            if current_url != config['web_url']:
                print("   ✅ SUCCESS! Login worked using Playwright selectors!")
                print("\n   💡 SOLUTION: We should use Playwright's locator.fill() instead of coordinates")
            else:
                print("   ⚠️  Still on login page")

    except Exception as e:
        print(f"   ❌ Error during fill: {e}")

else:
    print("\n❌ Could not find username/password fields using selectors")

print("\n" + "=" * 80)
print("DIAGNOSIS 4: Page source analysis")
print("=" * 80)

# Save page HTML
html = page.content()
with open('login_page_source.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("✓ Page HTML saved to: login_page_source.html")

# Check for common login frameworks
if 'angular' in html.lower():
    print("⚠️  Detected: Angular framework")
if 'react' in html.lower():
    print("⚠️  Detected: React framework")
if 'vue' in html.lower():
    print("⚠️  Detected: Vue framework")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if len(frames) > 1:
    print("\n1. ⚠️  Login is in an IFRAME - we need to switch frames first")
    print("   Fix: Use page.frame_locator() to access iframe content")

if username_field and password_field:
    print("\n2. ✅ Playwright selectors CAN find the fields")
    print("   Fix: Use locator.fill() instead of coordinates + keyboard.type()")
    print("   This is MORE RELIABLE than vision + coordinates")

print("\n3. 💡 HYBRID APPROACH:")
print("   - Use GPT-4o vision to VERIFY we're on login page")
print("   - Use Playwright selectors to FILL the form (more reliable)")
print("   - Best of both worlds!")

input("\nPress Enter to close browser...")

# Cleanup
context.close()
browser.close()
playwright.stop()

print("\n" + "=" * 80)
print("Diagnosis complete!")
print("\nFiles created:")
print("  - diagnosis_after_fill.png (after filling with selectors)")
print("  - diagnosis_after_login.png (after clicking login)")
print("  - login_page_source.html (full HTML source)")
print("=" * 80)
