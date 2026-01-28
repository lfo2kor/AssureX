"""
Basic Vision Test - Prove GPT-4o Can See Screenshots

This test verifies GPT-4o can analyze a screenshot and identify elements.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.vision_helper import AzureVisionClient
from playwright.sync_api import sync_playwright

print("=" * 80)
print("BASIC VISION TEST - Can GPT-4o See Screenshots?")
print("=" * 80)

# Load config
config = load_config("plcdtest_config.yaml")
logger = setup_logger(config['folders']['logs'], "VISION-TEST")

# Initialize vision client
vision_client = AzureVisionClient(config, logger)
print("✓ Vision client initialized")

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

# Take screenshot
print("\n📸 Taking screenshot...")
screenshot = page.screenshot(path='vision_test.png')
print("✓ Screenshot saved: vision_test.png")

# Test 1: Ask GPT-4o to describe what it sees
print("\n" + "=" * 80)
print("TEST 1: What does GPT-4o see in this screenshot?")
print("=" * 80)

prompt1 = """
Describe what you see in this screenshot.

What type of page is this? What elements are visible?
List the main UI elements you can see.

Be specific about:
- What text/labels are visible
- What input fields or buttons are present
- The general layout
"""

print("\n🤖 Asking GPT-4o: 'What do you see?'")
result1 = vision_client.call_vision(screenshot, prompt1)
print("\n📊 GPT-4o Response:")
print(result1.get('raw_response', result1))

# Test 2: Ask GPT-4o to count elements
print("\n" + "=" * 80)
print("TEST 2: Can GPT-4o count input fields?")
print("=" * 80)

prompt2 = """
How many input fields (text boxes) are visible on this page?
How many buttons are visible?

Return your answer as JSON:
{
    "input_fields_count": <number>,
    "buttons_count": <number>,
    "description": "brief description of what you see"
}
"""

print("\n🤖 Asking GPT-4o: 'Count the elements'")
result2 = vision_client.call_vision(screenshot, prompt2)
print("\n📊 GPT-4o Response:")
if 'input_fields_count' in result2:
    print(f"   Input fields: {result2.get('input_fields_count', 'unknown')}")
    print(f"   Buttons: {result2.get('buttons_count', 'unknown')}")
    print(f"   Description: {result2.get('description', 'N/A')}")
else:
    print(result2)

# Test 3: Ask for coordinates
print("\n" + "=" * 80)
print("TEST 3: Can GPT-4o find coordinates?")
print("=" * 80)

prompt3 = """
Look at this page and identify the position of the FIRST input field (text box) you see.

Return JSON (no markdown):
{
    "element_found": "description of the element",
    "approximate_x": <X coordinate in pixels>,
    "approximate_y": <Y coordinate in pixels>,
    "confidence": <0.0 to 1.0>
}
"""

print("\n🤖 Asking GPT-4o: 'Where is the first input field?'")
result3 = vision_client.call_vision(screenshot, prompt3)
print("\n📊 GPT-4o Response:")
if 'approximate_x' in result3:
    print(f"   Element: {result3.get('element_found', 'unknown')}")
    print(f"   Coordinates: X={result3.get('approximate_x', '?')}, Y={result3.get('approximate_y', '?')}")
    print(f"   Confidence: {result3.get('confidence', 0)}")

    # Draw a red circle on the screenshot at those coordinates
    print("\n📍 Marking the coordinate on screenshot...")
    x = result3.get('approximate_x', 960)
    y = result3.get('approximate_y', 540)

    # Inject JavaScript to draw a marker
    page.evaluate(f"""
        (x, y) => {{
            const div = document.createElement('div');
            div.style.position = 'absolute';
            div.style.left = (x - 20) + 'px';
            div.style.top = (y - 20) + 'px';
            div.style.width = '40px';
            div.style.height = '40px';
            div.style.border = '5px solid red';
            div.style.borderRadius = '50%';
            div.style.zIndex = '9999';
            div.style.pointerEvents = 'none';
            document.body.appendChild(div);
        }}
    """, x, y)

    page.wait_for_timeout(500)
    page.screenshot(path='vision_test_marked.png')
    print("✓ Screenshot with red circle saved: vision_test_marked.png")
    print(f"\n   👀 Check 'vision_test_marked.png' - Is the red circle on the input field?")
else:
    print(result3)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nIf GPT-4o correctly:")
print("1. Described what it sees (login page)")
print("2. Counted the input fields correctly")
print("3. Found coordinates and the red circle is on/near the input box")
print("\nThen GPT-4o vision IS WORKING! ✅")
print("\nScreenshots to check:")
print("  - vision_test.png (original)")
print("  - vision_test_marked.png (with red circle showing GPT-4o's detected position)")

input("\nPress Enter to close browser...")

# Cleanup
context.close()
browser.close()
playwright.stop()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
