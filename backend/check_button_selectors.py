"""
Check why button[type='submit'] selector was not found
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright
from utils.config_loader import load_config

config = load_config('plcdtest_config.yaml')

playwright = sync_playwright().start()
browser = playwright.chromium.launch(channel='msedge', headless=False)
context = browser.new_context(viewport={'width': 1920, 'height': 1080})
page = context.new_page()

page.goto(config['web_url'], wait_until='networkidle', timeout=30000)
page.wait_for_timeout(3000)

print('Checking different button selectors on login page:')
print('=' * 60)

# Check various button selectors
selectors = [
    'button[type="submit"]',
    'button[type="button"]',
    'button',
    'input[type="submit"]',
    'button:has-text("Login")',
]

for selector in selectors:
    count = page.locator(selector).count()
    print(f'{selector:40} -> Count: {count}')
    if count > 0:
        try:
            elem = page.locator(selector).first
            text = elem.text_content() or ''
            type_attr = elem.get_attribute('type') or 'none'
            print(f'    Text: "{text.strip()}", Type attribute: "{type_attr}"')
        except Exception as e:
            print(f'    Error: {e}')

# Get the HTML of the login button
print('\n' + '=' * 60)
print('Login button HTML:')
try:
    button = page.locator('button:has-text("Login")').first
    outer_html = button.evaluate('el => el.outerHTML')
    print(outer_html[:500])
except Exception as e:
    print(f'Error: {e}')

print('\n' + '=' * 60)
print('REASON WHY button[type="submit"] WAS NOT FOUND:')
print('=' * 60)

# Check the actual type attribute
try:
    button = page.locator('button:has-text("Login")').first
    type_value = button.get_attribute('type')
    print(f'\nLogin button type attribute value: "{type_value}"')

    if type_value == 'submit':
        print('Result: Button HAS type="submit" - should have been found!')
    elif type_value == 'button':
        print('Result: Button has type="button" (NOT "submit")')
        print('This is why button[type="submit"] returned 0 count!')
    elif type_value is None:
        print('Result: Button has NO type attribute')
        print('Default button type is "submit", but selector requires explicit attribute')
    else:
        print(f'Result: Button has type="{type_value}"')

except Exception as e:
    print(f'Error: {e}')

input('\nPress Enter to close...')

context.close()
browser.close()
playwright.stop()
