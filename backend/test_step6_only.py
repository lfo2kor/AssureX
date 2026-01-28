"""Quick test to check Step 6 character encoding"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.jira_parser_agent import JiraParserAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Parse the Jira ticket
parser = JiraParserAgent()
result = parser.parse_ticket("RBPLCD-8835")

# Get Step 6
step_6 = result['steps'][5]  # 0-indexed, so step 6 is index 5

print("="*70)
print("Step 6 Analysis")
print("="*70)
print(f"Step number: {step_6['num']}")
print(f"Step text: {step_6['text']}")
print(f"Step text (repr): {repr(step_6['text'])}")
print()
print("Character codes around 'select':")
step_text = step_6['text']
select_pos = step_text.lower().find('select')
if select_pos >= 0:
    substring = step_text[select_pos:select_pos+40]
    print(f"Substring: {substring}")
    print(f"Char codes: {[f'{c}({ord(c):04X})' for c in substring]}")
print()

# Test the regex
import re
pattern = r'select\s+.*?[""\']([^""\']+)[""\']'
match = re.search(pattern, step_text, re.IGNORECASE)

if match:
    print(f"[OK] Regex MATCHED!")
    print(f"Extracted value: '{match.group(1)}'")
else:
    print(f"[FAIL] Regex DID NOT MATCH")
    print(f"Pattern: {pattern}")
