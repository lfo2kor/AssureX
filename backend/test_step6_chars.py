"""Quick test to check Step 6 character encoding from file"""
import re

# Read the Jira ticket file with UTF-8
with open("Jira_Tickets/RBPLCD-8835.txt", 'r', encoding='utf-8') as f:
    content = f.read()

# Find Step 6 line
lines = content.split('\n')
step_6_line = None
for line in lines:
    if line.strip().startswith('6.'):
        step_6_line = line.strip()
        break

print("="*70)
print("Step 6 from file (UTF-8)")
print("="*70)
print(f"Line: {step_6_line}")
print(f"Repr: {repr(step_6_line)}")
print()

# Extract just the step text (remove "6. ")
step_text = step_6_line[3:].strip() if step_6_line else ""

print("Step text (without number):")
print(f"Text: {step_text}")
print(f"Repr: {repr(step_text)}")
print()

# Find the quotes
print("Quote characters:")
for i, char in enumerate(step_text):
    if ord(char) > 127:  # Non-ASCII
        print(f"  Position {i}: '{char}' = U+{ord(char):04X}")
print()

# Test the OLD regex
pattern_old = r'select\s+.*?[""\']([^""\']+)[""\']'
match_old = re.search(pattern_old, step_text, re.IGNORECASE)

print("OLD Pattern Test:")
if match_old:
    print(f"  [OK] Matched: '{match_old.group(1)}'")
else:
    print(f"  [FAIL] Did not match")
    print(f"  Pattern: {repr(pattern_old)}")
print()

# Test the NEW regex
pattern = r'select\s+.*?["\'\u201c\u201d]([^"\'\u201c\u201d]+)["\'\u201c\u201d]'
match = re.search(pattern, step_text, re.IGNORECASE)

print("NEW Pattern Test:")

if match:
    print(f"[OK] Regex MATCHED!")
    print(f"Extracted value: '{match.group(1)}'")
else:
    print(f"[FAIL] Regex DID NOT MATCH")
    print(f"Pattern used: {repr(pattern)}")
