import re

# Exact step text from Jira ticket
step_text = '6. Click on Type from mandatory field and select "Type 5" from drop down'

print(f"Step text: {step_text}")
print(f"Step text (repr): {repr(step_text)}")
print()

# Current regex pattern
pattern = r'select\s+.*?[""\']([^""\']+)[""\']'

match = re.search(pattern, step_text, re.IGNORECASE)

if match:
    print(f"[OK] Regex MATCHED")
    print(f"Extracted value: {match.group(1)}")
else:
    print(f"[FAIL] Regex DID NOT MATCH")
    print()
    print("Let's check the quote characters:")
    # Find the quotes around "Type 5"
    for i, char in enumerate(step_text):
        if char in '""\'"\'':
            print(f"  Position {i}: '{char}' (Unicode: U+{ord(char):04X})")
