"""Test dropdown value extraction regex"""
import re

step_texts = [
    "Select Type 5 from dropdown menu",
    "Choose Type 5 in dropdown menu",
    "Select Type 5 in from the dropdown menu",
    "Pick Color Red from dropdown",
    "Choose Status Active in menu"
]

for step_text in step_texts:
    step_lower = step_text.lower()
    value_match = re.search(r'(?:select|choose|pick)\s+([^from]+?)\s+(?:from|in|dropdown)', step_lower)

    if value_match:
        value_to_select = value_match.group(1).strip()
        print(f"Step: {step_text}")
        print(f"  Extracted value: '{value_to_select}'")
    else:
        print(f"Step: {step_text}")
        print(f"  NO MATCH!")
    print()
