"""Analyze selectors.json for RBPLCD-8835 test steps"""
import json

# Load selectors
with open(r'C:\Projects\AI_Chat\PLCD\TA_AI_Project\Selectors_Folder\selectors.json') as f:
    selectors = json.load(f)

print(f"Total selectors in file: {len(selectors)}")
print("\n" + "=" * 80)

# Keywords from RBPLCD-8835 steps
step_keywords = {
    "Step 2 - navigate to teststep": ["teststep", "navigate", "menu", "nav"],
    "Step 3 - click on teststep": ["teststep", "click", "row", "table"],
    "Step 4 - open parts accordion": ["parts", "accordion", "expand"],
    "Step 5 - click edit button": ["edit", "btn", "button"],
    "Step 6 - Type dropdown": ["type", "dropdown", "select"],
    "Step 7 - click save": ["save", "btn", "button"],
}

for step_name, keywords in step_keywords.items():
    print(f"\n{step_name}")
    print("-" * 80)

    found = []
    for selector in selectors:
        attr = selector.get('attr', '').lower()
        value = selector.get('value', '').lower()
        label = selector.get('label', '').lower()
        module = selector.get('module', '').lower()

        # Check if any keyword matches
        for kw in keywords:
            if kw in attr or kw in value or kw in label or kw in module:
                found.append(selector)
                break

    print(f"Found {len(found)} matching selectors:")
    for s in found[:8]:  # Show first 8
        module = s.get('module', 'N/A')
        print(f"  [{s['attr']}=\"{s['value']}\"] (module: {module})")

    if len(found) > 8:
        print(f"  ... and {len(found) - 8} more")

print("\n" + "=" * 80)
print("\nSpecific searches:")
print("-" * 80)

# Search for save button
save_btns = [s for s in selectors if 'save' in s.get('attr', '').lower() and 'btn' in s.get('attr', '').lower()]
print(f"\nSave buttons: {len(save_btns)}")
for s in save_btns[:5]:
    print(f"  {s['attr']}={s['value']} (module: {s.get('module', 'N/A')})")

# Search for edit buttons
edit_btns = [s for s in selectors if 'edit' in s.get('attr', '').lower() and 'btn' in s.get('attr', '').lower()]
print(f"\nEdit buttons: {len(edit_btns)}")
for s in edit_btns[:5]:
    print(f"  {s['attr']}={s['value']} (module: {s.get('module', 'N/A')})")

# Search for accordion
accordions = [s for s in selectors if 'accordion' in s.get('attr', '').lower() or 'accordion' in s.get('value', '').lower()]
print(f"\nAccordions: {len(accordions)}")
for s in accordions[:5]:
    print(f"  {s['attr']}={s['value']} (module: {s.get('module', 'N/A')})")
