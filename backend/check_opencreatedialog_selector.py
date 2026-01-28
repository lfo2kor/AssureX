"""
Check if data-opencreatedialogdropdown selector exists in JSON.
"""
import json

with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

selectors = data['selectors']

# Search for the specific selector
target_attr = 'data-opencreatedialogdropdown'
target_value = 'aeName.StructureLevel.name'

print('='*80)
print('SEARCHING FOR: data-opencreatedialogdropdown="aeName.StructureLevel.name"')
print('='*80)

# Exact match
exact = [s for s in selectors if s.get('attr') == target_attr and s.get('value') == target_value]

if exact:
    print('\n[SUCCESS] FOUND EXACT MATCH!')
    for i, sel in enumerate(exact, 1):
        print(f'\n{i}. [{sel["attr"]}="{sel["value"]}"]')
        print(f'   Module: {sel.get("module", "N/A")}')
        print(f'   Priority: {sel.get("priority", 0)}')
        print(f'   Context: {sel.get("context", [])}')
        print(f'   IsDynamic: {sel.get("isDynamic", sel.get("dynamic", False))}')
        print(f'   PossibleValues: {sel.get("possibleValues", [])}')
        print(f'   TagName: {sel.get("tagName", "N/A")}')
        print(f'   ClassName: {sel.get("className", "N/A")[:60]}')
        print(f'   Source: {sel.get("source", "N/A")}')
else:
    print('\n[FAILED] EXACT MATCH NOT FOUND!')

    # Search for similar selectors
    similar_attr = [s for s in selectors if 'opencreatedialog' in s.get('attr', '').lower()]

    if similar_attr:
        print(f'\nFound {len(similar_attr)} selectors with "opencreatedialog":')
        for i, s in enumerate(similar_attr[:5], 1):
            print(f'\n{i}. [{s["attr"]}="{s["value"]}"]')
            print(f'   Module: {s.get("module", "N/A")}')
            print(f'   Priority: {s.get("priority", 0)}')
    else:
        print('\nNo selectors with "opencreatedialog" found!')

# Also search for dropdown selectors
print('\n' + '='*80)
print('DROPDOWN SELECTORS IN JSON:')
print('='*80)

dropdown_selectors = [s for s in selectors if 'dropdown' in s.get('attr', '').lower()]
print(f'\nFound {len(dropdown_selectors)} selectors with "dropdown":')
for s in dropdown_selectors[:10]:
    print(f'  [{s["attr"]}="{s["value"]}"] module={s.get("module", "N/A")}')

print('\n' + '='*80)
print('RECOMMENDATION:')
print('='*80)

if not exact:
    print('''
The selector data-opencreatedialogdropdown="aeName.StructureLevel.name" is NOT in JSON!

SOLUTION:
1. Add it manually to the JSON file, OR
2. Extract it from runtime using extract_runtime_selectors.py

To add manually, use this format:
{
  "attr": "data-opencreatedialogdropdown",
  "value": "aeName.StructureLevel.name",
  "tagName": "mat-select",  // or actual tag
  "className": "",
  "module": "Teststep",
  "context": ["dropdown", "select", "project", "create"],
  "priority": 20,
  "isClickable": true,
  "isDynamic": false
}
''')
