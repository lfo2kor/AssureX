"""
Manually add the correct Type dropdown selector to the JSON file.
"""
import json

# Load current selectors
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# The correct selector that L2 uses
new_selector = {
    "attr": "data-attribute",
    "value": "Type",
    "tagName": "input",
    "className": "mat-mdc-input-element mat-mdc-form-field-input-control mdc-text-field__input mat-mdc-autocomplete-trigger mat-autocomplete-trigger",
    "module": "DetailView",
    "context": ["type", "dropdown", "autocomplete", "select", "input", "field"],
    "priority": 25,  # Higher than data-model (12)
    "isClickable": True,
    "isVisible": True,
    "role": "combobox",
    "source": "manual_fix_step6",
    "label": "Type input field for dropdown selection",
    "textContent": "",
    "id": None
}

# Check if already exists
existing = [s for s in data['selectors']
            if s.get('attr') == 'data-attribute'
            and s.get('value') == 'Type'
            and s.get('module') == 'DetailView']

if existing:
    # Update existing
    idx = data['selectors'].index(existing[0])
    data['selectors'][idx] = new_selector
    print(f'Updated existing selector at index {idx}')
else:
    # Add new
    data['selectors'].append(new_selector)
    print('Added new selector')

# Save
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f'\nTotal selectors: {len(data["selectors"])}')
print('\nNew selector added:')
print(f'  [{new_selector["attr"]}="{new_selector["value"]}"]')
print(f'  Module: {new_selector["module"]}')
print(f'  Priority: {new_selector["priority"]}')
print(f'  Context: {new_selector["context"]}')
print(f'  TagName: {new_selector["tagName"]}')
print('\nL1 should now use this selector for Step 6!')
