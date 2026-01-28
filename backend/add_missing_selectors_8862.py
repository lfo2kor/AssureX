"""
Add missing selectors for RBPLCD-8862.

These selectors were mentioned by the user but are NOT in JSON:
1. data-opencreatedialogdropdown="aeName.StructureLevel.name"
2. data-optionchange="optionChange"
3. data-dropdownentitiesname="MyProject"
4. data-checkuniquename="Name"
5. data-savebtn="SaveBtn" (already exists)
6. data-deletebtn="DeleteBtn"
7. data-test="alert-dialog-left-button"
"""
import json

# Load current selectors
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define missing selectors
missing_selectors = [
    {
        "attr": "data-opencreatedialogdropdown",
        "value": "aeName.StructureLevel.name",
        "tagName": "mat-select",
        "className": "mat-mdc-select",
        "module": "Teststep",
        "context": ["dropdown", "select", "project", "create", "dialog", "structurelevel"],
        "priority": 25,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix_rbplcd8862",
        "label": "Create dialog - Project dropdown"
    },
    {
        "attr": "data-optionchange",
        "value": "optionChange",
        "tagName": "mat-option",
        "className": "mat-mdc-option",
        "module": "CreateNew",
        "context": ["option", "select", "dropdown", "change", "project"],
        "priority": 25,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix_rbplcd8862",
        "label": "Dropdown option selection"
    },
    {
        "attr": "data-dropdownentitiesname",
        "value": "MyProject",
        "tagName": "mat-option",
        "className": "mat-mdc-option",
        "module": "SearchBar",
        "context": ["dropdown", "option", "project", "entities", "myproject"],
        "priority": 20,
        "isClickable": True,
        "isVisible": True,
        "isDynamic": True,
        "possibleValues": ["MyProject", "TestProject"],
        "source": "manual_fix_rbplcd8862",
        "label": "Project dropdown - MyProject option"
    },
    {
        "attr": "data-checkuniquename",
        "value": "Name",
        "tagName": "input",
        "className": "mat-mdc-input-element",
        "module": "CreateNew",
        "context": ["name", "input", "field", "unique", "check"],
        "priority": 25,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix_rbplcd8862",
        "label": "Name input field with uniqueness check"
    },
    {
        "attr": "data-deletebtn",
        "value": "DeleteBtn",
        "tagName": "button",
        "className": "button-warn",
        "module": "Common",
        "context": ["delete", "btn", "button", "remove"],
        "priority": 20,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix_rbplcd8862",
        "label": "Delete button"
    },
    {
        "attr": "data-test",
        "value": "alert-dialog-left-button",
        "tagName": "button",
        "className": "mat-mdc-button",
        "module": "Common",
        "context": ["alert", "dialog", "button", "confirm", "left"],
        "priority": 25,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix_rbplcd8862",
        "label": "Alert dialog left button (confirm/remove)"
    }
]

added_count = 0
updated_count = 0

for new_sel in missing_selectors:
    # Check if already exists
    existing = [s for s in data['selectors']
                if s.get('attr') == new_sel['attr']
                and s.get('value') == new_sel['value']
                and s.get('module') == new_sel['module']]

    if existing:
        # Update existing
        idx = data['selectors'].index(existing[0])
        data['selectors'][idx] = new_sel
        updated_count += 1
        print(f"Updated: [{new_sel['attr']}=\"{new_sel['value']}\"] in {new_sel['module']}")
    else:
        # Add new
        data['selectors'].append(new_sel)
        added_count += 1
        print(f"Added: [{new_sel['attr']}=\"{new_sel['value']}\"] for {new_sel['module']}")

# Save
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'Added: {added_count} new selectors')
print(f'Updated: {updated_count} existing selectors')
print(f'Total selectors: {len(data["selectors"])}')
print('\nSelectors added for RBPLCD-8862:')
for sel in missing_selectors:
    print(f'  - [{sel["attr"]}="{sel["value"]}"] ({sel["label"]})')

print('\nNow run: python run_test.py RBPLCD-8862')
print('Expected: More steps should pass with L1!')
