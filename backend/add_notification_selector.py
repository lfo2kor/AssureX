"""
Add success notification/snackbar selector for Step 8.
"""
import json

# Load current selectors
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Add success notification selector
# Material Angular uses mat-snack-bar-container with role="status" or "alert"
notification_selector = {
    "attr": "role",
    "value": "status",
    "tagName": "div",
    "className": "mat-mdc-snack-bar-container mdc-snackbar",
    "module": "Common",  # Success messages can appear in any module
    "context": ["message", "notification", "alert", "snackbar", "success", "toast", "display"],
    "priority": 30,  # High priority for verification steps
    "isClickable": False,
    "isVisible": True,
    "role": "status",
    "source": "manual_fix_step8",
    "label": "Success notification message",
    "textContent": "Successfully",
    "id": None
}

# Check if already exists
existing = [s for s in data['selectors']
            if s.get('attr') == 'role'
            and s.get('value') == 'status'
            and s.get('module') == 'Common']

if existing:
    idx = data['selectors'].index(existing[0])
    data['selectors'][idx] = notification_selector
    print(f'Updated existing notification selector at index {idx}')
else:
    data['selectors'].append(notification_selector)
    print('Added new notification selector')

# Save
with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f'\nTotal selectors: {len(data["selectors"])}')
print('\nNew selector added:')
print(f'  [{notification_selector["attr"]}="{notification_selector["value"]}"]')
print(f'  Module: {notification_selector["module"]}')
print(f'  Priority: {notification_selector["priority"]}')
print(f'  Context: {notification_selector["context"]}')
print(f'  Role: {notification_selector["role"]}')
print('\nThis will match Material Angular snackbar notifications!')
print('L1 should now work for Step 8 message verification!')
