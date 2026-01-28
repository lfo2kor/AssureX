"""
Debug why Step 6 L1 failed - compare wrong vs correct selector.
"""
import json

with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

selectors = data['selectors']

# Find the WRONG selector that L1 chose
model = [s for s in selectors if s.get('attr') == 'data-model' and
         s.get('module', '').lower() in ['detailview', 'teststep']]

# Find the CORRECT selector that L2 used
attribute_type = [s for s in selectors if s.get('attr') == 'data-attribute' and
                  s.get('value') == 'Type']

print('='*80)
print('STEP 6 L1 FAILURE ANALYSIS')
print('='*80)

print('\n1. WRONG SELECTOR (what L1 chose): [data-model="model"]')
print('-'*80)
if model:
    s = model[0]
    print(f'Module: {s.get("module")}')
    print(f'Priority: {s.get("priority")}')
    print(f'Context: {s.get("context", [])}')
    print(f'TagName: {s.get("tagName")}')
    print(f'ClassName: {s.get("className", "N/A")[:60]}')
    print(f'TextContent: {s.get("textContent", "N/A")[:40]}')
    print(f'IsClickable: {s.get("isClickable")}')
else:
    print('NOT FOUND IN JSON!')

print('\n2. CORRECT SELECTOR (what L2 used): [data-attribute="Type"]')
print('-'*80)
if attribute_type:
    s = attribute_type[0]
    print(f'Module: {s.get("module")}')
    print(f'Priority: {s.get("priority")}')
    print(f'Context: {s.get("context", [])}')
    print(f'TagName: {s.get("tagName")}')
    print(f'ClassName: {s.get("className", "N/A")[:60]}')
    print(f'TextContent: {s.get("textContent", "N/A")[:40]}')
    print(f'IsClickable: {s.get("isClickable")}')
    print(f'\nL2 used FULL selector: input.mat-mdc-autocomplete-trigger[data-attribute="Type"]')
else:
    print('NOT FOUND IN JSON!')
    print('This is why L1 failed - the correct selector is missing!')

print('\n'+'='*80)
print('ROOT CAUSE:')
print('='*80)

if model and not attribute_type:
    print('❌ The correct selector [data-attribute="Type"] is NOT in JSON file')
    print('   L1 chose the only matching selector it could find: [data-model="model"]')
    print('   But this was the WRONG element!')

elif model and attribute_type:
    print('✅ Both selectors are in JSON file')
    print(f'   BUT [data-model="model"] scored HIGHER (priority={model[0].get("priority")})')
    print(f'   than [data-attribute="Type"] (priority={attribute_type[0].get("priority")})')
    print(f'   Context keywords also matter - comparing:')
    print(f'   - data-model context: {model[0].get("context", [])}')
    print(f'   - data-attribute context: {attribute_type[0].get("context", [])}')

print('\n'+'='*80)
print('SOLUTION:')
print('='*80)
print('Option 1: Add [data-attribute="Type"] to JSON with proper metadata')
print('          - Priority: 15-20 (higher than data-model)')
print('          - Context: ["type", "dropdown", "select", "attribute"]')
print('          - TagName: "input"')
print('          - ClassName: "mat-mdc-autocomplete-trigger"')
print('')
print('Option 2: Use build_selector() to create FULL selector in L1')
print('          - Instead of: [data-attribute="Type"]')
print('          - Use: input.mat-mdc-autocomplete-trigger[data-attribute="Type"]')
print('          - This makes it unique (count=1 instead of 2)')
