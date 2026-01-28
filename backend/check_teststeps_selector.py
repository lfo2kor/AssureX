import json

with open('Selectors_Folder/selectors_merged_runtime_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

teststeps = [s for s in data['selectors'] if 'teststeps' in s.get('value', '')]

print(f'Found {len(teststeps)} teststeps selectors\n')

for i, sel in enumerate(teststeps):
    print(f'{i+1}. [{sel["attr"]}="{sel["value"]}"]')
    print(f'   Priority: {sel.get("priority", 0)}')
    print(f'   Context: {sel.get("context", [])}')
    print(f'   TextContent: {sel.get("textContent", "N/A")[:30]}')
    print(f'   Learned: {bool(sel.get("learned_from"))}')
    print(f'   Source: {sel.get("source", "N/A")}')
    print()
