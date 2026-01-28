import json

# Load runtime selectors
with open('Selectors_Folder/runtime_selectors_RBPLCD-8835.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=' * 80)
print('   RUNTIME EXTRACTION RESULTS')
print('=' * 80)
print()
print(f'Total selectors extracted: {data["metadata"]["totalSelectors"]}')
print(f'Total steps processed: {data["metadata"]["totalSteps"]}')
print()

# Count learned selectors
learned = [s for s in data['selectors'] if s.get('learned_from')]
print(f'Learned from L2 successes: {len(learned)} selectors')
print()

if learned:
    print('Learned Selectors:')
    print('-' * 80)
    for s in learned[:10]:
        print(f"Step {s['step_num']}: [{s['attr']}=\"{s['value']}\"]")
        print(f"  Learned from: {s['learned_from'][:70]}")
        print(f"  Clickable: {s['isClickable']}, Visible: {s['isVisible']}")
        print(f"  Tag: {s['tagName']}, Text: '{s['textContent'][:50]}'")
        print()

# Analyze by step
print('Selectors by Step:')
print('-' * 80)
from collections import Counter
steps = Counter(s['step_num'] for s in data['selectors'])
for step_num in sorted(steps.keys()):
    count = steps[step_num]
    print(f"  Step {step_num}: {count} selectors extracted")
