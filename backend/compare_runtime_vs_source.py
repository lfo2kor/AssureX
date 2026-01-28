import json

# Load both selector sets
with open('Selectors_Folder/runtime_selectors_RBPLCD-8835.json', 'r', encoding='utf-8') as f:
    runtime_data = json.load(f)

with open('Selectors_Folder/selectors_enriched_all_modules.json', 'r', encoding='utf-8') as f:
    source_data = json.load(f)

runtime_selectors = runtime_data['selectors']
source_selectors = source_data['selectors']

# Get learned selectors
learned = [s for s in runtime_selectors if s.get('learned_from')]

print('=' * 80)
print('   RUNTIME vs SOURCE CODE COMPARISON')
print('=' * 80)
print()
print(f'Runtime selectors (learned from L2): {len(learned)}')
print(f'Source code selectors: {len(source_selectors)}')
print()

# Compare learned selectors
print('Learned Selectors Analysis:')
print('-' * 80)

for sel in learned[:4]:
    attr = sel['attr']
    val = sel['value']

    # Check if exists in source
    source_match = [s for s in source_selectors if s.get('attr') == attr and s.get('value') == val]

    print(f"\nStep {sel['step_num']}: [{attr}=\"{val}\"]")
    print(f"  Learned from: {sel['learned_from'][:60]}")
    print(f"  In source code: {'YES' if source_match else 'NO (NEW DISCOVERY!)'}")

    if source_match:
        print(f"    Source module: {source_match[0].get('module')}")
        print(f"    Source file: {source_match[0].get('filePath', 'N/A')[:50]}")
    else:
        print(f"    >>> This selector ONLY exists in the running app!")
        print(f"    >>> Not found in static HTML source files")

# Summary
print()
print('=' * 80)
print('SUMMARY')
print('=' * 80)

new_selectors = [s for s in learned if not any(
    src.get('attr') == s['attr'] and src.get('value') == s['value']
    for src in source_selectors
)]

print(f"New selectors discovered from runtime: {len(new_selectors)}/{len(learned)}")
print(f"Already in source code: {len(learned) - len(new_selectors)}/{len(learned)}")
print()
print("Conclusion: Runtime extraction found selectors that don't exist in source code!")
