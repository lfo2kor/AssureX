"""
Fix selector priorities V2 - Better logic to handle specific vs generic selectors.

Issues found:
1. data-test="sidebar-nav-item-undefined" has priority 130
2. data-test="sidebar-nav-item-nav_item_teststeps" has priority 100
3. Generic wins over specific!
4. Context keywords not being added properly

Solution:
1. LOWER priority for generic/undefined selectors (set to 3)
2. KEEP priority for specific selectors (10-15)
3. ADD context keywords from textContent and step_text
4. FILTER OUT completely useless selectors (data-test="undefined", empty values, etc.)
"""

import json
import re
from typing import List, Dict, Any


def is_generic_selector(selector: Dict[str, Any]) -> bool:
    """
    Check if selector is too generic to be useful.

    Generic selectors include:
    - data-test="sidebar-nav-item-undefined"
    - data-test="undefined"
    - Any attribute with value "undefined", "null", "", etc.
    """
    value = selector.get('value', '').lower()

    # Check for generic/useless values
    if value in ['undefined', 'null', '']:
        return True

    # Check for patterns like "sidebar-nav-item-undefined"
    if 'undefined' in value:
        return True

    # Check for empty or whitespace-only values
    if not value.strip():
        return True

    return False


def is_specific_selector(selector: Dict[str, Any]) -> bool:
    """
    Check if selector is specific and valuable.

    Specific selectors include:
    - Has textContent that's meaningful (like "Runs", "Save", etc.)
    - Has specific IDs (nav_item_teststeps)
    - Was learned from L2 (learned_from field)
    """
    # Learned selectors are always valuable
    if selector.get('learned_from'):
        return True

    # Has meaningful text content
    text = selector.get('textContent', '').strip()
    if text and len(text) > 0 and text.lower() not in ['', ' ', '\n']:
        return True

    # Has specific value (not generic)
    value = selector.get('value', '')
    if 'btn' in value.lower() or 'icon' in value.lower():
        return True

    return False


def extract_context_keywords(selector: Dict[str, Any]) -> List[str]:
    """Extract meaningful context keywords from selector."""
    keywords = set()

    # From textContent
    text = selector.get('textContent', '').strip()
    if text:
        # Extract words from text (like "Runs", "Save", "Edit")
        words = re.findall(r'\b[A-Za-z]+\b', text)
        meaningful = [w.lower() for w in words if 3 <= len(w) <= 15]
        keywords.update(meaningful[:3])  # Max 3 from text

    # From step_text
    step_text = selector.get('step_text', '').lower()
    if step_text:
        # Extract action words
        actions = ['click', 'select', 'enter', 'navigate', 'open', 'close', 'save', 'edit']
        for action in actions:
            if action in step_text:
                keywords.add(action)

        # Extract target words
        targets = ['button', 'input', 'dropdown', 'menu', 'link', 'teststep', 'test', 'project']
        for target in targets:
            if target in step_text:
                keywords.add(target)

    # From value (if it has meaningful parts)
    value = selector.get('value', '')
    if value and not is_generic_selector(selector):
        # Extract camelCase parts: "nav_item_teststeps" -> ["nav", "item", "teststeps"]
        parts = re.findall(r'[A-Za-z]+', value)
        meaningful = [p.lower() for p in parts if len(p) >= 4]
        keywords.update(meaningful[:2])  # Max 2 from value

    return sorted(list(keywords))


def fix_selectors(input_file: str, output_file: str):
    """Fix priorities and context with better logic."""

    # Load selectors
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selectors = data['selectors']
    print(f"Processing {len(selectors)} selectors...\n")

    removed_count = 0
    fixed_priority_count = 0
    added_context_count = 0

    filtered_selectors = []

    for selector in selectors:
        # Check if this is a runtime selector (any runtime source)
        source = selector.get('source', '')
        is_runtime = 'runtime' in source.lower() or selector.get('runtimeVerified') == True

        # FILTER OUT completely generic/useless selectors
        if is_runtime and is_generic_selector(selector):
            removed_count += 1
            continue  # Skip this selector entirely

        # FIX PRIORITY based on specificity
        if is_runtime:
            old_priority = selector.get('priority', 0)

            if is_specific_selector(selector):
                # Specific selector: keep reasonable priority (10-15)
                new_priority = 15 if selector.get('learned_from') else 12
            else:
                # Non-specific: lower priority
                new_priority = 5

            if old_priority != new_priority:
                selector['priority'] = new_priority
                fixed_priority_count += 1

        # ADD CONTEXT KEYWORDS
        if is_runtime and (not selector.get('context') or len(selector.get('context', [])) == 0):
            context = extract_context_keywords(selector)
            if context:
                selector['context'] = context
                added_context_count += 1

        filtered_selectors.append(selector)

    # Update data
    data['selectors'] = filtered_selectors
    data['metadata']['version'] = 'phase4_fixed_v2'
    data['metadata']['lastModified'] = '2025-11-04T16:00:00'
    data['metadata']['fixApplied'] = 'Removed generic selectors, fixed priorities, added context'

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Removed generic selectors: {removed_count}")
    print(f"Fixed priorities: {fixed_priority_count}")
    print(f"Added context: {added_context_count}")
    print(f"Total selectors after filtering: {len(filtered_selectors)}")
    print(f"\nOutput: {output_file}")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES OF FIXED SELECTORS:")
    print("="*80)

    # Show the teststeps selector
    teststeps = [s for s in filtered_selectors if 'teststeps' in s.get('value', '')]
    if teststeps:
        sel = teststeps[0]
        print(f"\n1. SPECIFIC SELECTOR (kept):")
        print(f"   [{sel['attr']}=\"{sel['value']}\"]")
        print(f"   Priority: {sel.get('priority', 0)}")
        print(f"   Context: {sel.get('context', [])}")
        print(f"   TextContent: {sel.get('textContent', 'N/A')[:30]}")
        print(f"   Learned: {bool(sel.get('learned_from'))}")

    # Show other learned selectors
    learned = [s for s in filtered_selectors if s.get('learned_from')][:2]
    for i, sel in enumerate(learned, 2):
        print(f"\n{i}. LEARNED SELECTOR:")
        print(f"   [{sel['attr']}=\"{sel['value']}\"]")
        print(f"   Priority: {sel.get('priority', 0)}")
        print(f"   Context: {sel.get('context', [])}")
        print(f"   From: {sel.get('learned_from', 'N/A')[:50]}")


if __name__ == '__main__':
    fix_selectors(
        'Selectors_Folder/selectors_merged_runtime.json',
        'Selectors_Folder/selectors_merged_runtime_fixed.json'
    )
