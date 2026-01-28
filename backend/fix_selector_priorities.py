"""
Fix selector priorities and add context keywords.

Problem: Runtime extraction assigned priorities that are too high (50-145),
causing wrong selectors to match due to priority dominating keyword scoring.

Solution:
1. Lower priorities to 5-20 range (learned) and 3-15 range (extracted)
2. Add context keywords from step_text to improve semantic matching
"""

import json
import re
from typing import List, Dict, Any


def extract_context_keywords(step_text: str, text_content: str = "") -> List[str]:
    """
    Extract context keywords from step text and element text.

    These keywords help with semantic matching in selector_loader_v2.
    """
    keywords = set()

    step_lower = step_text.lower()
    text_lower = text_content.lower()

    # Extract action words
    actions = ['click', 'select', 'enter', 'type', 'navigate', 'open', 'close', 'save', 'edit', 'delete']
    for action in actions:
        if action in step_lower:
            keywords.add(action)

    # Extract target words
    targets = ['button', 'input', 'field', 'dropdown', 'menu', 'link', 'accordion', 'panel',
               'teststep', 'test', 'part', 'project', 'name', 'type', 'description']
    for target in targets:
        if target in step_lower or target in text_lower:
            keywords.add(target)

    # Extract specific UI elements from step text
    # "navigate to teststep" -> ["navigate", "teststep"]
    words = re.findall(r'\b[a-z]+\b', step_lower)
    meaningful_words = [w for w in words if len(w) > 3 and w not in ['click', 'from', 'then', 'with', 'should']]
    keywords.update(meaningful_words[:5])  # Limit to 5 most relevant

    # Add text content keywords (short, descriptive)
    if text_content:
        text_words = re.findall(r'\b[A-Za-z]+\b', text_content)
        # Only short, single words (like "Save", "Edit", "Runs")
        text_keywords = [w.lower() for w in text_words if 3 <= len(w) <= 10]
        keywords.update(text_keywords[:3])  # Max 3 from text

    return sorted(list(keywords))


def normalize_priority(priority: int, is_learned: bool, selector: Dict[str, Any]) -> int:
    """
    Normalize priority to appropriate range.

    New ranges:
    - Learned selectors (from L2): 10-20
    - Extracted selectors: 3-15

    Priority should guide, not dominate. Keyword matching should be primary.
    """
    if is_learned:
        # Learned selectors: 10-20 range
        # These are verified to work, so higher base
        return 15  # Fixed mid-high priority
    else:
        # Extracted selectors: 3-15 range based on qualities
        base = 5

        # Add boosts for good qualities
        if selector.get('isClickable'):
            base += 3
        if selector.get('isVisible'):
            base += 2
        if selector.get('tagName') in ['button', 'a', 'input']:
            base += 3
        if selector.get('containsTargetText'):
            base += 2

        return min(base, 15)  # Cap at 15


def fix_selectors(input_file: str, output_file: str):
    """Fix priorities and add context to selectors."""

    # Load merged selectors
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selectors = data['selectors']

    print(f"Processing {len(selectors)} selectors...")

    fixed_count = 0
    context_added = 0

    for selector in selectors:
        # Check if this is a runtime selector
        is_runtime = selector.get('source') in ['runtime_only', 'runtime_verified']
        is_learned = selector.get('learned_from') is not None

        if not is_runtime:
            continue  # Don't modify source code selectors

        # Fix priority
        old_priority = selector.get('priority', 0)
        if old_priority >= 50:  # Only fix if too high
            new_priority = normalize_priority(old_priority, is_learned, selector)
            selector['priority'] = new_priority
            fixed_count += 1

        # Add context keywords if missing
        if 'context' not in selector or not selector['context']:
            step_text = selector.get('step_text', '')
            text_content = selector.get('textContent', '')

            if step_text or text_content:
                context = extract_context_keywords(step_text, text_content)
                selector['context'] = context
                context_added += 1

    # Update metadata
    data['metadata']['version'] = 'phase4_fixed_priorities'
    data['metadata']['lastModified'] = '2025-11-04T13:00:00'
    data['metadata']['fixApplied'] = 'Normalized priorities (5-20 range) and added context keywords'

    # Save fixed selectors
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFixed {fixed_count} priorities")
    print(f"Added context to {context_added} selectors")
    print(f"\nOutput: {output_file}")

    # Show example fixes
    print("\nExample fixes:")
    examples = [s for s in selectors if s.get('learned_from')][:3]
    for i, sel in enumerate(examples, 1):
        print(f"\n{i}. [{sel['attr']}=\"{sel['value']}\"]")
        print(f"   Step: {sel.get('step_text', 'N/A')[:50]}")
        print(f"   Priority: {sel.get('priority', 0)}")
        print(f"   Context: {sel.get('context', [])[:5]}")


if __name__ == '__main__':
    fix_selectors(
        'Selectors_Folder/selectors_merged_runtime.json',
        'Selectors_Folder/selectors_merged_runtime_fixed.json'
    )
