"""
Analyze JSON fields to determine which are useful for embeddings
"""
import json
from pathlib import Path

# Load the JSON file
json_path = Path("Selectors_Folder/selectors_merged_runtime_fixed.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    selectors = data.get('selectors', data)

print("=" * 80)
print("JSON Field Analysis for Embedding Optimization")
print("=" * 80)
print()

# Get all unique fields
all_fields = set()
field_examples = {}
field_frequency = {}
field_null_count = {}
field_empty_count = {}

for selector in selectors:
    for field, value in selector.items():
        all_fields.add(field)

        # Count frequency
        if field not in field_frequency:
            field_frequency[field] = 0
            field_null_count[field] = 0
            field_empty_count[field] = 0

        field_frequency[field] += 1

        # Track nulls and empties
        if value is None:
            field_null_count[field] += 1
        elif value == "" or value == []:
            field_empty_count[field] += 1

        # Store example
        if field not in field_examples and value not in [None, "", []]:
            field_examples[field] = value

total_selectors = len(selectors)

print(f"Total Selectors: {total_selectors}")
print(f"Total Unique Fields: {len(all_fields)}")
print()

# Categorize fields
print("=" * 80)
print("FIELD CATEGORIZATION")
print("=" * 80)
print()

# Category 1: High-value semantic fields (should include in embedding)
high_value = {
    'step_text': 'The actual test step description - CRITICAL for matching',
    'textContent': 'Button/link text visible to users - CRITICAL',
    'module': 'Which module/page - Important for context',
    'context': 'Keywords describing when to use - Important',
    'attr': 'Data attribute name (data-test, data-date) - Useful',
    'value': 'Data attribute value - Useful',
    'ariaLabel': 'Accessibility label - Can be useful for matching',
    'role': 'Element role (button, link) - Useful for action type',
    'action': 'What action to perform (click, type) - Useful'
}

# Category 2: Low-value technical fields (might remove)
low_value = {
    'tagName': 'HTML tag - Less useful for semantic search',
    'className': 'CSS classes - Usually very long and technical',
    'id': 'Element ID - Technical, but sometimes useful',
    'isVisible': 'Boolean - Not useful for text embedding',
    'isClickable': 'Boolean - Not useful for text embedding',
    'width': 'Number - Not useful for text embedding',
    'height': 'Number - Not useful for text embedding',
    'step_num': 'Number - Not useful for text embedding',
    'extractedDate': 'Timestamp - Not useful for semantic search',
    'extractionMode': 'Technical metadata - Not useful',
    'pageUrl': 'Full URL - Too specific, use module instead',
    'learned_from': 'Old selector - Technical, not semantic',
    'priority': 'Number - Not useful for text embedding',
    'allDataAttrs': 'Duplicate of attr/value - Redundant',
    'source': 'Technical metadata - Not useful',
    'runtimeVerified': 'Boolean - Not useful',
    'runtimeExtractedDate': 'Timestamp - Not useful'
}

# Category 3: Maybe useful (context-dependent)
maybe_useful = {
    'elementType': 'button, input, etc - Useful for action matching',
    'label': 'Field label - Useful if present',
    'isDynamic': 'Boolean - Could indicate if selector needs context'
}

print("HIGH-VALUE FIELDS (Should Include in Embedding):")
print("-" * 80)
for field, reason in high_value.items():
    freq = field_frequency.get(field, 0)
    null_count = field_null_count.get(field, 0)
    empty_count = field_empty_count.get(field, 0)
    populated = freq - null_count - empty_count
    percent = (populated / total_selectors * 100) if freq > 0 else 0
    example = field_examples.get(field, 'N/A')
    if isinstance(example, list):
        example = ', '.join(str(e) for e in example[:3])
    print(f"\n{field}:")
    print(f"  Reason: {reason}")
    print(f"  Populated: {populated}/{total_selectors} ({percent:.1f}%)")
    print(f"  Example: {example}")

print("\n" + "=" * 80)
print("\nLOW-VALUE FIELDS (Consider Removing):")
print("-" * 80)
for field, reason in low_value.items():
    freq = field_frequency.get(field, 0)
    null_count = field_null_count.get(field, 0)
    empty_count = field_empty_count.get(field, 0)
    populated = freq - null_count - empty_count
    percent = (populated / total_selectors * 100) if freq > 0 else 0
    example = field_examples.get(field, 'N/A')
    if isinstance(example, str) and len(example) > 50:
        example = example[:50] + "..."
    print(f"\n{field}:")
    print(f"  Reason: {reason}")
    print(f"  Populated: {populated}/{total_selectors} ({percent:.1f}%)")

print("\n" + "=" * 80)
print("\nMAYBE USEFUL FIELDS:")
print("-" * 80)
for field, reason in maybe_useful.items():
    freq = field_frequency.get(field, 0)
    null_count = field_null_count.get(field, 0)
    empty_count = field_empty_count.get(field, 0)
    populated = freq - null_count - empty_count
    percent = (populated / total_selectors * 100) if freq > 0 else 0
    example = field_examples.get(field, 'N/A')
    print(f"\n{field}:")
    print(f"  Reason: {reason}")
    print(f"  Populated: {populated}/{total_selectors} ({percent:.1f}%)")
    print(f"  Example: {example}")

print("\n" + "=" * 80)
print("\nRECOMMENDED EMBEDDING FORMAT")
print("=" * 80)
print()

print("OPTIMAL FIELDS FOR EMBEDDING:")
print("1. step_text       - The actual step (highest value)")
print("2. textContent     - Visible text on button/link")
print("3. ariaLabel       - Accessibility label (if present)")
print("4. module          - Module/page context")
print("5. action          - What action to perform")
print("6. role            - Element role (button, link, input)")
print("7. attr            - Data attribute name")
print("8. value           - Data attribute value")
print("9. context         - Keywords")
print("10. label          - Field label (if present)")
print()

print("RECOMMENDED COMPOSITE FORMAT:")
print("-" * 80)
composite = "{step_text} | {textContent} {ariaLabel} | {action} {role} | {module} | {attr}={value} | {context}"
print(composite)
print()

print("Example output:")
example = "Navigate to teststep | Runs | click link | Teststep | data-test=sidebar-nav-item-nav_item_teststeps | item navigate runs sidebar test teststep"
print(example)
print()

print("FIELDS TO REMOVE FROM JSON:")
print("-" * 80)
remove_fields = [
    'className',           # Too long and technical
    'width', 'height',     # Numbers, not useful
    'isVisible', 'isClickable',  # Booleans
    'extractedDate', 'runtimeExtractedDate',  # Timestamps
    'extractionMode', 'source',  # Technical metadata
    'learned_from',        # Old selector, redundant
    'allDataAttrs',        # Duplicate of attr/value
    'runtimeVerified',     # Boolean
    'pageUrl',             # Too specific, use module instead
]

for field in remove_fields:
    print(f"- {field}")

print()
print("FIELDS TO KEEP IN JSON:")
print("-" * 80)
keep_fields = [
    'step_text',      # CRITICAL
    'textContent',    # CRITICAL
    'ariaLabel',      # Important
    'module',         # Important
    'action',         # Important
    'role',           # Useful
    'attr',           # Useful
    'value',          # Useful
    'context',        # Useful
    'label',          # Useful
    'elementType',    # Useful
    'id',             # Keep for indexing
    'step_num',       # Keep for ordering
    'priority',       # Keep for ranking
    'tagName',        # Keep for selector building
    'isDynamic',      # Keep for logic
]

for field in keep_fields:
    print(f"- {field}")

print()
print("=" * 80)
