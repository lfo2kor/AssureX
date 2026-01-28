# How to Use Extracted Selectors with L1/L2/L3 System

## What You Have Now

**File:** `Selectors_Folder/phase2_dynamic_selectors.json`

**Statistics:**
- Total selectors: 884
- Static selectors: 595 (67.3%)
- Dynamic selectors: 289 (32.7%)
  - 7 dynamic selectors have extracted values
  - 282 dynamic selectors need runtime detection

---

## Data Structure

### Static Selector Example:
```json
{
  "id": "selector_0003",
  "attr": "data-SaveBtn",
  "value": "AddBtn",
  "module": "AddExisting",
  "filePath": "app\\add-existing\\add-existing.component.html",
  "elementType": "button",
  "isDynamic": false
}
```

**How L1 uses this:**
```python
# Selector string: [data-SaveBtn="AddBtn"]
selector = f'[{item["attr"]}="{item["value"]}"]'
element = driver.find_element(By.CSS_SELECTOR, selector)
```

---

### Dynamic Selector Example (with values):
```json
{
  "id": "selector_0225",
  "attr": "data-routeToDetailViewDialog",
  "value": "{{buttonName}}",
  "module": "CreateNew",
  "elementType": "button",
  "isDynamic": true,
  "possibleValues": [
    "aeName.Test.name",
    "aeName.TestStep.name"
  ]
}
```

**How L1 uses this:**
```python
# Try each possible value
for possible_value in item.get("possibleValues", []):
    selector = f'[{item["attr"]}="{possible_value}"]'
    try:
        element = driver.find_element(By.CSS_SELECTOR, selector)
        break  # Found it!
    except NoSuchElementException:
        continue  # Try next value
```

---

### Dynamic Selector Example (no values extracted):
```json
{
  "id": "selector_0005",
  "attr": "data-addInstance",
  "value": "{{templateName}}",
  "module": "AddExisting",
  "elementType": "h4",
  "isDynamic": true
}
```

**How L1 uses this:**
```python
# Option 1: Try wildcard selector (finds any value)
selector = f'[{item["attr"]}]'  # Just [data-addInstance]
element = driver.find_element(By.CSS_SELECTOR, selector)

# Option 2: Skip and let L2 handle it
if item["isDynamic"] and "possibleValues" not in item:
    return None  # Fall through to L2
```

---

## L1 Strategy with This Data

### Recommended Approach:

```python
def L1_find_element(target_description, selectors_json):
    """
    L1: Custom selector-based element finding
    """
    # Load selectors
    with open('phase2_dynamic_selectors.json') as f:
        data = json.load(f)
        selectors = data['selectors']

    # Filter by module (if known)
    if current_module:
        selectors = [s for s in selectors if s['module'] == current_module]

    # Strategy 1: Try static selectors first (fast & reliable)
    for selector_obj in selectors:
        if not selector_obj['isDynamic']:
            css_selector = f'[{selector_obj["attr"]}="{selector_obj["value"]}"]'
            element = try_find(css_selector)
            if element:
                return element

    # Strategy 2: Try dynamic selectors with known values
    for selector_obj in selectors:
        if selector_obj['isDynamic'] and 'possibleValues' in selector_obj:
            for value in selector_obj['possibleValues']:
                css_selector = f'[{selector_obj["attr"]}="{value}"]'
                element = try_find(css_selector)
                if element:
                    return element

    # Strategy 3: Try dynamic selectors with wildcard
    for selector_obj in selectors:
        if selector_obj['isDynamic']:
            css_selector = f'[{selector_obj["attr"]}]'
            element = try_find(css_selector)
            if element:
                return element

    # Failed - fall through to L2
    return None
```

---

## Key Information Available

### For Each Selector:

| Field | Description | Use Case |
|-------|-------------|----------|
| `attr` | Selector attribute (e.g., "data-SaveBtn") | Build CSS selector |
| `value` | Selector value (e.g., "AddBtn" or "{{variable}}") | Build exact selector |
| `module` | Module name (e.g., "AddExisting") | Filter by current page/module |
| `elementType` | HTML tag (e.g., "button", "input") | Additional validation |
| `isDynamic` | Boolean | Decide strategy (exact match vs wildcard) |
| `possibleValues` | Array of strings (if found) | Try each value for dynamic selectors |
| `filePath` | Source file path | Debug/reference |
| `htmlSnippet` | HTML context | Understand element context |

---

## Testing Scenarios

### Scenario 1: Find Save Button in AddExisting Module

**What you know:**
- Current module: "AddExisting"
- Looking for: "save button"

**L1 Strategy:**
```python
# Filter by module
selectors = [s for s in all_selectors if s['module'] == 'AddExisting']

# Filter by element type
button_selectors = [s for s in selectors if s['elementType'] == 'button']

# Try each selector
for s in button_selectors:
    if 'Save' in s['attr'] or 'save' in s['value'].lower():
        css = f'[{s["attr"]}="{s["value"]}"]'
        element = try_find(css)
        if element:
            return element
```

**Result:**
```
Found: [data-SaveBtn="AddBtn"]
Success rate: HIGH (static selector)
```

---

### Scenario 2: Find Dynamic Button in CreateNew Module

**What you know:**
- Current module: "CreateNew"
- Looking for: "create test button"

**L1 Strategy:**
```python
# Filter by module
selectors = [s for s in all_selectors if s['module'] == 'CreateNew']

# Find dynamic selectors with values
dynamic_selectors = [s for s in selectors if s['isDynamic'] and 'possibleValues' in s]

# Try possibleValues
for s in dynamic_selectors:
    if 'create' in s['attr'].lower():
        for value in s['possibleValues']:
            if 'Test' in value:
                css = f'[{s["attr"]}="{value}"]'
                element = try_find(css)
                if element:
                    return element
```

**Result:**
```
Found: [data-routeToDetailViewDialog="aeName.Test.name"]
Success rate: MEDIUM (dynamic with extracted values)
```

---

### Scenario 3: Unknown Dynamic Value

**What you know:**
- Current module: "AddExisting"
- Looking for: element with "data-addInstance"

**L1 Strategy:**
```python
# Try wildcard selector
css = '[data-addInstance]'
elements = driver.find_elements(By.CSS_SELECTOR, css)

if len(elements) == 1:
    return elements[0]  # Found unique element
elif len(elements) > 1:
    # Multiple matches - need more context
    return None  # Fall to L2
else:
    return None  # Not found - fall to L2
```

**Result:**
```
Found: [data-addInstance] (matches 1 element)
Success rate: LOW-MEDIUM (depends on uniqueness)
```

---

## Expected L1 Success Rate

### Before (without enriched selectors):
- L1 success: ~25%
- Why: Manually maintained selector file, incomplete coverage

### After (with enriched selectors):
- **Static selectors (595):** ~85-90% success
  - Exact match, no ambiguity
  - Example: `[data-SaveBtn="AddBtn"]`

- **Dynamic with values (7):** ~70-80% success
  - Multiple values to try
  - Example: Try "aeName.Test.name" then "aeName.TestStep.name"

- **Dynamic without values (282):** ~40-50% success
  - Wildcard match, may have duplicates
  - Example: `[data-addInstance]` may match multiple elements

- **Overall L1 success estimate:** ~70-75%

---

## What Happens When L1 Fails?

### Fall through to L2:
```python
def find_element(target):
    # Try L1
    element = L1_find_element(target)
    if element:
        return element

    # L1 failed → Try L2 (generic patterns)
    element = L2_find_element(target)
    if element:
        return element

    # L2 failed → Try L3 (CV-guided vision)
    element = L3_find_element(target)
    return element
```

---

## Summary

### What You Can Do Now:

1. **Load the JSON file:**
   ```python
   import json
   with open('Selectors_Folder/phase2_dynamic_selectors.json') as f:
       data = json.load(f)
       selectors = data['selectors']
   ```

2. **Filter by module:**
   ```python
   module_selectors = [s for s in selectors if s['module'] == 'CreateNew']
   ```

3. **Try static selectors first:**
   ```python
   static = [s for s in module_selectors if not s['isDynamic']]
   ```

4. **Try dynamic selectors with values:**
   ```python
   dynamic_with_values = [s for s in module_selectors
                          if s['isDynamic'] and 'possibleValues' in s]
   ```

5. **Fall back to wildcard for others:**
   ```python
   css = f'[{selector["attr"]}]'
   ```

---

## Next Steps

### Option 1: Test Now
- Integrate this JSON into your L1 agent
- Run your test suite
- Measure L1 success rate improvement

### Option 2: Add Phase 3 & 4 First
- Phase 3: Add keywords for semantic matching
  - Example: "save button" → matches selectors with "save" keyword
- Phase 4: Add priority scores
  - Example: Try priority=10 selectors before priority=5

### Recommendation:
**Test now with what we have!** See if L1 improves from 25% → 70%+.
If that's good enough, you're done. If not, come back for Phase 3 & 4.

---

## Questions to Test:

1. Can L1 find static selectors reliably? (Should be ~90%)
2. Can L1 find dynamic selectors with possibleValues? (Should be ~70%)
3. Do wildcard selectors work when unique? (Varies by page)
4. What's the overall L1 success rate now?

**After testing, you'll know if you need Phase 3 & 4!**
