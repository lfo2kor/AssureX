# Complete Analysis: L1, L2, L3 Selector Strategy

**Document Purpose**: Detailed analysis of current implementation, problems, and solutions for all three levels

---

## LEVEL 1: Custom Selector Matching (JSON-based)

### CURRENT IMPLEMENTATION

**Location**: `utils/selector_loader_v2.py` (with sequential context) or `utils/selector_loader.py` (basic version)

**What It Does**:
- Searches through `selectors_enriched_all_modules.json` (1340+ selectors)
- Uses **keyword extraction** from step text (e.g., "Click edit button" → `['edit', 'btn', 'button']`)
- **Scores each matching selector** based on:
  - Keyword in `attr/value/label`: **+5 points** each
  - Keyword in `context` array: **+8 points** each (stronger signal)
  - `priority` field: **+0 to +12 points**
  - Sequential state boost: **+0 to +30 points** (if recently used module)
- Returns **highest-scoring selector**

**Algorithm Flow** (step_executor.py:242-349):
```
1. Extract keywords from step text
2. Get search scope (active modules from sequential context)
3. Filter 1340 selectors → ~250 selectors (by module)
4. Score each candidate
5. Sort by score descending
6. Try highest-scoring selector
7. If count == 1: Execute action ✅
8. If count > 1: Skip to L3 (ambiguous)
9. If count == 0: Try next level
```

**Example** (RBPLCD-8835 Step 5):
```
Step: "Click edit button of part default_testobject_01"
Keywords: ['edit', 'btn', 'button']
Search Scope: [AddExisting, TestObject, Parts] (from state)

Candidate 1: ReplaceBtn
  - 'btn' in value: +5
  - 'edit' in context: +8
  - 'button' in context: +8
  - priority: +12
  - state boost (TestObject active): +10
  → Total: 43 ✅ Winner

Candidate 2: SaveBtn
  - 'btn' in value: +5
  - 'button' in context: +8
  → Total: 13

Result: button[data-cy="ReplaceBtn"] (Count: 1) → Execute Click
```

---

### PROBLEMS WITH L1

#### Problem 1: **Keyword Ambiguity - Multiple Matches**

**Issue**: Same keywords appear in MANY selectors

**Real Example**:
```
Step 4: "Select a project and product"
Keywords: ['project', 'product', 'select']

Step 5: "Click edit button"
Keywords: ['edit', 'btn', 'button']

Step 6: "Click save button"
Keywords: ['save', 'btn', 'button']  ← Same 'btn', 'button' as Step 5!
```

**What Happens**:
- Both Step 5 and Step 6 will match ALL button selectors: `EditBtn`, `SaveBtn`, `DeleteBtn`, `CancelBtn`, `CloseBtn`, etc.
- Scoring tries to differentiate, but if:
  - `SaveBtn` has `context: ['save', 'button']` → Score: 5+8+8 = **21**
  - `EditBtn` has `context: ['edit', 'button']` → Score: 5+8+8 = **21**
- If scores are equal, **first match wins** → Unreliable!

**Why This Fails**:
- Generic keywords like "button", "click", "select" appear in 100+ selectors
- Context keywords are **static** (defined when JSON was created)
- Step semantics are NOT captured (e.g., "Click EDIT button" vs "Click SAVE button" both have 'button')

---

#### Problem 2: **Static Context Keywords Don't Match Runtime Semantics**

**Issue**: JSON context keywords are predetermined, not adaptive

**Example from JSON**:
```json
{
  "attr": "data-addExisting",
  "value": "addExisting",
  "context": ["add", "existing", "modal", "component"]  ← Static!
}
```

**Problem**:
- Step text: "Click to **include** existing item"
- Keywords: `['include', 'item']`
- JSON context: `['add', 'existing']`
- **No match!** Even though semantically it's the same action

**Why This Fails**:
- Users write steps with varying wording: "add", "include", "insert", "attach"
- JSON context was extracted from HTML at **design time**, not **test time**
- No semantic understanding of synonyms

---

#### Problem 3: **Scoring Overlaps Lead to Wrong Selector**

**Issue**: Multiple selectors get similar high scores

**Real Scenario**:
```
Step: "Click edit button of part default_testobject_01"

Candidate 1: ReplaceBtn (in TestObject module)
  - 'edit' in context: +8
  - 'btn' in value: +5
  - 'button' in context: +8
  - priority: +12
  - state boost (TestObject recently used): +10
  → Score: 43

Candidate 2: EditBtn (in Parts module)
  - 'edit' in value: +5
  - 'edit' in context: +8
  - 'btn' in value: +5
  - 'button' in context: +8
  - priority: +10
  → Score: 36

Candidate 3: EditIcon (in AddExisting module)
  - 'edit' in value: +5
  - 'edit' in context: +8
  - priority: +8
  - state boost (AddExisting recently used): +15
  → Score: 36
```

**Problem**:
- `ReplaceBtn` wins (score 43), but is it the CORRECT button?
- If `ReplaceBtn` is actually "Replace existing item" and NOT "Edit item", L1 will click the wrong button!
- User expects "Edit button" but gets "Replace button" because of keyword overlap

**Why This Fails**:
- Scoring is **heuristic**, not semantic
- Priority and state boost can **override actual intent**
- No verification that the selector's PURPOSE matches the step's INTENT

---

#### Problem 4: **Dynamic Selectors with Multiple Values Are Unreliable**

**Issue**: Dynamic selectors try all `possibleValues`, but don't know which is correct

**Example**:
```json
{
  "attr": "data-testobject",
  "value": "{{testobject_name}}",
  "isDynamic": true,
  "possibleValues": ["default_testobject_01", "default_testobject_02", "Measurement01"]
}
```

**Algorithm** (step_executor.py:309-326):
```python
for value in possible_values:
    selector_str = f'[data-testobject="{value}"]'
    count = page.locator(selector_str).count()
    if count > 0:
        return execute_action(selector_str)  # First match wins!
```

**Problem**:
- Step: "Click on teststep named as **Measurement01**"
- L1 tries:
  1. `[data-testobject="default_testobject_01"]` → Count: 1 ✅ **WRONG!**
  2. `[data-testobject="default_testobject_02"]` → Never tried
  3. `[data-testobject="Measurement01"]` → Never tried (correct one!)

**Why This Fails**:
- L1 doesn't parse the **specific value** from step text ("Measurement01")
- Just tries all values in order, **first match wins**
- Should extract "Measurement01" from step and try THAT value only!

---

### SOLUTIONS FOR L1

#### Solution 1: **Use Semantic Embeddings Instead of Keywords**

**Current**: Keyword matching (exact substring match)
**Proposed**: Embedding-based semantic similarity

**Implementation**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings

# At load time: Pre-compute embeddings for all selectors
for selector in selectors:
    metadata_text = f"{selector['attr']} {selector['value']} {' '.join(selector.get('context', []))}"
    selector['embedding'] = model.encode(metadata_text)

# At search time: Find most similar selector
step_embedding = model.encode(step_text)
scores = []
for selector in filtered_selectors:
    similarity = cosine_similarity(step_embedding, selector['embedding'])
    scores.append((selector, similarity))

# Return highest similarity (NOT keyword score)
best_selector = max(scores, key=lambda x: x[1])
```

**Advantages**:
- "Click edit button" semantically matches `EditBtn` better than `SaveBtn`
- Handles synonyms: "add existing" ≈ "include existing" ≈ "attach existing"
- No manual keyword extraction needed
- More robust to wording variations

**Trade-offs**:
- Slower (embedding computation takes ~10-50ms per step)
- Requires external library (`sentence-transformers`)
- Need to pre-compute embeddings when selectors.json loads

---

#### Solution 2: **Hybrid Approach (Keywords for Filtering + Embeddings for Ranking)**

**Best of Both Worlds**:

```python
def find_best_selector(step_text, module):
    # PHASE 1: Fast keyword filtering (reduce search space)
    keywords = extract_keywords(step_text)
    candidates = filter_by_keywords(keywords, module)  # 1340 → 50 selectors

    # PHASE 2: Semantic ranking (pick best match)
    step_embedding = model.encode(step_text)
    scores = []
    for candidate in candidates:
        semantic_score = cosine_similarity(step_embedding, candidate['embedding'])
        priority_boost = candidate.get('priority', 0) * 0.01  # Small boost
        final_score = semantic_score + priority_boost
        scores.append((candidate, final_score))

    return max(scores, key=lambda x: x[1])
```

**Advantages**:
- **Fast**: Keywords reduce 1340 → 50 selectors quickly
- **Accurate**: Embeddings choose the BEST match from filtered candidates
- **Backward compatible**: Still uses existing keyword logic

---

#### Solution 3: **Extract Specific Values from Step Text for Dynamic Selectors**

**Problem**: Dynamic selectors try all `possibleValues` blindly

**Solution**: Parse the specific value from step text FIRST

```python
import re

def extract_dynamic_value(step_text, selector):
    """Extract the specific value mentioned in step text"""

    # Example: "Click on teststep named as Measurement01"
    if 'named as' in step_text.lower():
        match = re.search(r'named as\s+(\S+)', step_text, re.IGNORECASE)
        if match:
            extracted_value = match.group(1)  # "Measurement01"

            # Check if this value is in possibleValues
            possible_values = selector.get('possibleValues', [])
            if extracted_value in possible_values:
                return extracted_value  # Only try THIS value!

    # Fallback: try all possible values (current behavior)
    return None
```

**Modified Algorithm**:
```python
if is_dynamic:
    # NEW: Try to extract specific value from step text
    specific_value = extract_dynamic_value(step_text, selector_obj)

    if specific_value:
        # Try ONLY the extracted value
        selector_str = build_selector(selector_obj, value_override=specific_value)
        count = page.locator(selector_str).count()
        if count == 1:
            return execute_action(selector_str)
    else:
        # Fallback: try all possible values (current logic)
        for value in selector_obj.get('possibleValues', []):
            ...
```

**Advantages**:
- More accurate (tries the CORRECT value first)
- Faster (doesn't iterate through all values)
- Fixes the "first match wins" problem

---

#### Solution 4: **Add Intent Verification (Check Selector Purpose Before Using)**

**Problem**: High-scoring selector might have WRONG purpose

**Solution**: Add `purpose` or `intent` field to selectors.json

**Enhanced JSON**:
```json
{
  "attr": "data-replacebtn",
  "value": "ReplaceBtn",
  "context": ["replace", "edit", "button"],
  "purpose": "replace existing item with new one",  ← NEW!
  "priority": 12
}
```

**Verification**:
```python
def verify_intent(step_text, selector):
    """Check if selector's purpose matches step's intent"""

    # Use embeddings to compare step text with selector purpose
    step_embedding = model.encode(step_text)
    purpose_embedding = model.encode(selector['purpose'])
    similarity = cosine_similarity(step_embedding, purpose_embedding)

    # Only use this selector if intent matches (>0.6 similarity)
    return similarity > 0.6

# In find_best_selector:
for candidate in candidates:
    if verify_intent(step_text, candidate):
        # Proceed with this candidate
        ...
```

**Advantages**:
- Prevents wrong selector from being used (even if score is high)
- Explicit semantic check
- Can reject candidates that score well but don't match intent

---

### RECOMMENDED SOLUTION FOR L1

**Immediate (Phase 1)**: Solution 3 - Extract dynamic values
- **Effort**: Low (1-2 hours)
- **Impact**: High (fixes dynamic selector issues)
- **Risk**: Low (no dependencies)

**Short-term (Phase 2)**: Solution 2 - Hybrid keyword + embeddings
- **Effort**: Medium (1-2 days)
- **Impact**: Very High (fixes keyword ambiguity)
- **Risk**: Medium (requires `sentence-transformers` library)

**Long-term (Phase 3)**: Solution 4 - Intent verification
- **Effort**: High (requires enriching all 1340+ selectors with purpose field)
- **Impact**: High (best accuracy)
- **Risk**: Low (optional enhancement)

---

---

## LEVEL 2: Generic Pattern Matching (Hardcoded CSS)

### CURRENT IMPLEMENTATION

**Location**: `utils/step_executor.py:351-510`

**What It Does**:
- Uses **hardcoded CSS selector patterns** for common UI actions
- Detects action type from step text (button click, dropdown, message verification, etc.)
- Extracts text from step (e.g., "Save" from "Click save button")
- Tries each pattern with extracted text until **unique match** (count == 1)

**Pattern Dictionary** (step_executor.py:63-107):
```python
generic_patterns = {
    'button_click': [
        "button:has-text('{text}')",           # Native button with text
        "a:has-text('{text}')",                # Link
        "[role='button']:has-text('{text}')",  # ARIA button
        "button[type='submit']",               # Submit button
        "button",                               # Any button (fallback)
    ],
    'dropdown_select': [
        "[data-attribute='{text}']",           # Data attribute
        "label:has-text('{text}') .mat-select",  # Material select
        "[role='combobox']",                   # Generic combobox
        "select",                               # Native select
    ],
    'verify_message': [
        ":has-text('Successfully edited')",    # Hardcoded message!
        "div:has-text('Successfully edited')",
        ".mat-snack-bar-container",            # Material snackbar
        "[role='alert']",                      # ARIA alert
        "*",                                   # Any element (fallback)
    ],
}
```

**Algorithm Flow**:
```
1. Detect action type from step text
   - Contains "button" → button_click
   - Contains "select" or "dropdown" → dropdown_select
   - Contains "message" or "displayed" → verify_message

2. Extract relevant text
   - "Click save button" → extracted_text = "Save"
   - "Message 'Success' should be displayed" → extracted_text = "Success"

3. Try each pattern in order
   for pattern in patterns:
       selector = pattern.format(text=extracted_text)
       count = page.locator(selector).count()

       if count == 1:
           return execute_action(selector)  ✅
       elif count > 1:
           if action == 'verify_message':
               return execute_action(selector)  ✅ (multiple OK for verification)
           else:
               continue  # Try next pattern

4. If all patterns fail → L3
```

**Example Success**:
```
Step: "Click save button"
Action: button_click
Extracted: "Save"

Try 1: button:has-text('Save') → Count: 1 ✅
→ Execute Click
```

**Example Failure**:
```
Step: "Click edit button"
Action: button_click
Extracted: "Edit"

Try 1: button:has-text('Edit') → Count: 0 (no native button)
Try 2: a:has-text('Edit') → Count: 0 (no link)
Try 3: [role='button']:has-text('Edit') → Count: 2 (ambiguous!)
Try 4: button[type='submit'] → Count: 1, but it's "Save" button (wrong!)

→ All patterns failed or ambiguous → Go to L3
```

---

### PROBLEMS WITH L2

#### Problem 1: **Hardcoded Message Text**

**Issue**: `verify_message` patterns have HARDCODED text "Successfully edited"

**Code** (step_executor.py:98-106):
```python
'verify_message': [
    ":has-text('Successfully edited')",  # ← Hardcoded!
    "div:has-text('Successfully edited')",
    ".mat-snack-bar-container",
    "[role='alert']",
    "*",  # Matches everything (unreliable)
]
```

**Problem**:
```
Step: "Message 'Project created successfully' should be displayed"
Extracted message: (IGNORED!)

Try 1: :has-text('Successfully edited') → Count: 0 (wrong message!)
Try 2: div:has-text('Successfully edited') → Count: 0
Try 3: .mat-snack-bar-container → Count: 1 (any snackbar, not verified!)
Try 4: * → Count: 100+ (matches everything!)

Result: PASSES even if actual message is "Error: Failed"!
```

**Why This Fails**:
- L2 doesn't use the extracted message text from step
- Falls back to `*` (matches all elements) → False positive!
- No semantic verification

**Root Cause** (step_executor.py:382-394):
```python
# Extract message text from quotes
match = re.search(r'^"([^"]*)"', step_text)
if match:
    extracted_text = match.group(1)  # Extracted: "Project created successfully"
else:
    extracted_text = step_text  # Fallback: use whole step
```

But then patterns DON'T use `{text}` placeholder:
```python
":has-text('Successfully edited')"  # Should be: ":has-text('{text}')"
```

---

#### Problem 2: **Static Patterns Don't Cover All UI Variations**

**Issue**: Hardcoded patterns only cover common cases, miss edge cases

**Example - Material Dropdown**:
```
HTML: <input class="mat-mdc-autocomplete-trigger" data-attribute="Type">
Step: "Click on Type and select 'Type 5' from dropdown"

L2 Patterns:
Try 1: [data-attribute='Type'] → Count: 1 ✅

But if HTML is different:
HTML: <mat-select data-field="Type">
Try 1: [data-attribute='Type'] → Count: 0
Try 2: label:has-text('Type') .mat-select → Count: 0 (no label!)
Try 3: [role='combobox'] → Count: 5 (ambiguous!)

→ Fails even though element exists!
```

**Why This Fails**:
- UI frameworks change (Material 15 vs 16 have different structures)
- Custom components don't follow standard patterns
- Patterns are FIXED at code time, can't adapt

---

#### Problem 3: **Text Extraction Is Fragile**

**Issue**: Regex-based text extraction misses variations

**Current Logic** (step_executor.py:428-431):
```python
# Extract button text
for word in ['save', 'edit', 'close', 'cancel', 'submit', 'login', 'add']:
    if word in step_lower:
        extracted_text = word.capitalize()
        break
```

**Problems**:

1. **Limited word list**: Only 7 words!
   - Step: "Click delete button" → NO MATCH! (delete not in list)
   - Step: "Click create button" → NO MATCH! (create not in list)

2. **No multi-word extraction**:
   - Step: "Click 'Add Existing' button" → extracted_text = "Add" (partial!)
   - Should be: "Add Existing"

3. **Hardcoded capitalization**:
   - Step: "Click SAVE button" → extracted_text = "Save"
   - But HTML might have: `<button>SAVE</button>` (uppercase!)
   - Pattern: `button:has-text('Save')` → Count: 0

**Example Failure**:
```
Step: "Click 'Create New Project' button"
Extracted: (NO MATCH, not in word list!)

Tries generic patterns:
Try 1: button → Count: 10 (ambiguous!)
Try 2: button[type='submit'] → Count: 1 (but it's "Save", not "Create New Project")

→ Clicks WRONG button!
```

---

#### Problem 4: **Row Scoping Is Complex and Brittle**

**Issue**: Scoped selectors for table rows are hardcoded and fragile

**Code** (step_executor.py:434-457):
```python
if row_identifier and extracted_text:
    scoped_patterns = [
        f":text-is('{row_identifier}') >> xpath=.. >> [data-{extracted_text.lower()}icon]",
        f"div:has(:text-is('{row_identifier}')) >> [data-{extracted_text.lower()}icon]",
        f":text-is('{row_identifier}') >> xpath=following-sibling::*[1] >> [data-{extracted_text.lower()}icon]",
        ...
    ]
```

**Problems**:

1. **Assumes specific HTML structure**:
   - Pattern assumes: `<div>row_name</div><icon>`
   - If structure is: `<tr><td>row_name</td><td><icon></td></tr>`, patterns fail!

2. **Uses `.first` without verification**:
   ```python
   return self._execute_action(step_text, pattern)  # In _execute_action:
   page.locator(pattern).first.click()  # Clicks first match, might be wrong!
   ```

3. **Nested structures cause issues**:
   - Pattern: `div:has-text('default_testobject_01')`
   - HTML:
     ```html
     <div>default_testobject_01
       <div>default_testobject_01_child
         <button data-editicon>Edit</button>  ← Wrong button!
       </div>
       <button data-editicon>Edit</button>  ← Correct button!
     </div>
     ```
   - `.first` clicks the CHILD's button (wrong!)

---

#### Problem 5: **No Embedding Support (As Documented by Your Team)**

**Issue**: Your team's document says L2 SHOULD use embeddings for message verification, but **code doesn't have it yet**

**From your document** (L1_L2_L3.txt:40-46):
```
Before Embedding (L2):
  Step: Message "Welcome..." must be visible
  Result: No exact pattern matches, falls back to * (unreliable)

After Embedding (L2):
  Step: Message "Welcome..." must be visible
  Result: Embedding-based semantic match finds message despite different wording
```

**Current Code Reality**:
- No embedding imports in `step_executor.py`
- No semantic similarity calculations
- **Embeddings are NOT implemented yet!**

This is a **planned feature** but not currently working.

---

### SOLUTIONS FOR L2

#### Solution 1: **Fix Message Verification - Use Extracted Text Instead of Hardcoded**

**Current**:
```python
'verify_message': [
    ":has-text('Successfully edited')",  # Hardcoded!
]
```

**Fixed**:
```python
'verify_message': [
    ":has-text('{text}')",  # Use extracted text!
    "div:has-text('{text}')",
    ".mat-snack-bar-container:has-text('{text}')",  # Verify content!
    "[role='alert']:has-text('{text}')",
]
```

**Verification Logic**:
```python
# If specific message was extracted, verify it
if extracted_text and action_type == 'verify_message':
    for pattern in patterns:
        selector = pattern.format(text=extracted_text)
        count = page.locator(selector).count()

        if count > 0:
            # Verify text content matches
            element_text = page.locator(selector).first.text_content()
            if extracted_text.lower() in element_text.lower():
                return (True, selector)  ✅
```

**Advantages**:
- Verifies ACTUAL message content
- No false positives from `*` fallback
- Simple fix (30 minutes)

---

#### Solution 2: **Implement Embedding-Based Message Verification**

**As documented by your team**, use semantic similarity for fuzzy matching:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def verify_message_with_embeddings(page, expected_message, threshold=0.75):
    """
    Find messages on page that semantically match expected message.

    Args:
        expected_message: Message from step text
        threshold: Similarity threshold (0-1)

    Returns:
        (success, selector) tuple
    """
    # Get all visible text elements on page
    text_elements = page.locator("div, span, p, [role='alert'], .notification").all()

    # Embed expected message
    expected_embedding = model.encode(expected_message)

    # Check each element for semantic similarity
    for element in text_elements:
        try:
            element_text = element.text_content().strip()
            if not element_text or len(element_text) < 5:
                continue

            # Embed element text
            element_embedding = model.encode(element_text)

            # Calculate similarity
            similarity = cosine_similarity(
                expected_embedding.reshape(1, -1),
                element_embedding.reshape(1, -1)
            )[0][0]

            # If similar enough, consider it a match
            if similarity >= threshold:
                return (True, f"matched text: '{element_text}' (similarity: {similarity:.2f})")
        except:
            continue

    return (False, "")
```

**Integration**:
```python
# In _try_level2_generic_patterns:
if action_type == 'verify_message':
    # Try exact patterns first
    success, selector = try_exact_patterns(extracted_text)
    if success:
        return (True, selector)

    # Fallback: embedding-based fuzzy match
    success, selector = verify_message_with_embeddings(page, extracted_text)
    return (success, selector)
```

**Advantages**:
- Handles wording variations: "Successfully saved" ≈ "Saved successfully"
- Works with different languages
- More robust than regex

---

#### Solution 3: **Improve Text Extraction - Use NLP Instead of Keyword List**

**Current**:
```python
for word in ['save', 'edit', ...]:  # Only 7 words!
    if word in step_lower:
        extracted_text = word.capitalize()
```

**Improved**:
```python
import re

def extract_button_text(step_text):
    """Extract button text from step using regex patterns"""

    # Pattern 1: Quoted text (highest priority)
    # "Click 'Save Changes' button" → "Save Changes"
    match = re.search(r"['\"]([^'\"]+)['\"]", step_text)
    if match:
        return match.group(1)

    # Pattern 2: Text before "button"
    # "Click Save Changes button" → "Save Changes"
    match = re.search(r'\b([\w\s]+)\s+button\b', step_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 3: Text after "Click" or "Click on"
    # "Click Save" → "Save"
    match = re.search(r'\bclick(?:\s+on)?\s+([\w\s]+)', step_text, re.IGNORECASE)
    if match:
        button_text = match.group(1).strip()
        # Remove common words
        button_text = re.sub(r'\b(the|a|an|button|btn)\b', '', button_text, flags=re.IGNORECASE).strip()
        return button_text

    return ""
```

**Advantages**:
- Works for ANY button text (not just 7 words)
- Handles multi-word buttons: "Add Existing", "Create New Project"
- Respects quoted text (exact match)

---

#### Solution 4: **Add Dynamic Pattern Generation Based on Page Context**

**Current**: Patterns are FIXED at code time
**Proposed**: Generate patterns dynamically based on page state

```python
def generate_dynamic_patterns(page, action_type, extracted_text):
    """
    Generate selector patterns based on actual page elements.

    Example: If page has Material components, generate Material patterns.
             If page has Bootstrap, generate Bootstrap patterns.
    """
    patterns = []

    if action_type == 'button_click':
        # Detect UI framework on page
        has_material = page.locator('mat-button, .mat-mdc-button').count() > 0
        has_bootstrap = page.locator('.btn').count() > 0

        if has_material:
            patterns.append(f"button[mat-button]:has-text('{extracted_text}')")
            patterns.append(f".mat-mdc-button:has-text('{extracted_text}')")

        if has_bootstrap:
            patterns.append(f"button.btn:has-text('{extracted_text}')")

        # Generic fallback
        patterns.append(f"button:has-text('{extracted_text}')")
        patterns.append(f"[role='button']:has-text('{extracted_text}')")

    return patterns
```

**Advantages**:
- Adapts to different UI frameworks
- More likely to find correct element
- Reduces ambiguous matches

---

#### Solution 5: **Fix Row Scoping with Better Verification**

**Problem**: `.first` clicks wrong button in nested structures

**Solution**: Verify the scoped selector is CLOSEST to the row identifier

```python
def find_scoped_button(page, row_identifier, button_type):
    """
    Find button in specific row using proximity check.

    Returns selector that is CLOSEST to row_identifier text.
    """
    # Find all matching buttons
    button_selector = f"[data-{button_type.lower()}icon]"
    all_buttons = page.locator(button_selector).all()

    # Find row element
    row_element = page.locator(f":text-is('{row_identifier}')").first
    row_box = row_element.bounding_box()

    # Find button closest to row
    min_distance = float('inf')
    closest_button_idx = None

    for idx, button in enumerate(all_buttons):
        button_box = button.bounding_box()

        # Calculate distance (vertical + horizontal)
        distance = abs(button_box['y'] - row_box['y']) + abs(button_box['x'] - row_box['x'])

        if distance < min_distance:
            min_distance = distance
            closest_button_idx = idx

    # Return selector for closest button
    return f"({button_selector}) >> nth={closest_button_idx}"
```

**Advantages**:
- Finds button CLOSEST to row identifier (not just first match)
- Works with nested structures
- More reliable

---

### RECOMMENDED SOLUTION FOR L2

**Immediate (Phase 1)**: Solution 1 - Fix message verification
- **Effort**: Low (1 hour)
- **Impact**: High (fixes false positives)
- **Risk**: Low

**Short-term (Phase 2)**: Solution 3 - Improve text extraction
- **Effort**: Low (2-3 hours)
- **Impact**: High (handles any button text)
- **Risk**: Low

**Medium-term (Phase 3)**: Solution 2 - Implement embeddings
- **Effort**: Medium (1-2 days)
- **Impact**: Very High (enables semantic matching)
- **Risk**: Medium (requires external library)

**Long-term (Phase 4)**: Solution 4 & 5 - Dynamic patterns + row scoping
- **Effort**: High (1 week)
- **Impact**: High (handles edge cases)
- **Risk**: Medium

---

---

## LEVEL 3: Computer Vision Guided (GPT-4o Vision API)

### CURRENT IMPLEMENTATION

**Location**: `utils/vision_helper.py` + `utils/step_executor.py:512-561`

**What It Does**:
- Takes **screenshot** + **step text** as input
- Calls **Azure OpenAI GPT-4o Vision API**
- AI analyzes screenshot and suggests **CSS selector**
- Returns `{selector, reasoning, fallback_selectors}`
- Tries primary selector, then fallbacks

**Algorithm Flow**:
```
1. Take screenshot (before step execution)
2. Call GPT-4o Vision API:
   - Prompt: "Analyze this screenshot and suggest selector for: {step_text}"
   - Response: {
       "selector": "button:has-text('Save')",
       "reasoning": "I see a blue button labeled 'Save' in the form",
       "fallback_selectors": ["button[type='submit']", ".save-btn"]
     }
3. Try primary selector:
   count = page.locator(selector).count()
   if count > 0: execute_action(selector)
4. If primary fails, try fallbacks
5. If all fail → L3 FAILS → Test step FAILS
```

**API Call** (vision_helper.py:66-150):
```python
def call_vision(screenshot_bytes, prompt, temperature=0.1, max_tokens=500):
    # Encode screenshot to base64
    base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')

    # Build message
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]

    # Call API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Parse JSON response
    result = json.loads(response.choices[0].message.content)
    return result
```

---

### PROBLEMS WITH L3

#### Problem 1: **Non-Deterministic Responses**

**Issue**: AI can return DIFFERENT selectors for the SAME screenshot

**Example**:
```
Run 1:
  Screenshot: [Login page]
  Step: "Click login button"
  Response: {"selector": "button:has-text('Login')"}
  Result: Count = 1 ✅

Run 2 (same screenshot, same step):
  Response: {"selector": "button[type='submit']"}
  Result: Count = 3 (ambiguous!)

Run 3:
  Response: {"selector": "#loginBtn"}
  Result: Count = 0 (ID doesn't exist!)
```

**Why This Happens**:
- GPT-4o is a **language model**, not deterministic
- Even with `temperature=0.1`, responses vary slightly
- Vision model "hallucinates" selectors that don't exist

**Impact**:
- **Flaky tests**: Same step passes sometimes, fails other times
- Hard to debug (different error each run)
- Can't trust L3 as reliable fallback

---

#### Problem 2: **Vision API Cannot See HTML Structure**

**Issue**: GPT-4o only sees PIXELS, not DOM structure

**What GPT-4o Sees**:
```
[Screenshot pixels]
- Blue button labeled "Save"
- Red button labeled "Cancel"
- Input field with placeholder "Enter name"
```

**What GPT-4o DOESN'T See**:
```html
<button class="btn-primary" data-cy="saveBtn">Save</button>  ← Best selector!
<button class="btn-secondary">Cancel</button>
<input type="text" id="nameInput" placeholder="Enter name">
```

**Problem**:
- GPT-4o suggests: `button:has-text('Save')` (generic)
- Actual best selector: `[data-cy="saveBtn"]` (unique, reliable)
- GPT-4o can't know about `data-cy` attributes because they're not visible in pixels!

**Impact**:
- L3 suggests **generic selectors** (text-based, less reliable)
- Misses **custom data attributes** (most reliable)
- May suggest selectors that match MULTIPLE elements

---

#### Problem 3: **Expensive and Slow**

**Issue**: Vision API calls cost money and time

**Costs**:
- GPT-4o Vision: **$5 per 1000 requests** (Azure pricing)
- Average test: 10 steps
- If 5 steps fall through to L3: 5 API calls
- 100 tests → 500 API calls → **$2.50 per test run**

**Latency**:
- Average API call: **2-5 seconds**
- 5 L3 calls per test → **10-25 seconds overhead**
- CI/CD with 100 tests → **1000-2500 seconds (17-42 minutes) just for L3 calls**

**Impact**:
- Tests are SLOW
- CI/CD costs increase
- Developers wait longer for feedback

---

#### Problem 4: **Prompt Engineering Required**

**Issue**: GPT-4o response quality depends on prompt wording

**Example Prompt** (from your code):
```python
prompt = f"Analyze this screenshot and suggest a CSS selector for: {step_text}"
```

**Problems**:
1. **No context**: GPT-4o doesn't know application structure
2. **No format specification**: Response format might vary
3. **No examples**: GPT-4o might suggest xpath instead of CSS
4. **No constraints**: GPT-4o might suggest `#id-12345` (dynamic IDs)

**Better Prompt**:
```python
prompt = f"""
You are a Playwright test automation expert. Analyze this screenshot and suggest the BEST CSS selector for the following test step:

Step: "{step_text}"

Requirements:
1. Return ONLY valid Playwright CSS selectors (not xpath)
2. Prefer stable selectors (avoid dynamic IDs)
3. Prefer unique selectors (count should be 1)
4. Use Playwright syntax: :has-text(), :visible, etc.
5. Provide 2-3 fallback selectors in case primary fails

Return JSON format:
{{
  "selector": "primary selector here",
  "reasoning": "why this selector is best",
  "fallback_selectors": ["fallback1", "fallback2"]
}}

Context: This is a web application for test management.
"""
```

**Impact**:
- Better prompts → better selectors
- But requires experimentation and tuning
- Different apps might need different prompts

---

#### Problem 5: **No Learning or Feedback Loop**

**Issue**: L3 doesn't learn from past successes/failures

**Scenario**:
```
Test Run 1:
  Step 5: L3 suggests "button:has-text('Edit')" → Count = 0 → FAILS
  (Human fixes: actual selector is "[data-editicon]")

Test Run 2 (same test):
  Step 5: L3 suggests "button:has-text('Edit')" AGAIN → FAILS AGAIN!
```

**Why This Happens**:
- Each L3 call is **stateless** (no memory)
- GPT-4o doesn't know what worked before
- No feedback from test results to improve prompts

**Impact**:
- Repeated failures on same steps
- No improvement over time
- Manual fixes don't help future runs

---

#### Problem 6: **JSON Parsing Failures**

**Issue**: GPT-4o sometimes returns invalid JSON

**Code** (vision_helper.py:126-144):
```python
try:
    # Clean up markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]

    result = json.loads(content)
    return result
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse JSON: {e}")
    return {"raw_response": content, "error": "Invalid JSON"}
```

**Common Issues**:
1. GPT-4o adds extra text:
   ```
   Here's the selector you need:
   {"selector": "button:has-text('Save')"}
   This should work because...
   ```

2. GPT-4o uses wrong quotes:
   ```
   {'selector': 'button'}  ← Single quotes (invalid JSON!)
   ```

3. GPT-4o returns explanation instead of JSON:
   ```
   I can see a blue button labeled "Save" in the center of the screen...
   ```

**Impact**:
- L3 fails even when correct selector exists
- Error handling returns `{"error": "Invalid JSON"}` → L3 fails → test fails

---

### SOLUTIONS FOR L3

#### Solution 1: **Add HTML Context to Vision Prompt**

**Problem**: GPT-4o only sees pixels, not DOM structure

**Solution**: Send **screenshot + simplified HTML** to GPT-4o

```python
def get_simplified_html(page):
    """Extract simplified HTML with only interactive elements"""

    # Get all interactive elements
    html_snippet = page.evaluate("""() => {
        const elements = Array.from(document.querySelectorAll('button, input, a, select, [role="button"], [data-cy], [data-testid]'));

        return elements.map(el => {
            const tag = el.tagName.toLowerCase();
            const text = el.textContent.trim().slice(0, 50);
            const attrs = Array.from(el.attributes)
                .filter(attr => ['id', 'class', 'data-cy', 'data-testid', 'role', 'type'].includes(attr.name))
                .map(attr => `${attr.name}="${attr.value}"`)
                .join(' ');

            return `<${tag} ${attrs}>${text}</${tag}>`;
        }).join('\n');
    }""")

    return html_snippet
```

**Enhanced Vision Call**:
```python
def identify_step_selector_with_html(screenshot, step_text, page):
    html_context = get_simplified_html(page)

    prompt = f"""
Analyze this screenshot AND HTML structure to find the best selector.

Step: "{step_text}"

HTML (interactive elements only):
{html_context}

Suggest the BEST CSS selector from the HTML above.
Return JSON: {{"selector": "...", "reasoning": "..."}}
"""

    return call_vision(screenshot, prompt)
```

**Advantages**:
- GPT-4o sees ACTUAL selectors (data-cy, IDs, classes)
- Can suggest **reliable** selectors instead of generic text-based
- More accurate (uses both visual + structural info)

---

#### Solution 2: **Implement Caching/Memoization for L3 Results**

**Problem**: Same step calls L3 multiple times (expensive, slow)

**Solution**: Cache L3 results by screenshot hash + step text

```python
import hashlib
from functools import lru_cache

class VisionClientWithCache:
    def __init__(self):
        self.cache = {}  # {cache_key: result}

    def identify_step_selector_cached(self, screenshot, step_text):
        # Create cache key (screenshot hash + step text)
        screenshot_hash = hashlib.md5(screenshot).hexdigest()[:16]
        cache_key = f"{screenshot_hash}_{step_text}"

        # Check cache
        if cache_key in self.cache:
            logger.info(f"L3 CACHE HIT: {step_text[:50]}...")
            return self.cache[cache_key]

        # Call API
        logger.info(f"L3 CACHE MISS: Calling Vision API...")
        result = self.call_vision(screenshot, step_text)

        # Store in cache
        self.cache[cache_key] = result
        return result
```

**Advantages**:
- **Saves money**: Same screenshots reuse cached results
- **Faster**: No API call if cached
- **Deterministic**: Same screenshot always returns same selector (fixes Problem 1!)

**Trade-offs**:
- Cache can grow large (limit to 100 entries)
- Cache is per test run (doesn't persist across runs)

---

#### Solution 3: **Add Feedback Loop - Learn from Failures**

**Problem**: L3 doesn't learn from past mistakes

**Solution**: Store L3 failures and successes, improve prompts

```python
class VisionClientWithFeedback:
    def __init__(self):
        self.feedback_log = []  # [(step_text, screenshot_hash, suggested_selector, actual_selector, success)]

    def identify_step_selector_with_feedback(self, screenshot, step_text):
        # Get similar past steps
        similar_steps = self.find_similar_steps(step_text)

        # Add context to prompt
        prompt = f"""
Step: "{step_text}"

Past similar steps:
{self.format_similar_steps(similar_steps)}

Suggest selector based on past successes.
"""

        result = self.call_vision(screenshot, prompt)
        return result

    def log_result(self, step_text, screenshot, suggested_selector, actual_selector, success):
        """Log L3 result for future learning"""
        self.feedback_log.append({
            "step_text": step_text,
            "screenshot_hash": hashlib.md5(screenshot).hexdigest(),
            "suggested_selector": suggested_selector,
            "actual_selector": actual_selector,  # From manual fix
            "success": success,
            "timestamp": time.time()
        })

        # Save to file (persist across runs)
        with open("l3_feedback.json", "w") as f:
            json.dump(self.feedback_log, f, indent=2)
```

**Advantages**:
- L3 improves over time
- Learns from manual fixes
- Can suggest "similar steps used this selector successfully"

---

#### Solution 4: **Improve Prompt Engineering with Few-Shot Examples**

**Problem**: GPT-4o needs better prompts for consistent results

**Solution**: Add few-shot examples in prompt

```python
VISION_PROMPT_TEMPLATE = """
You are a Playwright test automation expert. Analyze this screenshot and suggest the BEST CSS selector.

Step: "{step_text}"

EXAMPLES OF GOOD SELECTORS:

Example 1:
Step: "Click save button"
Screenshot: [Blue button labeled "Save"]
Best selector: button[data-cy="saveBtn"]  ← Preferred (unique, stable)
Fallback 1: button:has-text("Save")       ← Good (visible text)
Fallback 2: button[type="submit"]         ← OK (generic)

Example 2:
Step: "Enter name in input field"
Best selector: input[data-testid="nameInput"]
Fallback 1: input[placeholder*="name"]
Fallback 2: input[type="text"]

Now analyze THIS screenshot:

Requirements:
- Prefer [data-cy], [data-testid], or unique IDs
- Avoid dynamic IDs (e.g., #mat-input-123)
- Use Playwright syntax (:has-text, :visible)
- Provide 2-3 fallbacks

Return JSON:
{{
  "selector": "primary selector",
  "reasoning": "why this is best",
  "fallback_selectors": ["fallback1", "fallback2"]
}}
"""
```

**Advantages**:
- More consistent responses
- GPT-4o understands expected format
- Better selector quality

---

#### Solution 5: **Implement Retry with Refinement**

**Problem**: L3 sometimes suggests wrong selector

**Solution**: If selector fails (count=0), ask GPT-4o to refine

```python
def identify_step_selector_with_retry(screenshot, step_text, page, max_retries=2):
    """
    Try L3 with refinement if first attempt fails.
    """

    for attempt in range(max_retries):
        # Get selector suggestion
        result = call_vision(screenshot, step_text)
        selector = result['selector']

        # Try selector
        count = page.locator(selector).count()

        if count == 1:
            return (True, selector)  ✅
        elif count > 1:
            # Ambiguous - ask GPT to refine
            prompt = f"""
Your previous suggestion "{selector}" matched {count} elements (ambiguous).

Suggest a MORE SPECIFIC selector that matches only ONE element.
"""
            continue
        else:
            # Not found - ask GPT to try different approach
            prompt = f"""
Your previous suggestion "{selector}" matched 0 elements.

Suggest an ALTERNATIVE selector (try different attributes or text matching).
"""
            continue

    return (False, "")
```

**Advantages**:
- Gives GPT-4o a chance to correct itself
- Better success rate
- Still deterministic (same screenshot, same retries)

---

#### Solution 6: **Hybrid L3: Vision + DOM Analysis**

**Problem**: Vision alone is insufficient

**Solution**: Combine GPT-4o Vision + Playwright DOM analysis

```python
def hybrid_l3_selector_discovery(screenshot, step_text, page):
    """
    Phase 1: Use GPT-4o Vision to IDENTIFY the target element visually
    Phase 2: Use Playwright to find the BEST selector for that element
    """

    # PHASE 1: Vision identifies element location
    vision_prompt = f"""
Analyze this screenshot and identify the LOCATION and APPEARANCE of the element for: "{step_text}"

Return JSON:
{{
  "element_type": "button | input | link | ...",
  "visible_text": "exact text visible on element",
  "approximate_position": "top-left | center | bottom-right | ...",
  "color": "blue | red | ...",
  "other_identifiers": ["icon", "bordered", ...]
}}
"""

    vision_result = call_vision(screenshot, vision_prompt)

    # PHASE 2: Use Playwright to find element matching description
    element_info = vision_result

    # Build selector candidates based on description
    candidates = []

    if element_info['visible_text']:
        candidates.append(f"{element_info['element_type']}:has-text('{element_info['visible_text']}')")

    # Get all elements of this type
    all_elements = page.locator(element_info['element_type']).all()

    # Find element matching description
    for elem in all_elements:
        text = elem.text_content()
        if element_info['visible_text'] in text:
            # Found it! Get BEST selector for THIS specific element
            best_selector = page.evaluate("""(elem) => {
                // Generate best selector (prefer data-cy, id, unique class)
                if (elem.dataset.cy) return `[data-cy="${elem.dataset.cy}"]`;
                if (elem.id) return `#${elem.id}`;
                // ... more logic
            }""", elem)

            return (True, best_selector)

    return (False, "")
```

**Advantages**:
- Vision identifies WHAT to click (visual understanding)
- Playwright generates BEST selector (DOM structure)
- Best of both worlds!

---

### RECOMMENDED SOLUTION FOR L3

**Immediate (Phase 1)**: Solution 2 - Implement caching
- **Effort**: Low (2-3 hours)
- **Impact**: High (saves money, improves speed, fixes non-determinism)
- **Risk**: Low

**Short-term (Phase 2)**: Solution 1 - Add HTML context
- **Effort**: Medium (1 day)
- **Impact**: Very High (better selectors, more reliable)
- **Risk**: Low

**Medium-term (Phase 3)**: Solution 4 - Improve prompt engineering
- **Effort**: Medium (1-2 days of experimentation)
- **Impact**: High (more consistent results)
- **Risk**: Low

**Long-term (Phase 4)**: Solution 6 - Hybrid L3 (Vision + DOM)
- **Effort**: High (1 week)
- **Impact**: Very High (best accuracy)
- **Risk**: Medium (complex implementation)

---

---

## SUMMARY TABLE: Current State, Problems, Solutions

| Level | Current Implementation | Main Problems | Recommended Solutions |
|-------|----------------------|--------------|---------------------|
| **L1** | Keyword extraction + scoring (1340+ selectors from JSON) | 1. Keyword ambiguity (same keywords → multiple selectors)<br>2. Static context (no semantic understanding)<br>3. Scoring overlaps (wrong selector wins)<br>4. Dynamic selectors try all values (first match wins) | **Phase 1**: Extract specific dynamic values from step text<br>**Phase 2**: Hybrid keyword + embeddings<br>**Phase 3**: Intent verification |
| **L2** | Hardcoded CSS patterns (button, dropdown, message) | 1. Hardcoded "Successfully edited" message (not extracted)<br>2. Limited text extraction (only 7 words)<br>3. Static patterns don't cover all UI variations<br>4. Row scoping is fragile<br>5. No embeddings (planned but not implemented) | **Phase 1**: Fix message verification (use extracted text)<br>**Phase 2**: Improve text extraction (regex patterns)<br>**Phase 3**: Implement embeddings<br>**Phase 4**: Dynamic patterns + better row scoping |
| **L3** | GPT-4o Vision API (screenshot → selector suggestion) | 1. Non-deterministic responses (flaky tests)<br>2. Only sees pixels (misses HTML attributes)<br>3. Expensive ($5/1000 calls) and slow (2-5 sec)<br>4. Needs prompt engineering<br>5. No learning from failures<br>6. JSON parsing failures | **Phase 1**: Caching (save money, fix non-determinism)<br>**Phase 2**: Add HTML context<br>**Phase 3**: Improve prompts (few-shot)<br>**Phase 4**: Hybrid Vision + DOM analysis |

---

## NEXT STEPS

### Immediate Actions (Week 1):
1. **L1**: Implement dynamic value extraction (Solution 3)
2. **L2**: Fix message verification to use extracted text (Solution 1)
3. **L3**: Add caching (Solution 2)

**Estimated Effort**: 1-2 days
**Expected Impact**: 30-40% fewer L3 calls, more reliable message verification

### Short-term Actions (Week 2-3):
1. **L1**: Implement hybrid keyword + embeddings (Solution 2)
2. **L2**: Improve text extraction with regex (Solution 3)
3. **L3**: Add HTML context to vision prompts (Solution 1)

**Estimated Effort**: 1 week
**Expected Impact**: 50-60% improvement in L1 accuracy, better L3 selectors

### Medium-term Actions (Month 1-2):
1. **L2**: Implement embeddings for semantic message matching
2. **L3**: Improve prompt engineering with few-shot examples
3. **All Levels**: Add comprehensive logging and metrics

**Estimated Effort**: 2-3 weeks
**Expected Impact**: 70-80% test reliability, reduced L3 dependency

### Long-term Vision (Month 3+):
1. **L1**: Intent verification with purpose field
2. **L2**: Dynamic pattern generation based on page context
3. **L3**: Hybrid Vision + DOM analysis
4. **All Levels**: Feedback loop for continuous improvement

**Estimated Effort**: 1-2 months
**Expected Impact**: 90%+ test reliability, minimal manual intervention

---

**Document Created**: 2025-11-07
**Author**: Analysis based on codebase review
**Version**: 1.0
