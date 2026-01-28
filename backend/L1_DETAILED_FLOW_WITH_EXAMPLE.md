# L1 (Level 1) - Detailed Flow Analysis

**Purpose:** Find the correct selector from selectors.json by matching step text to selector metadata

---

## THE PURPOSE OF L1

**Primary Goal:** Find the best matching custom selector from `selectors_merged_runtime_fixed.json` file

**Why L1 exists:**
- Your JSON contains 1340+ pre-captured selectors from the application
- Each selector has metadata: `attr`, `value`, `module`, `context`, `priority`
- L1 tries to match step text to this metadata to find the right selector
- If L1 succeeds, you get a precise, application-specific selector

**What happens if multiple selectors match:**
- L1 scores ALL matching selectors
- Selects the one with the **highest score**
- If the highest-scoring selector doesn't work (count=0), L1 fails

---

## EXAMPLE: RBPLCD-8835 Step 5

Let's trace Step 5 from RBPLCD-8835 through L1 in complete detail.

### **Test Case Context:**

**Ticket:** RBPLCD-8835 - "edit part details"
**Step 5:** "click on edit button of part default_testobject_01"

**All 8 Steps:**
1. Login
2. navigate to teststep
3. click on teststep named as default_Measurement01
4. open parts accordion
5. **click on edit button of part default_testobject_01** ← WE'LL TRACE THIS!
6. Click on Type from mandatory field and select "Type 5" from drop down
7. click on save
8. "Successfully edited: 'TestObject' default_testobject_01" message should be displayed

---

## L1 EXECUTION - STEP-BY-STEP TRACE

---

### **STEP 1: Entry Point**

**File:** `utils/step_executor.py`
**Function:** `_execute_three_level_strategy()` (line 202)

```python
def _execute_three_level_strategy(self, step_text: str, screenshot: bytes) -> tuple:
    # Update sequential context state BEFORE trying any levels
    self.selector_loader.update_state_for_step(step_text, self.module)

    # LEVEL 1: Custom selectors from JSON
    self.logger.info("LEVEL 1: Trying custom selectors from JSON...")
    success, selector = self._try_level1_custom_selectors(step_text)
    if success:
        self.logger.info(f"✅ Level 1 succeeded with selector: {selector}")
        return (True, selector, "Level 1 (Custom)")
```

**Input:**
- `step_text`: "click on edit button of part default_testobject_01"
- `self.module`: "Teststep" (from Jira ticket component)
- `screenshot`: bytes (screenshot_before.png)

**Output:**
- Calls `_try_level1_custom_selectors(step_text)`

---

### **STEP 2: L1 Function Entry**

**File:** `utils/step_executor.py`
**Function:** `_try_level1_custom_selectors()` (line 242)

```python
def _try_level1_custom_selectors(self, step_text: str) -> tuple:
    try:
        # Check if step requires row scoping
        import re
        step_lower = step_text.lower()
        row_identifier = None

        # Check for "named as X"
        if 'named as' in step_lower:
            match = re.search(r'named as\s+(\S+)', step_text, re.IGNORECASE)
            if match:
                row_identifier = match.group(1)

        # Check for "of part X" or "of testobject X"
        elif 'of part' in step_lower or 'of testobject' in step_lower:
            match = re.search(r'of (?:part|testobject)\s+(\S+)', step_text, re.IGNORECASE)
            if match:
                row_identifier = match.group(1)
                self.logger.info(f"Level 1: Extracted row identifier: {row_identifier}")
```

**Analysis for Step 5:**

```python
step_text = "click on edit button of part default_testobject_01"
step_lower = "click on edit button of part default_testobject_01"

# Check: 'named as' in step_lower? → NO
# Check: 'of part' in step_lower? → YES!

match = re.search(r'of (?:part|testobject)\s+(\S+)', step_text, re.IGNORECASE)
# Pattern matches: "of part default_testobject_01"
# Group 1: "default_testobject_01"

row_identifier = "default_testobject_01"
```

**Result:** `row_identifier = "default_testobject_01"`

---

### **STEP 3: Row Scoping Check**

```python
# If row scoping is needed, skip Level 1 and go to Level 2
if row_identifier:
    self.logger.info(f"Row scoping required - skipping Level 1, will use Level 2")
    return (False, "")
```

**What happens:**
- Step 5 has row_identifier = "default_testobject_01"
- **L1 SKIPS itself!**
- Returns `(False, "")` immediately

**Why skip L1?**
- Row scoping means: "find the row with this identifier, THEN click the button in that row"
- L1 only has selectors for the button itself (e.g., `data-cy="ReplaceBtn"`)
- L1 doesn't have row-scoped logic: `tr:has-text('default_testobject_01') button[data-cy='ReplaceBtn']`
- L2 has better row scoping logic (line 428+ in step_executor.py)

**Result:** L1 returns `(False, "")` → proceeds to L2

---

## WHAT IF THERE WAS NO ROW SCOPING?

Let's modify Step 5 to show L1's full flow:

**Modified Step 5:** "click on edit button"

Now there's NO row identifier. Let's continue the trace.

---

### **STEP 4: Find Best Selector (No Row Scoping)**

```python
# Find best matching selector from JSON
selector_obj = self.selector_loader.find_best_selector(step_text, self.module)

if not selector_obj:
    self.logger.info("No matching selector found in JSON")
    return (False, "")
```

**Calls:** `selector_loader_v2.py:find_best_selector()` (line 81)

---

### **STEP 5: find_best_selector() - Keyword Extraction**

**File:** `utils/selector_loader_v2.py`
**Function:** `find_best_selector()` (line 81)

```python
def find_best_selector(self, step_text: str, module: Optional[str] = None) -> Optional[Dict[str, Any]]:
    # Extract keywords from step text
    keywords = self._extract_keywords(step_text)
```

**Calls:** `_extract_keywords()` (line 222)

---

### **STEP 6: Extract Keywords from Step Text**

**File:** `utils/selector_loader_v2.py`
**Function:** `_extract_keywords()` (line 222)

```python
def _extract_keywords(self, step_text: str) -> List[str]:
    step_lower = step_text.lower()
    keywords = []

    # Check for message verification
    is_verification = False
    if 'message should be displayed' in step_lower or 'should display' in step_lower:
        keywords.extend(['message', 'notification', 'alert', 'snackbar', 'success', 'toast'])
        is_verification = True
```

**Analysis for Step 5:**

```python
step_text = "click on edit button"
step_lower = "click on edit button"
keywords = []

# Check: 'message should be displayed' in step_lower? → NO
is_verification = False
```

**Continue extraction:**

```python
# Specific UI elements
if '... +' in step_text or 'more' in step_lower or 'vertical' in step_lower:
    keywords.extend(['showmoreverticalbtn', 'showmore', 'vertical', 'more'])
```

**Check:** None of these patterns match → skip

```python
# Project/Product pattern
if ('project' in step_lower or 'product' in step_lower) and not ('dropdown' in step_lower or 'select' in step_lower):
    keywords.extend(['selectproject', 'project', 'product'])
elif 'project' in step_lower or 'product' in step_lower:
    keywords.extend(['project', 'product'])
```

**Check:** 'project' not in "click on edit button" → skip

```python
# Accordion
if 'accordion' in step_lower:
    keywords.extend(['accordion', 'panel'])
```

**Check:** 'accordion' not in step → skip

```python
# Parts
if 'parts' in step_lower:
    keywords.append('parts')
```

**Check:** 'parts' not in step → skip

```python
# Teststep
if 'teststep' in step_lower:
    keywords.append('teststep')
```

**Check:** 'teststep' not in step → skip

```python
# Action buttons - SKIP if verification
if not is_verification:
    if 'save' in step_lower:
        keywords.extend(['save', 'btn', 'button'])
    if 'edit' in step_lower:
        keywords.extend(['edit', 'btn', 'button'])
    if 'delete' in step_lower:
        keywords.extend(['delete', 'btn', 'button'])
```

**MATCH!**

```python
is_verification = False
'edit' in "click on edit button" → TRUE!

keywords.extend(['edit', 'btn', 'button'])
# → keywords = ['edit', 'btn', 'button']
```

**Continue:**

```python
if 'remove' in step_lower:
    keywords.extend(['remove', 'btn', 'button'])
if 'close' in step_lower or 'cancel' in step_lower:
    keywords.extend(['close', 'cancel', 'btn'])
```

**Check:** Neither 'remove' nor 'close' in step → skip

```python
# Generic patterns
if 'navigate' in step_lower:
    keywords.extend(['navigate', 'nav', 'menu'])
if 'dropdown' in step_lower or 'select' in step_lower:
    keywords.extend(['dropdown', 'select', 'type'])
```

**Check:** Neither 'navigate' nor 'dropdown' in step → skip

**Final Result:**
```python
return ['edit', 'btn', 'button']
```

---

### **STEP 7: Get Search Scope**

**Back in:** `find_best_selector()` (line 106)

```python
# Get search scope (V2: from state, V1: from module)
if self.use_sequential_context and self.context_tracker:
    search_modules = self.context_tracker.get_search_scope(module)
else:
    search_modules = [module] if module else []

self.logger.info(f"L1 Search: keywords={keywords}, modules={search_modules}")
```

**For RBPLCD-8835:**

```python
self.use_sequential_context = True  # Enabled by default
module = "Teststep"  # From Jira ticket

search_modules = self.context_tracker.get_search_scope("Teststep")
# → Returns: ["Teststep", "NestedTree"]
#   Why? Sequential context knows that after opening parts accordion,
#   the NestedTree module is now visible!
```

**Log Output:**
```
[INFO] L1 Search: keywords=['edit', 'btn', 'button'], modules=['Teststep', 'NestedTree']
```

---

### **STEP 8: Find and Score Candidates**

**Function:** `_find_and_score_candidates()` (line 132)

```python
def _find_and_score_candidates(self, keywords: List[str], search_modules: List[str], step_text: str) -> List[tuple[Dict[str, Any], int]]:
    candidates = []

    for selector in self.selectors:
        # Check module scope
        selector_module = selector.get('module', '').lower()

        if search_modules:
            # Check if in search scope
            if not any(sm.lower() in selector_module for sm in search_modules):
                continue
```

**Process:**

1. **Loop through ALL 1340 selectors**
2. **Filter by module scope first**

**Example selectors checked:**

```python
# Selector 1: Save button (Teststep module)
{
    "attr": "data-savebtn",
    "value": "SaveBtn",
    "module": "Teststep",
    "context": ["button", "save", "action"]
}
selector_module = "teststep"
any(sm.lower() in "teststep" for sm in ["Teststep", "NestedTree"]) → TRUE!
# → Keep this selector, calculate score

# Selector 2: Edit button (NestedTree module) - THIS IS THE ONE!
{
    "id": "selector_0691",
    "attr": "data-cy",
    "value": "ReplaceBtn",
    "module": "NestedTree",
    "context": ["button", "click", "edit", "replacebtn", "test"],
    "priority": 12,
    "step_text": "click on edit button of part default_testobject_01"
}
selector_module = "nestedtree"
any(sm.lower() in "nestedtree" for sm in ["Teststep", "NestedTree"]) → TRUE!
# → Keep this selector, calculate score

# Selector 3: Project menu (Commandbar module)
{
    "attr": "data-labelvalue",
    "value": "ui.commandbar.Project",
    "module": "Commandbar"
}
selector_module = "commandbar"
any(sm.lower() in "commandbar" for sm in ["Teststep", "NestedTree"]) → FALSE!
# → SKIP this selector (wrong module)
```

---

### **STEP 9: Calculate Score for Each Candidate**

**Function:** `_calculate_selector_score()` (line 173)

```python
def _calculate_selector_score(self, selector: Dict[str, Any], keywords: List[str], step_text: str) -> int:
    score = 0

    # 1. Keyword matching in basic fields (5 points each)
    attr = selector.get('attr', '').lower()
    value = selector.get('value', '').lower()
    label = selector.get('label', '').lower()

    for keyword in keywords:
        kw = keyword.lower()
        if kw in attr or kw in value or kw in label:
            score += 5
```

**Our keywords:** `['edit', 'btn', 'button']`

---

#### **Candidate 1: Save Button**

```json
{
    "attr": "data-savebtn",
    "value": "SaveBtn",
    "module": "Teststep",
    "context": ["button", "save", "action"]
}
```

**Score Calculation:**

```python
# 1. Basic field matching (5 points each)
attr = "data-savebtn"
value = "savebtn"
label = ""

keywords = ['edit', 'btn', 'button']

# Keyword 'edit':
#   'edit' in "data-savebtn"? → NO
#   'edit' in "savebtn"? → NO
#   score += 0

# Keyword 'btn':
#   'btn' in "data-savebtn"? → YES! (substring)
#   score += 5

# Keyword 'button':
#   'button' in "data-savebtn"? → NO
#   'button' in "savebtn"? → NO
#   score += 0

# Score after basic fields: 5
```

```python
# 2. Context matching (8 points each)
context = ["button", "save", "action"]

for keyword in ['edit', 'btn', 'button']:
    if keyword.lower() in [c.lower() for c in context]:
        score += 8

# Keyword 'edit':
#   'edit' in ["button", "save", "action"]? → NO
#   score += 0

# Keyword 'btn':
#   'btn' in ["button", "save", "action"]? → NO
#   score += 0

# Keyword 'button':
#   'button' in ["button", "save", "action"]? → YES!
#   score += 8

# Score after context: 5 + 8 = 13
```

```python
# 3. Priority boost
priority = 0  # No priority field
score += 0

# 4. State boost (sequential context)
state_boost = self.context_tracker.get_state_score_boost(selector)
# For SaveBtn in Teststep: +0 (not edit-mode specific)
score += 0

# FINAL SCORE: 13
```

---

#### **Candidate 2: Edit Button (ReplaceBtn) - THE CORRECT ONE!**

```json
{
    "id": "selector_0691",
    "attr": "data-cy",
    "value": "ReplaceBtn",
    "module": "NestedTree",
    "context": ["button", "click", "edit", "replacebtn", "test"],
    "priority": 12
}
```

**Score Calculation:**

```python
# 1. Basic field matching (5 points each)
attr = "data-cy"
value = "replacebtn"
label = ""

keywords = ['edit', 'btn', 'button']

# Keyword 'edit':
#   'edit' in "data-cy"? → NO
#   'edit' in "replacebtn"? → NO
#   score += 0

# Keyword 'btn':
#   'btn' in "data-cy"? → NO
#   'btn' in "replacebtn"? → YES!
#   score += 5

# Keyword 'button':
#   'button' in "data-cy"? → NO
#   'button' in "replacebtn"? → NO
#   score += 0

# Score after basic fields: 5
```

```python
# 2. Context matching (8 points each) - THIS IS KEY!
context = ["button", "click", "edit", "replacebtn", "test"]

for keyword in ['edit', 'btn', 'button']:
    if keyword.lower() in [c.lower() for c in context]:
        score += 8

# Keyword 'edit':
#   'edit' in ["button", "click", "edit", "replacebtn", "test"]? → YES!
#   score += 8

# Keyword 'btn':
#   'btn' in context? → NO (context has 'button', not 'btn')
#   score += 0

# Keyword 'button':
#   'button' in ["button", "click", "edit", "replacebtn", "test"]? → YES!
#   score += 8

# Score after context: 5 + 8 + 8 = 21
```

```python
# 3. Priority boost
priority = 12
score += 12

# Score after priority: 21 + 12 = 33
```

```python
# 4. State boost (sequential context)
state_boost = self.context_tracker.get_state_score_boost(selector)
# NestedTree is currently visible (parts accordion opened in step 4)
# → +10 boost
score += 10

# FINAL SCORE: 43
```

---

### **STEP 10: Sort Candidates by Score**

**Back in:** `_find_and_score_candidates()` (line 162)

```python
# Sort by score descending
candidates.sort(key=lambda x: x[1], reverse=True)

self.logger.debug(f"Found {len(candidates)} candidates")
if candidates:
    # Show top 3
    for i, (sel, score) in enumerate(candidates[:3]):
        self.logger.debug(f"  #{i+1}: {sel.get('attr')} (score={score}, module={sel.get('module')})")

return candidates
```

**Result:**

```python
candidates = [
    (selector_0691, 43),  # data-cy="ReplaceBtn" (Edit button)
    (selector_xxxx, 13),  # data-savebtn="SaveBtn" (Save button)
    (selector_yyyy, 10),  # Some other button
    # ... more candidates
]
```

**Log Output:**
```
[DEBUG] Found 15 candidates
[DEBUG]   #1: data-cy (score=43, module=NestedTree)
[DEBUG]   #2: data-savebtn (score=13, module=Teststep)
[DEBUG]   #3: data-closebtn (score=10, module=Teststep)
```

---

### **STEP 11: Return Best Candidate**

**Back in:** `find_best_selector()` (line 124)

```python
if not candidates:
    self.logger.warning(f"L1 FAIL: No selectors found for keywords={keywords}")
    return None

# Return best candidate
best_selector, best_score = candidates[0]

self.logger.info(f"L1 SUCCESS: Found '{best_selector.get('attr')}' "
                f"(score={best_score}, module={best_selector.get('module')})")

return best_selector
```

**Result:**

```python
best_selector = {
    "id": "selector_0691",
    "attr": "data-cy",
    "value": "ReplaceBtn",
    "module": "NestedTree",
    "context": ["button", "click", "edit", "replacebtn", "test"],
    "priority": 12
}
best_score = 43
```

**Log Output:**
```
[INFO] L1 SUCCESS: Found 'data-cy' (score=43, module=NestedTree)
```

**Returns:** `selector_obj` to `_try_level1_custom_selectors()`

---

### **STEP 12: Build Selector String**

**Back in:** `step_executor.py:_try_level1_custom_selectors()` (line 284)

```python
# Check if selector is dynamic
is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

if not is_dynamic:
    # STATIC SELECTOR - Try exact match
    selector_str = self.selector_loader.build_selector(selector_obj)
    self.logger.info(f"L1: Trying static selector: {selector_str}")
```

**Calls:** `selector_loader_v2.py:build_selector()` (line 286)

```python
def build_selector(self, selector_obj: Dict[str, Any], value_override: str = None) -> str:
    attr = selector_obj.get('attr', '')       # "data-cy"
    value = value_override or selector_obj.get('value', '')  # "ReplaceBtn"
    tagName = selector_obj.get('tagName', '').lower()  # "button"
    className = selector_obj.get('className', '')  # ""

    # Build attribute selector
    attr_selector = f'[{attr}="{value}"]'
    # → '[data-cy="ReplaceBtn"]'

    # Enhance with tag if available
    if tagName:
        return f"{tagName}{attr_selector}"
        # → 'button[data-cy="ReplaceBtn"]'
    else:
        return attr_selector
```

**Result:**
```python
selector_str = 'button[data-cy="ReplaceBtn"]'
```

**Log Output:**
```
[INFO] L1: Trying static selector: button[data-cy="ReplaceBtn"]
```

---

### **STEP 13: Check if Selector Exists on Page**

**Back in:** `_try_level1_custom_selectors()` (line 292)

```python
count = self.page.locator(selector_str).count()
self.logger.info(f"L1: Static selector count: {count}")

if count == 1:
    # Unique match - execute action
    return self._execute_action(step_text, selector_str)
elif count > 1:
    self.logger.warning(f"L1: Multiple matches ({count}) - ambiguous, skipping to Level 3")
    return (False, "")
else:
    self.logger.info("L1: Static selector not found on page")
    return (False, "")
```

**Execution:**

```python
selector_str = 'button[data-cy="ReplaceBtn"]'
count = self.page.locator('button[data-cy="ReplaceBtn"]').count()
# → count = 1 (exactly one edit button on the page!)
```

**Log Output:**
```
[INFO] L1: Static selector count: 1
```

**Since count == 1:**
```python
return self._execute_action(step_text, selector_str)
```

---

### **STEP 14: Execute Action**

**Function:** `_execute_action()` (line 670)

```python
def _execute_action(self, step_text: str, selector: str) -> tuple:
    """Execute appropriate action based on step text"""
    step_lower = step_text.lower()

    try:
        if 'message should be displayed' in step_lower or 'should display' in step_lower:
            # Verification step
            text_content = self.page.locator(selector).first.text_content()
            self.logger.info(f"✅ Verification passed: '{text_content}'")
            return (True, selector)

        elif 'select' in step_lower or 'dropdown' in step_lower:
            # Dropdown selection
            # ...

        else:
            # Default: CLICK
            self.page.locator(selector).first.click()
            self.page.wait_for_timeout(self.config['wait_times']['after_click'])
            self.logger.info(f"✅ Clicked: {selector}")
            return (True, selector)
```

**Analysis for Step 5:**

```python
step_text = "click on edit button"
step_lower = "click on edit button"

# Check: 'message should be displayed' in step? → NO
# Check: 'select' in step? → NO
# Default: CLICK

selector = 'button[data-cy="ReplaceBtn"]'
self.page.locator('button[data-cy="ReplaceBtn"]').first.click()
self.page.wait_for_timeout(500)  # wait_times.after_click

return (True, 'button[data-cy="ReplaceBtn"]')
```

**Log Output:**
```
[INFO] ✅ Clicked: button[data-cy="ReplaceBtn"]
```

---

### **STEP 15: Return Success to Main Flow**

**Back in:** `_execute_three_level_strategy()` (line 222)

```python
success, selector = self._try_level1_custom_selectors(step_text)
if success:
    self.logger.info(f"✅ Level 1 succeeded with selector: {selector}")
    return (True, selector, "Level 1 (Custom)")
```

**Result:**
```python
success = True
selector = 'button[data-cy="ReplaceBtn"]'

return (True, 'button[data-cy="ReplaceBtn"]', "Level 1 (Custom)")
```

**Log Output:**
```
[INFO] ✅ Level 1 succeeded with selector: button[data-cy="ReplaceBtn"]
```

---

## COMPLETE L1 FLOW SUMMARY

```
┌──────────────────────────────────────────────────────────────────┐
│ Step 5: "click on edit button"                                  │
└──────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 1: Check for row scoping                            │
         │ Pattern: "of part X" or "named as X"                     │
         │ Result: No row identifier (we removed it)                │
         │ → Continue to selector search                            │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 2: Extract keywords                                 │
         │ Input: "click on edit button"                            │
         │ Logic:                                                    │
         │   - 'edit' in step? YES → add ['edit', 'btn', 'button']  │
         │ Output: keywords = ['edit', 'btn', 'button']             │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 3: Get search scope                                 │
         │ Module: "Teststep"                                       │
         │ Sequential context knows:                                │
         │   - Parts accordion is open (Step 4)                     │
         │   - NestedTree is visible                                │
         │ Output: search_modules = ['Teststep', 'NestedTree']      │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 4: Filter selectors by module                       │
         │ Total selectors: 1340                                    │
         │ Filter: module in ['Teststep', 'NestedTree']             │
         │ Output: ~250 selectors                                   │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 5: Score each candidate                             │
         │ Keywords: ['edit', 'btn', 'button']                      │
         │                                                           │
         │ Candidate 1: data-cy="ReplaceBtn" (NestedTree)           │
         │   - 'btn' in value: +5                                   │
         │   - 'edit' in context: +8                                │
         │   - 'button' in context: +8                              │
         │   - priority: +12                                        │
         │   - state boost (NestedTree visible): +10                │
         │   → Score: 43 ✅                                         │
         │                                                           │
         │ Candidate 2: data-savebtn="SaveBtn" (Teststep)           │
         │   - 'btn' in attr: +5                                    │
         │   - 'button' in context: +8                              │
         │   - priority: +0                                         │
         │   - state boost: +0                                      │
         │   → Score: 13                                            │
         │                                                           │
         │ Winner: data-cy="ReplaceBtn" (score 43)                  │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 6: Build selector string                            │
         │ Input: {attr:"data-cy", value:"ReplaceBtn", tag:"button"}│
         │ Output: button[data-cy="ReplaceBtn"]                     │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 7: Check if exists on page                          │
         │ Selector: button[data-cy="ReplaceBtn"]                   │
         │ Count: 1 ✅ (exactly one match!)                         │
         └──────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────────────────┐
         │ STEP 8: Execute action                                   │
         │ Step has 'click' → Perform click                         │
         │ page.locator('button[data-cy="ReplaceBtn"]').click()     │
         │ Result: SUCCESS ✅                                       │
         └──────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │ L1 SUCCESS ✅ │
                    │ No need for   │
                    │ L2 or L3      │
                    └───────────────┘
```

---

## KEY POINTS ABOUT L1

### **1. Purpose:**
- Find the best custom selector from JSON (1340 selectors)
- Use keyword matching + context + priority + state awareness

### **2. When L1 skips itself:**
- Row scoping needed ("named as X" or "of part X")
- L2 has better row scoping logic

### **3. What if multiple selectors match?**
- ALL matching selectors are scored
- Highest score wins
- If highest-scoring selector count ≠ 1, L1 fails → goes to L2

### **4. Scoring components:**
- **Keyword in attr/value/label:** +5 each
- **Keyword in context:** +8 each (stronger!)
- **Priority field:** +0 to +10
- **State boost:** +0 to +30 (if sequential context enabled)

### **5. Why keywords are risky:**
- Step 4: "Click on 'Project' from menu" → keywords: ['project', 'product']
- Step 5: "Select 'MyProject' from dropdown" → keywords: ['project', 'product']
- **IDENTICAL KEYWORDS!** → Wrong selector matched

### **6. Sequential context helps:**
- Tracks state: menu_open, dropdown_open, accordion_open
- Adjusts search scope: visible modules only
- Adds state boost: +10 for visible elements

---

## THE ACTUAL RBPLCD-8835 STEP 5 BEHAVIOR

**Real Step 5:** "click on edit button of part default_testobject_01"

**What actually happened:**
1. L1 detected row scoping: "of part default_testobject_01"
2. L1 skipped itself immediately: `return (False, "")`
3. L2 handled the step with row-scoped selector:
   ```
   tr:has-text('default_testobject_01') button[data-cy='ReplaceBtn']
   ```

**This is by design!** L1 is smart enough to know when it can't handle row scoping.

---

## WHAT WOULD HAPPEN IF L1 HAD MULTIPLE HIGH-SCORING MATCHES?

**Example scenario:**

```python
candidates = [
    (selector_edit_btn_1, 43),  # data-cy="ReplaceBtn"
    (selector_edit_btn_2, 43),  # data-editbtn="EditBtn" (same score!)
    (selector_save_btn, 13),
]
```

**Result:**
```python
best_selector, best_score = candidates[0]
# → selector_edit_btn_1 (first one with score 43)

selector_str = 'button[data-cy="ReplaceBtn"]'
count = self.page.locator(selector_str).count()

# If count == 1: Execute ✅
# If count == 0: L1 FAIL (element not found) → go to L2
# If count > 1: L1 FAIL (ambiguous) → go to L2
```

**Key insight:** Even if multiple selectors have the same score, L1 only tries the first one. If it doesn't work, L1 fails and proceeds to L2.

---

*Analysis Date: November 5, 2024*
*Test Case: RBPLCD-8835 Step 5*
*Purpose: Complete L1 flow explanation with scoring details*
