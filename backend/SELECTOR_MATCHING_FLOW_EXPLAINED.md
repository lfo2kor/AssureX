# SELECTOR MATCHING FLOW - DETAILED EXPLANATION

## Your Question
**"How are you matching L1 selector, L2 and L3? Clear steps now i do not know where you are using keywords? why you are using?"**

---

## COMPLETE FLOW: Step Text → Selector Match

### INPUT EXAMPLE
```
Step 5: "Select 'MyProject' from dropdown"
```

---

## PHASE 1: KEYWORD EXTRACTION (Where Keywords Come From)

**File:** `utils/selector_loader_v2.py`
**Function:** `_extract_keywords()` (lines 222-284)

### Step-by-Step Keyword Extraction:

```python
# INPUT: step_text = "Select 'MyProject' from dropdown"

step_lower = step_text.lower()
# → "select 'myproject' from dropdown"

keywords = []

# Rule 1: Check if it's a dropdown/select step (line 280-282)
if 'dropdown' in step_lower or 'select' in step_lower:
    keywords.extend(['dropdown', 'select', 'type'])
# ✅ MATCHED! Keywords: ['dropdown', 'select', 'type']

# Rule 2: Check for 'project' keyword (lines 250-254)
if ('project' in step_lower or 'product' in step_lower) and not ('dropdown' in step_lower or 'select' in step_lower):
    keywords.extend(['selectproject', 'project', 'product'])
elif 'project' in step_lower or 'product' in step_lower:
    # For dropdown/select steps, just add project without selectproject
    keywords.extend(['project', 'product'])
# ✅ MATCHED second condition! Keywords: ['dropdown', 'select', 'type', 'project', 'product']

# FINAL KEYWORDS for Step 5:
# ['dropdown', 'select', 'type', 'project', 'product']
```

### Why Keywords?
**Purpose:** Keywords are used to match step text against selector properties (attr, value, label, context) in JSON.

---

## PHASE 2: L1 SELECTOR SEARCH (Score-Based Matching)

**File:** `utils/selector_loader_v2.py`
**Function:** `find_best_selector()` → `_find_and_score_candidates()` → `_calculate_selector_score()`

### Step 1: Filter Selectors by Module (lines 146-154)

```python
# From JSON: selectors_merged_runtime_fixed.json
# Total selectors: 1340

search_modules = ['Teststep']  # From Jira ticket

# Filter:
for selector in self.selectors:  # All 1340 selectors
    selector_module = selector.get('module', '').lower()

    if search_modules:
        if not any(sm.lower() in selector_module for sm in search_modules):
            continue  # SKIP selectors not in Teststep module

    # Only Teststep selectors remain → ~300 selectors
```

### Step 2: Score Each Remaining Selector (lines 156-160)

**Scoring Formula (lines 173-220):**
```
Total Score = Keyword Matches (5 pts each)
            + Context Matches (8 pts each)
            + Priority
            + State Boost
```

#### Example 1: `data-dropdownentitiesname`

```json
{
  "attr": "data-dropdownentitiesname",
  "value": "MyProject",
  "module": "Teststep",
  "context": ["dropdown", "option", "project", "entities", "myproject"],
  "priority": 30,
  "source": "manual_fix"
}
```

**Scoring:**
```python
keywords = ['dropdown', 'select', 'type', 'project', 'product']

# 1. Keyword matches in attr/value/label (line 194-201)
attr = "data-dropdownentitiesname"
value = "MyProject"
label = ""

for keyword in keywords:
    if keyword in attr or keyword in value or keyword in label:
        score += 5

# Matches:
# - 'project' in value? NO (value = "MyProject", keyword = "project")
# - 'dropdown' in attr? YES → +5
# - No other matches in attr/value/label

# Keyword score: 5

# 2. Context matches (line 203-208)
context = ["dropdown", "option", "project", "entities", "myproject"]

for keyword in keywords:
    if keyword in context:
        score += 8

# Matches:
# - 'dropdown' in context? YES → +8
# - 'select' in context? NO
# - 'type' in context? NO
# - 'project' in context? YES → +8
# - 'product' in context? NO

# Context score: 16

# 3. Priority (line 210-213)
priority = 30
score += priority

# Priority score: 30

# 4. State boost (line 215-218)
state_boost = 0  # (No sequential context for this example)

# TOTAL SCORE: 5 + 16 + 30 + 0 = 51
```

#### Example 2: `data-opencreatedialogdropdown`

```json
{
  "attr": "data-opencreatedialogdropdown",
  "value": "aeName.StructureLevel.name",
  "module": "Teststep",
  "context": ["create", "dialog", "structurelevel", "opendialog"],
  "priority": 20,
  "source": "manual_fix"
}
```

**Scoring:**
```python
keywords = ['dropdown', 'select', 'type', 'project', 'product']

# 1. Keyword matches in attr/value/label
attr = "data-opencreatedialogdropdown"
value = "aeName.StructureLevel.name"

# Matches:
# - 'dropdown' in attr? YES → +5
# Keyword score: 5

# 2. Context matches
context = ["create", "dialog", "structurelevel", "opendialog"]

# Matches:
# - 'dropdown' in context? NO
# - 'select' in context? NO
# - 'type' in context? NO
# - 'project' in context? NO
# - 'product' in context? NO
# Context score: 0

# 3. Priority
priority = 20
# Priority score: 20

# TOTAL SCORE: 5 + 0 + 20 + 0 = 25
```

### Step 3: Sort by Score and Return Best (line 163)

```python
candidates = [
    (selector_dropdownentitiesname, 51),  # ← HIGHEST SCORE
    (selector_opencreatedialog, 25),
    (selector_other1, 18),
    ...
]

candidates.sort(key=lambda x: x[1], reverse=True)

best_selector = candidates[0]  # Returns data-dropdownentitiesname
```

**Result:**
```
L1 SUCCESS: Found 'data-dropdownentitiesname' (score=51, module=Teststep)
```

---

## PHASE 3: SELECTOR EXECUTION (Try to Use It)

**File:** `utils/step_executor.py`
**Function:** `_try_level1_custom_selectors()` (lines 242-349)

### Step 1: Build Selector String (line 289)

```python
selector_obj = {
    "attr": "data-dropdownentitiesname",
    "value": "MyProject",
    ...
}

selector_str = self.selector_loader.build_selector(selector_obj)
# → "[data-dropdownentitiesname='MyProject']"
```

### Step 2: Check Count on Page (line 292)

```python
count = self.page.locator(selector_str).count()
# → Checks how many elements match [data-dropdownentitiesname='MyProject']
```

**3 Possible Outcomes:**

#### Outcome A: count == 1 (Perfect Match)
```python
if count == 1:
    return self._execute_action(step_text, selector_str)
    # ✅ Clicks the element
    # Returns: (True, selector_str)
```

#### Outcome B: count > 1 (Ambiguous)
```python
elif count > 1:
    self.logger.warning(f"L1: Multiple matches ({count}) - ambiguous, skipping to Level 3")
    return (False, "")
    # ❌ Too many matches, can't determine which one
    # Goes to L2
```

#### Outcome C: count == 0 (Not Found)
```python
else:
    self.logger.info("L1: Static selector not found on page")
    return (False, "")
    # ❌ Element doesn't exist
    # Goes to L2
```

---

## PHASE 4: L2 - Generic HTML Patterns (If L1 Fails)

**File:** `utils/step_executor.py`
**Function:** `_try_level2_generic_patterns()` (lines 351-500)

### What L2 Does:
```python
# L2 uses hardcoded HTML patterns based on action type

step_lower = step_text.lower()

# Example: Dropdown selection
if 'select' in step_lower and 'dropdown' in step_lower:
    # Look for generic dropdown selectors
    selectors = [
        'mat-select',
        'select',
        '[role="combobox"]',
        '[role="listbox"]'
    ]

    for selector in selectors:
        count = self.page.locator(selector).count()
        if count == 1:
            # Click to open dropdown
            self.page.locator(selector).first.click()

            # Wait for options panel
            self.page.wait_for_selector('mat-option', timeout=3000)

            # Extract value from step text
            match = re.search(r"'([^']+)'", step_text)
            value = match.group(1)  # → "MyProject"

            # Find and click option
            option_selector = f"mat-option:has-text('{value}')"
            self.page.locator(option_selector).first.click()

            return (True, option_selector)
```

**L2 is NOT using keywords** - it uses:
1. Hardcoded HTML patterns (mat-select, mat-option, etc.)
2. Regex to extract values from step text
3. Generic role attributes

---

## PHASE 5: L3 - CV-Guided (If L1 and L2 Fail)

**File:** `utils/step_executor.py`
**Function:** `_try_level3_cv_guided()` (lines 502-600)

### What L3 Does:
```python
# L3 uses Azure Computer Vision to:
# 1. Analyze screenshot
# 2. Find UI element mentioned in step text
# 3. Return bounding box coordinates
# 4. Generate selector based on coordinates

# Send screenshot + step text to Azure Vision API
response = azure_vision.analyze(screenshot, step_text)

# Response:
{
    "element_found": true,
    "bounding_box": {"x": 350, "y": 280, "width": 200, "height": 40},
    "suggested_text": "MyProject"
}

# Use coordinates to find element
element = self.page.locator(f"xpath=//*[@x>=350][@y>=280]...")

# OR use OCR text to find element
element = self.page.locator(f"text=MyProject")

return (True, "text=MyProject")
```

**L3 is NOT using keywords** - it uses:
1. Computer Vision API
2. OCR text recognition
3. Coordinate-based element location

---

## SUMMARY: Where Keywords Are Used

| Phase | Uses Keywords? | What It Uses Instead |
|-------|----------------|---------------------|
| **L1 Scoring** | ✅ YES | Keywords from step text matched against JSON selector properties |
| **L1 Execution** | ❌ NO | Just checks if selector exists on page (count) |
| **L2** | ❌ NO | Hardcoded HTML patterns (mat-select, mat-option, etc.) |
| **L3** | ❌ NO | Computer Vision + OCR |

---

## THE PROBLEM WITH KEYWORDS (Why It's Not Scalable)

### Issue 1: Hard-Coded Keyword Rules

**Current Code (lines 249-282):**
```python
if 'dropdown' in step_lower or 'select' in step_lower:
    keywords.extend(['dropdown', 'select', 'type'])

if ('project' in step_lower or 'product' in step_lower):
    keywords.extend(['project', 'product'])

if 'save' in step_lower:
    keywords.extend(['save', 'btn', 'button'])
```

**Problem:**
- ❌ Need to add code for EVERY new element type (checkbox, radio, table, etc.)
- ❌ Not maintainable - 50 different element types = 50 if-statements
- ❌ What about "pick from list"? "choose option"? "select item"? All need new rules!

### Issue 2: Keyword Conflicts (The Current Problem)

**Example:**
```
Step 4: "Select 'Project' from dropdown"
→ Keywords: ['dropdown', 'select', 'project']

Step 5: "Select 'MyProject' from dropdown"
→ Keywords: ['dropdown', 'select', 'project']

SAME KEYWORDS! → Both get score boost for same selectors
```

**Result:**
- Both steps find `data-opencreatedialogdropdown` (score=87)
- Both steps find `data-dropdownentitiesname` (score=84)
- Only difference is priority (25 vs 30)
- **Very fragile!** One priority change breaks everything

### Issue 3: Doesn't Check Existence First

**Current Flow:**
```
1. Score all 695 Teststep selectors (SLOW)
2. Pick highest score
3. Check if it exists on page
4. If count=0, fail and go to L2
```

**Better Flow:**
```
1. Filter to only selectors that exist on page (count > 0)
2. Score only those ~10 selectors (FAST)
3. Pick highest score
4. Execute
```

### Issue 4: No Learning from History

**Current:**
```
Run 1: Step 5 uses wrong selector → fail
Run 2: Step 5 uses wrong selector → fail (SAME MISTAKE!)
Run 3: Manual JSON fix → works
Run 4: Keyword change breaks it again → fail
```

**Better:**
```
Run 1: Step 5 uses selector A → fail, uses selector B → success
       → Record: Step 5 + selector B = SUCCESS

Run 2: Step 5 → check history → selector B worked before → boost score
       → Uses selector B immediately
```

---

## BETTER SOLUTIONS (From Yesterday's Discussion)

### Solution 1: Check Page State First ✅ IMMEDIATE FIX

```python
def find_best_selector(self, step_text, ticket_id, step_num):
    keywords = self._extract_keywords(step_text)

    # NEW: Filter by existence FIRST
    candidates = []
    for s in self.selectors:
        if s.get('module') != 'Teststep':
            continue

        selector_str = self.build_selector(s)
        count = self.page.locator(selector_str).count()

        if count > 0:  # ← ONLY score selectors that exist!
            candidates.append(s)

    # Now score only ~10 selectors instead of 695
    for s in candidates:
        s['score'] = self._calculate_score(s, keywords)

    return max(candidates, key=lambda s: s['score'])
```

**Benefits:**
- ✅ Eliminates selectors with count=0 before scoring
- ✅ Fixes Step 5 issue automatically (opencreatedialogdropdown won't be scored because count=0)
- ✅ 70x faster (score 10 instead of 695)

### Solution 2: Semantic Similarity (No Keywords!) ✅ LONG-TERM FIX

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# At startup: Embed all selector labels
for selector in selectors:
    label = selector['context'] + [selector['attr'], selector['value']]
    label_text = " ".join(label)
    selector['embedding'] = model.encode(label_text)

# At runtime: Find best match
step_embedding = model.encode(step_text)

for selector in selectors:
    similarity = cosine_similarity(step_embedding, selector['embedding'])
    selector['score'] = similarity * 100

best = max(selectors, key=lambda s: s['score'])
```

**How It Works:**
```
Step text: "Select 'MyProject' from dropdown"
Embedding: [0.23, -0.45, 0.67, ...] (384 dimensions)

Selector 1: "dropdown option project entities myproject"
Embedding: [0.21, -0.43, 0.69, ...]
Similarity: 0.94 (94% match!) ✅

Selector 2: "create dialog structurelevel opendialog"
Embedding: [-0.12, 0.34, -0.23, ...]
Similarity: 0.31 (31% match) ❌
```

**Benefits:**
- ✅ No keyword extraction needed
- ✅ Understands "select", "choose", "pick" are similar
- ✅ Understands "MyProject" is more similar to "project entities" than "create dialog"
- ✅ Works for ANY new element type without code changes
- ✅ No manual priority tuning

### Solution 3: Usage History ✅ LEARNING SYSTEM

```python
# selector_history.json
{
    "RBPLCD-8862_Step4": {
        "data-opencreatedialogdropdown": {
            "success": 10,
            "fail": 0,
            "last_used": "2024-11-04",
            "avg_score": 87
        }
    },
    "RBPLCD-8862_Step5": {
        "data-dropdownentitiesname": {
            "success": 10,
            "fail": 0,
            "last_used": "2024-11-04",
            "avg_score": 51
        },
        "data-opencreatedialogdropdown": {
            "success": 0,
            "fail": 5,
            "last_used": "2024-11-04",
            "avg_score": 25
        }
    }
}

# In scoring:
history_key = f"{ticket_id}_Step{step_num}"
if history_key in self.history:
    success_rate = self.history[history_key][selector['attr']]['success_rate']
    score += success_rate * 50  # BIG boost for known working selectors
```

**Benefits:**
- ✅ Self-correcting over time
- ✅ Learns which selectors work for which steps
- ✅ No manual tuning needed after first run

---

## VISUAL FLOW DIAGRAM

```
USER INPUT: "Select 'MyProject' from dropdown"
│
├─► PHASE 1: KEYWORD EXTRACTION (selector_loader_v2.py:222-284)
│   │
│   ├─ Rule: "dropdown" in text? YES → add ['dropdown', 'select', 'type']
│   ├─ Rule: "project" in text? YES → add ['project', 'product']
│   │
│   └─► KEYWORDS: ['dropdown', 'select', 'type', 'project', 'product']
│
├─► PHASE 2: L1 SELECTOR SCORING (selector_loader_v2.py:132-220)
│   │
│   ├─ Filter: Only Teststep module selectors → 695 selectors
│   │
│   ├─ For each selector:
│   │   ├─ Match keywords in attr/value/label → +5 per match
│   │   ├─ Match keywords in context → +8 per match
│   │   └─ Add priority → +0 to +30
│   │
│   ├─ Selector A (data-dropdownentitiesname):
│   │   └─ Score: 5 + 16 + 30 = 51 ✅ HIGHEST
│   │
│   ├─ Selector B (data-opencreatedialogdropdown):
│   │   └─ Score: 5 + 0 + 20 = 25
│   │
│   └─► BEST: data-dropdownentitiesname (score=51)
│
├─► PHASE 3: L1 EXECUTION (step_executor.py:242-349)
│   │
│   ├─ Build: [data-dropdownentitiesname='MyProject']
│   │
│   ├─ Check count: page.locator(...).count()
│   │
│   ├─ IF count == 1 → ✅ Execute action (click)
│   ├─ IF count > 1  → ❌ Ambiguous, go to L2
│   └─ IF count == 0 → ❌ Not found, go to L2
│
├─► PHASE 4: L2 GENERIC PATTERNS (step_executor.py:351-500)
│   │   [Only runs if L1 fails]
│   │
│   └─ Use hardcoded HTML patterns: mat-select, mat-option
│
└─► PHASE 5: L3 CV-GUIDED (step_executor.py:502-600)
    │   [Only runs if L1 and L2 fail]
    │
    └─ Use Azure Vision API + OCR
```

---

## ANSWER TO YOUR QUESTION

### Q: "Where are you using keywords?"
**A:** Keywords are ONLY used in L1 scoring (selector_loader_v2.py:194-208) to:
1. Match against selector attr/value/label fields
2. Match against selector context array
3. Calculate a score to pick the best selector

Keywords are NOT used in:
- L1 execution (just checks count)
- L2 (uses hardcoded HTML patterns)
- L3 (uses Computer Vision)

### Q: "Why are you using keywords?"
**A:** Keywords are the current method to match step text to selectors because:
1. **Simple to implement** - just string matching
2. **Worked initially** - for basic cases like "click save" → "save" keyword
3. **No dependencies** - doesn't require ML models or embeddings

### Q: "Why is it not scalable?"
**A:** Because:
1. **Hard-coded rules** - need code changes for every element type
2. **Keyword conflicts** - different steps generate same keywords
3. **No context understanding** - "select MyProject" vs "create project" both have "project"
4. **No learning** - makes same mistakes every time
5. **Scores before checking existence** - wastes time on non-existent selectors

---

## RECOMMENDATION

**Immediate (Today):**
1. Implement page state filtering (Solution 1)
   - Filter count > 0 before scoring
   - 2 hours of work
   - Fixes Step 5 issue

**Short-term (This Week):**
2. Add usage history tracking (Solution 3)
   - Learn from successful runs
   - 1 day of work
   - Self-correcting system

**Long-term (Next Week):**
3. Replace keywords with embeddings (Solution 2)
   - No more manual keyword rules
   - 2 days of work (including testing)
   - Future-proof solution

---

*Document created: November 5, 2024*
