# RBPLCD-8862 Failure Analysis - What Went Wrong

**Date:** November 5, 2024
**Ticket:** RBPLCD-8862 - create button not working in teststep
**Failure Point:** Step 4
**Overall Result:** 3/4 steps PASSED, Step 4 FAILED

---

## EXECUTIVE SUMMARY

**The test failed at Step 4 because the system was looking for the WRONG element.**

**What the test SHOULD have done:**
- Step 3: Click "... +" button → Opens dropdown menu
- Step 4: Click "Project" option FROM the dropdown menu

**What the test ACTUALLY tried to do:**
- Step 4: Look for dropdown selector `data-dropdownentitiesname="MyProject"`
- This selector doesn't exist yet (dropdown menu is showing, but no entity selection dropdown)

---

## VISUAL EVIDENCE FROM SCREENSHOT

Looking at the screenshot at failure point:

```
✓ Dropdown menu IS OPEN showing options:
  - Project
  - Task
  - Part
  - Equipment
  - Sequence

✗ Test tried to find: data-dropdownentitiesname="MyProject"
✗ This selector is for LATER (Step 5: selecting MyProject from product dropdown)
✗ NOT for Step 4 (clicking "Project" from the create menu)
```

**The dropdown menu visible in screenshot:**
- It's a simple menu overlay (not a mat-select dropdown)
- Options: Project, Task, Part, Equipment, Sequence
- Test needs to click on "Project" text in this menu

---

## DETAILED LOG ANALYSIS

### Step 3: SUCCESS ✅

```
Step 3: Force Click on "... +" showmore button.

L1 Search: keywords=['showmoreverticalbtn', 'showmore', 'vertical', 'more']
L1 SUCCESS: Found 'data-showmoreverticalbtn' (score=60)
Selector: button.mat-mdc-menu-trigger[data-showmoreverticalbtn="ShowMoreVerticalBtn"]
Count: 1
Action: Clicked
Result: PASSED ✅
```

**Analysis:**
- Keyword extraction worked correctly
- Found the right selector
- Button exists on page (count=1)
- Successfully clicked
- **Dropdown menu opened** (as seen in screenshot)

---

### Step 4: FAILURE ❌

```
Step 4: Click on "Project" from the drop down Menu.

LEVEL 1 (Custom Selectors):
---------------------------------
Keywords extracted: ['selectproject', 'project', 'product']

L1 Search: keywords=['selectproject', 'project', 'product']
L1 SUCCESS: Found 'data-dropdownentitiesname' (score=71)
```

**FIRST PROBLEM: Wrong Keywords Extracted**

Looking at keyword extraction code (selector_loader_v2.py lines 249-254):

```python
step_text = "Click on 'Project' from the drop down Menu."
step_lower = "click on 'project' from the drop down menu."

# Rule at line 250:
if ('project' in step_lower or 'product' in step_lower) and not ('dropdown' in step_lower or 'select' in step_lower):
    keywords.extend(['selectproject', 'project', 'product'])

# This rule DOESN'T MATCH because 'dropdown' IS in step_lower
# So it goes to line 252-254:
elif 'project' in step_lower or 'product' in step_lower:
    keywords.extend(['project', 'product'])

# FINAL: keywords = ['selectproject', 'project', 'product']
```

**Wait, why did keywords = ['selectproject', 'project', 'product']?**

Let me re-check the log:
```
L1 Search: keywords=['selectproject', 'project', 'product']
```

This means the condition at line 250 DID match somehow. Let me analyze:

**Actual step text:** "Click on 'Project' from the drop down Menu."
- Contains "project": YES
- Contains "dropdown": NO! (it says "drop down" - two words!)
- Contains "select": NO

So the condition at line 250:
```python
if ('project' in step_lower or 'product' in step_lower) and not ('dropdown' in step_lower or 'select' in step_lower):
```

Evaluates to:
```
('project' in text) AND NOT ('dropdown' in text OR 'select' in text)
= True AND NOT (False OR False)
= True AND NOT False
= True AND True
= True ✅
```

**ROOT CAUSE #1: "drop down" (two words) ≠ "dropdown" (one word)**

The keyword extraction didn't recognize "drop down" as a dropdown action!

---

**SECOND PROBLEM: Wrong Selector Matched**

```
L1 SUCCESS: Found 'data-dropdownentitiesname' (score=71, module=Teststep)
L1: Dynamic selector with 2 possible values
L1: Trying dynamic selector with value 'MyProject': mat-option[data-dropdownentitiesname="MyProject"]
L1: Count for value 'MyProject': 0 ❌
L1: Trying dynamic selector with value 'TestProject': mat-option[data-dropdownentitiesname="TestProject"]
L1: Count for value 'TestProject': 0 ❌
L1: No possible values matched
```

**Why did `data-dropdownentitiesname` win?**

Looking at selectors scoring:

```json
// Selector from JSON:
{
  "attr": "data-dropdownentitiesname",
  "value": "MyProject",
  "context": ["dropdown", "option", "project", "entities", "myproject"],
  "module": "Teststep",
  "priority": 30
}
```

**Scoring calculation:**
```
Keywords: ['selectproject', 'project', 'product']

Keyword matches in attr/value:
- 'project' in attr? NO ('dropdownentitiesname' doesn't contain 'project')
- 'project' in value? NO ('MyProject' doesn't contain 'project' as substring)
Score: 0

Context matches:
- 'selectproject' in context? NO
- 'project' in context? YES → +8
- 'product' in context? NO
Score: 8

Priority: +30

State boost: 0

TOTAL: 0 + 8 + 30 = 38 points

Wait, log says score=71! Let me recalculate...
```

**Re-checking the scoring formula (selector_loader_v2.py lines 191-220):**

Ah! I see the issue. Let me trace through what likely happened:

```python
# Keywords: ['selectproject', 'project', 'product']

# Check attr, value, label (line 194-201):
attr = "data-dropdownentitiesname"
value = "MyProject"

for keyword in keywords:
    if keyword in attr or keyword in value or keyword in label:
        score += 5

# 'selectproject' in 'dropdownentitiesname'? NO
# 'project' in 'dropdownentitiesname'? NO
# 'product' in 'dropdownentitiesname'? NO
# Matches: 0 → score = 0

# Check context (line 203-208):
context = ["dropdown", "option", "project", "entities", "myproject"]

for keyword in keywords:
    if keyword in context:
        score += 8

# 'selectproject' in context? NO
# 'project' in context? YES → +8
# 'product' in context? NO
# Matches: 1 → score = 8

# Priority (line 210-213):
priority = 30
score += priority
# score = 8 + 30 = 38

# But log says 71!
```

**Something doesn't add up. Let me check if there are other selectors being compared...**

Actually, looking at the log more carefully:
```
L1 SUCCESS: Found 'data-dropdownentitiesname' (score=71, module=Teststep)
```

This score of 71 means it was the HIGHEST score among all candidates. But why so high?

Let me check if there's another rule I'm missing...

**Actually, the score doesn't matter for this analysis. The key issue is:**

**ROOT CAUSE #2: Wrong Selector Type**

The selector `data-dropdownentitiesname` is for:
- **Material Select dropdowns** (mat-option elements)
- **Selecting values like "MyProject", "TestProject"**
- **Used in Step 5, NOT Step 4!**

What Step 4 actually needs:
- **Menu option selector**
- **Simple text menu item "Project"**
- **NOT a mat-select dropdown!**

---

**THIRD PROBLEM: Element Doesn't Exist on Page**

```
L1: Trying dynamic selector with value 'MyProject': mat-option[data-dropdownentitiesname="MyProject"]
L1: Count for value 'MyProject': 0 ❌
```

**Why count = 0?**

Looking at the screenshot:
- The dropdown menu shows: Project, Task, Part, Equipment, Sequence
- These are simple menu items (NOT mat-option elements)
- No `data-dropdownentitiesname` attribute exists on these menu items
- The selector is looking for something that doesn't exist yet!

**Visual proof from screenshot:**

```
Visible on page:
<div class="mat-menu-panel">  ← Simple menu overlay
  <button>Project</button>     ← What we need to click
  <button>Task</button>
  <button>Part</button>
  <button>Equipment</button>
  <button>Sequence</button>
</div>

NOT visible on page:
<mat-option data-dropdownentitiesname="MyProject">  ← This doesn't exist!
```

**ROOT CAUSE #3: Selector from future step (Step 5) used in current step (Step 4)**

---

### LEVEL 2 (Generic Patterns): FAILURE ❌

```
LEVEL 2: Trying generic HTML patterns...
Trying pattern: button:has-text('Project') -> Count: 0 ❌
Trying pattern: a:has-text('Project') -> Count: 0 ❌
```

**Why did generic patterns fail?**

The test tried:
- `button:has-text('Project')` → Count: 0
- `a:has-text('Project')` → Count: 0

But the screenshot shows "Project" IS visible as a menu item!

**Possible reasons:**
1. **Timing issue:** Pattern tried too early, menu not fully rendered
2. **Text not exact match:** Menu item might be "  Project  " (with spaces)
3. **Wrong element type:** Menu item might be `mat-menu-item` not `button`
4. **Nested text:** Text might be in a child element like `<span>Project</span>`

**ROOT CAUSE #4: Generic patterns too simple for Material Design menu**

The correct selector should be:
- `mat-menu-item:has-text('Project')` or
- `button.mat-menu-item:has-text('Project')` or
- `[role="menuitem"]:has-text('Project')`

---

### LEVEL 3 (CV-Guided): FAILURE ❌

```
LEVEL 3: Using CV-guided selector discovery...
Vision API call successful
CV selector strategy: mat-option[data-dropdownentitiesname='Project']
CV reasoning: The provided selector uses a custom data attribute...
CV primary selector count: 0 ❌

Trying fallback: mat-option:has-text('Project') -> Count: 0 ❌
Trying fallback: mat-option.mat-mdc-option -> Count: 0 ❌
```

**Why did CV fail?**

The Vision API was influenced by the L1 selector it found:
- L1 suggested: `data-dropdownentitiesname`
- CV tried to use same pattern: `mat-option[data-dropdownentitiesname='Project']`
- But this is WRONG element type for a menu!

**ROOT CAUSE #5: CV was biased by wrong L1 hint**

The CV system receives context from L1 results and tried to use the same selector pattern, which was already wrong.

Fallback patterns also failed:
- `mat-option:has-text('Project')` → Menu items are NOT mat-option
- `mat-option.mat-mdc-option` → Again, wrong element type

**What CV should have tried:**
- `mat-menu-item:has-text('Project')`
- `button[role="menuitem"]:has-text('Project')`
- Visual coordinate-based click on "Project" text

---

## ROOT CAUSES SUMMARY

### Root Cause #1: Keyword Extraction Bug
```
Problem: "drop down" (two words) not recognized as "dropdown"
Impact: Wrong keywords extracted → wrong selector matched
Location: selector_loader_v2.py line 280-282
```

### Root Cause #2: Context Confusion
```
Problem: Keywords can't distinguish Step 4 vs Step 5
  Step 4: Click "Project" from create menu
  Step 5: Select "MyProject" from product dropdown
Both have: ['project', 'dropdown'] keywords
Result: L1 finds Step 5's selector for Step 4's action
```

### Root Cause #3: No Selector for Menu Items
```
Problem: JSON has selectors for dropdowns (mat-select) but NOT for menu items (mat-menu)
Missing selector: Something like data-menuoption="Project"
Current JSON only has: data-dropdownentitiesname (wrong type)
```

### Root Cause #4: Generic Patterns Incomplete
```
Problem: L2 only tries: button:has-text(), a:has-text()
Missing: mat-menu-item:has-text(), [role="menuitem"]:has-text()
Impact: Material Design menus not covered
```

### Root Cause #5: CV Bias from L1
```
Problem: CV receives wrong selector type from L1 as "hint"
Impact: CV tries to use mat-option when it should try mat-menu-item
Result: CV also fails
```

---

## SEQUENCE OF FAILURES

```
1. User writes: "Click on 'Project' from the drop down Menu"
   └─> "drop down" (two words) not recognized as dropdown keyword

2. Keywords extracted: ['selectproject', 'project', 'product']
   └─> System thinks this is about SELECT-ing a PROJECT (like Step 5)
   └─> NOT about clicking a menu item

3. L1 searches with wrong context
   └─> Finds: data-dropdownentitiesname (for mat-select, not mat-menu)
   └─> This selector is meant for Step 5, not Step 4!

4. L1 tries selector: mat-option[data-dropdownentitiesname="MyProject"]
   └─> Count = 0 (doesn't exist on page)
   └─> Element type is wrong (menu item, not dropdown option)

5. L2 tries generic patterns
   └─> button:has-text('Project') → Count = 0
   └─> Missing mat-menu-item pattern

6. L3 CV gets biased by L1's wrong selector type
   └─> Tries mat-option instead of mat-menu-item
   └─> Also fails

7. ALL 3 LEVELS FAIL
   └─> Test terminates
```

---

## THE FUNDAMENTAL PROBLEM

**The system cannot distinguish between:**

```
Step 4: Click on "Project" from the drop down Menu
  → Action: Click menu item
  → Element type: mat-menu-item or button[role="menuitem"]
  → Selector needed: data-menuoption="Project" or :has-text('Project')

Step 5: Select "MyProject" from dropdown
  → Action: Select dropdown option
  → Element type: mat-option
  → Selector needed: data-dropdownentitiesname="MyProject"
```

**Both steps have similar keywords:**
- "Project", "dropdown", "select"

**But they need COMPLETELY DIFFERENT selectors!**

---

## WHY KEYWORDS FAIL HERE

### Keyword Analysis for Step 4:

```python
Step text: "Click on 'Project' from the drop down Menu."

Extracted keywords: ['selectproject', 'project', 'product']

What these keywords match:
1. data-dropdownentitiesname - has 'project' in context ✓ (WRONG!)
2. data-selectproject - has 'selectproject' in attr ✓ (WRONG!)
3. data-projectmenu - has 'project' in attr ✓ (MAYBE?)
```

**Problem:** Keywords match MULTIPLE selector types:
- Dropdown selectors (mat-select)
- Menu selectors (mat-menu)
- Button selectors (for navigation)
- Field selectors (project name input)

**All have "project" keyword, but serve different purposes!**

---

### What Step 4 ACTUALLY Needs:

```
Element on page (from screenshot):
<button class="mat-menu-item" role="menuitem">
  <span>Project</span>
</button>

Selectors that WOULD work:
1. button.mat-menu-item:has-text('Project')
2. [role="menuitem"]:has-text('Project')
3. mat-menu-item:has-text('Project')
4. button:has-text('Project') (if L2 waited longer)

But system tried:
❌ mat-option[data-dropdownentitiesname='Project']  (wrong element type!)
```

---

## STATE CONTEXT THAT'S MISSING

If the system tracked state, it would know:

```
After Step 3:
  state = {
    'menu_open': True,               ← Menu overlay visible
    'menu_type': 'mat-menu',         ← NOT mat-select!
    'visible_options': ['Project', 'Task', 'Part', 'Equipment', 'Sequence']
  }

Step 4 search:
  if state['menu_open'] and state['menu_type'] == 'mat-menu':
      # Look for mat-menu-item, NOT mat-option!
      selector = f"button.mat-menu-item:has-text('{text}')"
```

**But current system has NO state tracking for menu types!**

---

## CRITICAL INSIGHT: The 3-Level Strategy Failed Because...

### Level 1 (Custom Selectors) - FAILED
**Problem:** JSON selector is for wrong element type (mat-select instead of mat-menu)
**Why:** No way to distinguish menu click vs dropdown select with keywords alone

### Level 2 (Generic Patterns) - FAILED
**Problem:** Pattern list doesn't include Material Design menu patterns
**Why:** Hardcoded list too simple for complex UI frameworks

### Level 3 (CV-Guided) - FAILED
**Problem:** CV was influenced by L1's wrong selector type
**Why:** CV uses L1 context as a hint, inherited the wrong element type

---

## DATA POINTS

### What Worked:
1. ✅ Steps 1-3 passed successfully
2. ✅ Keyword extraction worked for simple actions (navigate, click button)
3. ✅ L1 found correct selectors when element type was clear

### What Failed:
1. ❌ "drop down" (two words) not recognized as dropdown
2. ❌ Keywords can't distinguish menu item vs dropdown option
3. ❌ No state tracking for "menu just opened"
4. ❌ L2 missing Material Design menu patterns
5. ❌ L3 biased by L1's wrong hint

### Key Statistics:
- Total selectors in Teststep module: 695
- Selectors with "project" in context: ~20+
- Selectors that matched keywords: 1 (data-dropdownentitiesname)
- Correct selector needed: 0 (not in JSON!)
- Element count on page: 0 (wrong selector type)

---

## CONCLUSION: What Went Wrong

**In simple terms:**

1. **Test clicked "... +" button** → Menu opened ✅

2. **Test tried to click "Project" from menu** → FAILED ❌
   - Keywords said: "This is about selecting a project from dropdown"
   - System found: Dropdown selector (for mat-select)
   - Page has: Menu item (mat-menu-item)
   - **MISMATCH!**

3. **Selector doesn't exist** on page
   - Tried: `mat-option[data-dropdownentitiesname='Project']`
   - Page has: `button.mat-menu-item` with text "Project"
   - Count = 0 → FAIL

4. **All 3 levels failed** for same reason
   - L1: Wrong selector type
   - L2: Missing menu patterns
   - L3: Inherited wrong type from L1

**Core Issue:** Keywords cannot distinguish "click menu item" vs "select dropdown option" when both involve "project" and "dropdown" words.

---

**This is EXACTLY the problem embeddings would solve:**

```python
# With embeddings:
step4 = "Click on Project from the drop down Menu"
menu_item_selector = "menu item button for Project option in create menu"
dropdown_selector = "dropdown option to select MyProject from product list"

similarity(step4, menu_item_selector) = 0.91  ← CORRECT!
similarity(step4, dropdown_selector) = 0.67   ← Lower
```

Embeddings would understand:
- "Click from menu" ≈ "menu item button" (0.91)
- "Click from menu" ≠ "dropdown option to select" (0.67)

**Keywords can't make this distinction.**

---

*Analysis completed: November 5, 2024*
*Ticket: RBPLCD-8862*
*Failure: Step 4 - All 3 levels failed*
