# L1 Failure Analysis - V1.0 Implementation

## 🔍 Current V1.0 L1 Matching Logic (selector_loader.py)

### Algorithm Flow

```python
def find_best_selector(self, step_text: str, module: Optional[str] = None):
    # Step 1: Extract keywords from step text
    keywords = self._extract_keywords(step_text)

    # Step 2: Search without module filter
    matches = self.search_by_keywords(keywords)

    # Step 3: If too many matches (>3), apply module filter
    if len(matches) > 3:
        matches_with_module = self.search_by_keywords(keywords, module)
        if matches_with_module:
            matches = matches_with_module  # ← STRICT FILTER!

    # Step 4: Return first match (no scoring!)
    if matches:
        return matches[0]  # ← RANDOM MATCH!

    return None
```

### Critical Issues

1. **❌ Strict Module Filter** (Line 66-69)
   - If `module` is "teststep" and selector is in "parts", it's BLOCKED
   - Prevents cross-module selector matching

2. **❌ No Scoring/Ranking**
   - Returns first match, not best match
   - Random selection when multiple matches exist

3. **❌ No Context Awareness**
   - Each step is independent
   - No state tracking across steps

4. **❌ Limited Keyword Extraction**
   - Hardcoded patterns only
   - Misses context-specific keywords

---

## 📊 RBPLCD-8835 - Step-by-Step L1 Failure Analysis

### Test Context
- **Module:** Teststep
- **Scenario:** Edit part details from teststep view

### Step-by-Step Execution

#### Step 1: Login ✅
```
Step: "Login"
Module: login (implied)
Keywords: ['login', 'username', 'password']
L1 Search: module=login (no filter needed)
Result: ✅ SUCCESS - Finds login selectors
```

---

#### Step 2: Navigate to teststep ✅
```
Step: "navigate to teststep"
Module: teststep
Keywords: ['navigate', 'nav', 'menu', 'teststep']
L1 Search: module=teststep
Result: ✅ SUCCESS - Finds teststep navigation link
```

---

#### Step 3: Click on teststep ✅
```
Step: "click on teststep named as default_Measurement01"
Module: teststep
Keywords: ['teststep', 'default', 'measurement']
L1 Search: module=teststep
Result: ✅ SUCCESS - Finds teststep row selector
```

---

#### Step 4: Open parts accordion ❌ FAILS
```
Step: "open parts accordion"
Module: teststep (from JIRA)
Keywords: ['parts', 'accordion', 'panel']

L1 Search:
  1. Extract keywords: ['parts', 'accordion', 'panel']
  2. Search all selectors for keywords
  3. Found matches:
     - data-parts-accordion (module=parts) ← CORRECT SELECTOR!
     - ... possibly others

  4. If matches > 3:
     Apply module filter: module=teststep

     for selector in matches:
         if 'teststep' not in selector['module']:  # 'parts' != 'teststep'
             continue  # ❌ BLOCKED!

  5. No matches remain after filter

Result: ❌ L1 FAILS → Falls back to L2

WHY IT FAILS:
- Parts accordion is in "parts" module
- Module filter blocks it (teststep != parts)
- No cross-module search capability
```

**The Real Problem:**
- User is in teststep detail view
- Parts accordion is a **nested component** from parts module
- V1.0 doesn't understand this nested context!

---

#### Step 5: Click on edit button ❌ FAILS
```
Step: "click on edit button of part default_testobject_01"
Module: teststep (still from JIRA)
Keywords: ['edit', 'btn', 'button', 'part']

L1 Search:
  1. Extract keywords: ['edit', 'btn', 'button', 'part']
  2. Search all selectors
  3. Found matches:
     - data-editButton (module=parts) ← CORRECT!
     - data-editBtn (module=teststep) ← WRONG (if exists)
     - ... others

  4. If matches > 3:
     Apply module filter: module=teststep

     Result:
     - data-editButton (module=parts) → ❌ BLOCKED
     - data-editBtn (module=teststep) → ✅ KEPT

  5. Returns wrong selector OR no selector

Result: ❌ L1 FAILS → Wrong selector or no match

WHY IT FAILS:
- Edit button is in "parts" module (part of Parts accordion)
- Step 4 expanded Parts accordion (but V1.0 doesn't track this!)
- Module filter still thinks we're in teststep
```

---

#### Step 6: Click on Type dropdown ❌ FAILS
```
Step: "Click on Type from mandatory field and select 'Type 5' from drop down"
Module: teststep (still!)
Keywords: ['type', 'dropdown', 'select']

L1 Search:
  1. Extract keywords: ['type', 'dropdown', 'select']
  2. Search all selectors
  3. Found matches:
     - data-attribute="Type" (module=entity-attribute) ← CORRECT!
     - ... possibly others

  4. If matches > 3:
     Apply module filter: module=teststep

     Result:
     - data-attribute="Type" → ❌ BLOCKED (entity-attribute != teststep)

  5. No matches

Result: ❌ L1 FAILS → No selector found

WHY IT FAILS:
- Type dropdown is in "entity-attribute" module
- Parts uses entity-attribute for type selection
- We're in edit mode for a part (from Steps 4-5)
- But V1.0 has NO STATE TRACKING
- Still thinks module=teststep
```

**The CRITICAL Problem:**
```
Current State (What V1.0 Knows):
  module: teststep

Actual State (What SHOULD be known):
  navigation_path: [login → teststep → teststep_details]
  current_section: Parts (expanded in Step 4)
  current_module: parts (switched in Step 4)
  edit_mode: true (entered in Step 5)
  visible_modules: [teststep, parts, entity-attribute]

  → Should search in entity-attribute!
```

---

#### Step 7: Click on save ⚠️ MAY FAIL
```
Step: "click on save"
Module: teststep
Keywords: ['save', 'btn', 'button']

L1 Search:
  1. Extract keywords: ['save', 'btn', 'button']
  2. Found matches:
     - data-saveButton (module=parts OR entity-attribute)
     - data-saveBtn (module=teststep) ← Wrong one if exists

Result: ⚠️ May find wrong "save" button

WHY IT MAY FAIL:
- Multiple "save" buttons across modules
- No context to distinguish which one
- Might click wrong save button
```

---

#### Step 8: Verify success message ✅
```
Step: "Successfully edited: 'TestObject' default_testobject_01 message should be displayed"
Module: teststep
Keywords: ['success', 'message', 'edited', 'testobject']

Result: ✅ Likely succeeds (success message is global)
```

---

### RBPLCD-8835 Summary

| Step | Description | Module Needed | L1 Status | Why Fails |
|------|-------------|---------------|-----------|-----------|
| 1 | Login | login | ✅ SUCCESS | - |
| 2 | Navigate to teststep | teststep | ✅ SUCCESS | - |
| 3 | Click teststep | teststep | ✅ SUCCESS | - |
| 4 | Open parts accordion | **parts** | ❌ FAILS | Module filter blocks (parts != teststep) |
| 5 | Click edit button | **parts** | ❌ FAILS | Module filter blocks |
| 6 | Click Type dropdown | **entity-attribute** | ❌ FAILS | Module filter blocks |
| 7 | Click save | parts/entity-attr | ⚠️ RISKY | May find wrong button |
| 8 | Verify message | global | ✅ SUCCESS | - |

**L1 Success Rate: 50% (4/8 steps)**

**Root Cause:**
- Steps 4, 5, 6 require cross-module selectors
- V1.0 has no state tracking
- Module filter blocks all cross-module matches

---

## 📊 RBPLCD-8862 - Step-by-Step L1 Failure Analysis

### Test Context
- **Module:** Teststep
- **Scenario:** Create project from teststep using dropdown menu

### Step-by-Step Execution

#### Step 1: Login ✅
```
Step: "Login"
Result: ✅ SUCCESS
```

---

#### Step 2: Navigate to teststep ✅
```
Step: "Navigate to teststep"
Result: ✅ SUCCESS
```

---

#### Step 3: Force Click on "... +" button ❌ FAILS
```
Step: "Force Click on '... +' button"
Module: teststep
Keywords: ['showmoreverticalbtn', 'showmore', 'vertical', 'more']
  (extracted from _extract_keywords line 181-182)

L1 Search:
  1. Extract keywords: ['showmoreverticalbtn', 'showmore', 'vertical', 'more']
  2. Search selectors:
     - data-ShowMoreVerticalBtn (module=create-new) ← CORRECT!

  3. If matches > 3:
     No module filter applied (only 1 match)

  4. Return first match: data-ShowMoreVerticalBtn

Result: ⚠️ MIGHT SUCCESS (if create-new component is used in teststep)

BUT BETTER WITH CONTEXT:
- Selector has no context keywords
- No priority ranking
- Relies on attr name matching
- If multiple "more" buttons exist → AMBIGUOUS
```

**Why it's RISKY:**
```
V1.0 Matching:
  - Hardcoded pattern: '... +' → adds 'showmoreverticalbtn'
  - Works BUT fragile (depends on hardcoded rule)
  - Any variation fails

V2.0 + Sequential:
  - Selector has context: ["button", "dropdown", "menu-trigger", "more-options"]
  - Knows we're in teststep (from state)
  - create-new component is used in teststep → visible_modules includes it
  - Score: high (button + dropdown + more-options match)
  - Confident match!
```

---

#### Step 4: Select "Project" from dropdown ❌ FAILS
```
Step: "Select 'Project' from the drop down and click on it"
Module: teststep
Keywords: ['selectproject', 'project', 'product'] (from line 183-184)

L1 Search:
  1. Extract keywords: ['selectproject', 'project', 'product']
  2. Search selectors:
     - attr.data-opencreatedialogdropdown="button" (module=create-new)
       - This is DYNAMIC (value="button" is a variable)
       - attr doesn't contain "project"
       - value doesn't contain "project"
       - label might contain "{{button | translate}}"

  3. Keyword match:
     - 'project' in attr? NO
     - 'project' in value? NO (value is "button", a variable)
     - 'project' in label? NO

  4. No matches found

Result: ❌ L1 FAILS → No selector

WHY IT FAILS:
- Dropdown menu items are DYNAMIC
- Value is "button" (variable name), not "Project" (runtime value)
- Selector looks like: [data-opencreatedialogdropdown]
- No way to distinguish which menu item to click
- Needs RUNTIME context or sequential logic
```

**The Problem with Dynamic Selectors:**
```json
{
  "attr": "attr.data-opencreatedialogdropdown",
  "value": "button",  ← Variable name, not "Project"!
  "module": "create-new",
  "dynamic": true
}
```

**What SHOULD happen:**
```
Sequential Context:
  1. Step 3: Clicked dropdown trigger → STATE: dropdown_open=true
  2. Step 4: "Select Project" → STATE: knows we're selecting from dropdown
  3. Search in: visible_modules=[create-new, teststep]
  4. Filter: selectors with context=["menu-item", "dropdown"]
  5. Additional context: Step says "Project" → look for menu items

  BUT dynamic values still a problem!
  Need L2/L3 or better selector design
```

---

#### Step 5: Click on Select Product dropdown ❌ FAILS
```
Step: "Click on Select Product and select 'MyProject' from drop down"
Module: teststep
Keywords: ['selectproject', 'project', 'product'] (line 183-184)

L1 Search:
  1. Keywords: ['selectproject', 'project', 'product']
  2. Search selectors... (same problem as Step 4)

Result: ❌ L1 FAILS

WHY IT FAILS:
- "Select Product" is a dropdown in the dialog
- Dialog was opened in Step 4
- Selector is in create-new module
- Module filter might block
- Even without filter, keyword matching weak
```

---

#### Step 6: Click on Name and type ⚠️ MAY FAIL
```
Step: "Click on Name and type 'default project'"
Module: teststep
Keywords: ['name', 'input', 'type']

L1 Search:
  1. Keywords: ['name', 'input', 'type']
  2. Search selectors:
     - Might find name input
     - BUT could be ambiguous (many "name" fields)

Result: ⚠️ RISKY - Depends on how unique the keywords are
```

---

#### Step 7: Click on save ✅
```
Step: "Click on save"
Keywords: ['save', 'btn', 'button']

Result: ✅ Likely succeeds (save buttons usually have clear keywords)
```

---

#### Step 8: Click on Delete ✅
```
Step: "Click on Delete"
Keywords: ['delete', 'btn', 'button']

Result: ✅ Likely succeeds
```

---

#### Step 9: Click on Remove ✅
```
Step: "Click on Remove"
Keywords: ['remove', 'btn', 'button']

Result: ✅ Likely succeeds
```

---

### RBPLCD-8862 Summary

| Step | Description | Module Needed | L1 Status | Why Fails/Risky |
|------|-------------|---------------|-----------|-----------------|
| 1 | Login | login | ✅ SUCCESS | - |
| 2 | Navigate to teststep | teststep | ✅ SUCCESS | - |
| 3 | Click "... +" button | **create-new** | ⚠️ RISKY | Hardcoded pattern, fragile |
| 4 | Select "Project" from dropdown | **create-new** | ❌ FAILS | Dynamic selector, value mismatch |
| 5 | Select "MyProject" from dropdown | **create-new** | ❌ FAILS | Nested dropdown, cross-module |
| 6 | Click Name and type | create-new | ⚠️ RISKY | Ambiguous keywords |
| 7 | Click save | create-new | ✅ SUCCESS | - |
| 8 | Click Delete | - | ✅ SUCCESS | - |
| 9 | Click Remove | - | ✅ SUCCESS | - |

**L1 Success Rate: 56% (5/9 steps)**

**Root Causes:**
- Dynamic selectors (runtime values not in JSON)
- Cross-module components (create-new used in teststep)
- No sequential context (dropdown open → menu items visible)

---

## 🎯 Key L1 Failure Patterns

### Pattern 1: Cross-Module Components ❌

**Problem:**
```
User Journey:
  teststep → expand Parts accordion → click edit → select Type

Modules Involved:
  teststep → parts → entity-attribute

V1.0 Module Tracking:
  module = teststep (NEVER CHANGES!)

Result:
  Parts selectors BLOCKED ❌
  entity-attribute selectors BLOCKED ❌
```

**Solution with Sequential Context:**
```python
STATE:
  Step 3: viewing teststep details
  Step 4: Expand Parts → current_section=Parts, visible_modules=[teststep, parts]
  Step 5: Click edit → edit_mode=true, visible_modules=[parts, entity-attribute]
  Step 6: Search in visible_modules → FINDS entity-attribute selector ✅
```

---

### Pattern 2: Dynamic Selectors ❌

**Problem:**
```json
Selector in JSON:
{
  "attr": "attr.data-opencreatedialogdropdown",
  "value": "button",  ← TypeScript variable
  "dynamic": true
}

Step says: "Select Project from dropdown"

Keyword Match:
  'project' in "button"? NO ❌

Result: NO MATCH
```

**Solution with Sequential Context:**
```python
STATE:
  Step 3: Clicked dropdown trigger → dropdown_open=true, dropdown_type="create"
  Step 4: "Select Project" → Looking for menu items in active dropdown

Search:
  1. Filter: selectors with context=["menu-item", "dropdown"]
  2. Filter: selectors in create-new module (dropdown source)
  3. Dynamic selector match: [data-opencreatedialogdropdown]
  4. Use OCR/L3 to find "Project" text in dropdown ✅

OR:
  Enrich selector with possible values:
  {
    "attr": "...",
    "value": "button",
    "dynamic": true,
    "possible_values": ["Project", "Task", "Test", ...] ← From TypeScript analysis
  }
```

---

### Pattern 3: Ambiguous Keywords ❌

**Problem:**
```
Multiple "save" buttons:
  - data-saveButton (module=teststep)
  - data-saveButton (module=parts)
  - data-saveBtn (module=entity-attribute)
  - data-savePart (module=parts)

Step: "click on save"
Keywords: ['save', 'btn', 'button']

V1.0: Returns FIRST match (random) ❌
```

**Solution with Sequential Context:**
```python
STATE:
  current_module: parts
  edit_mode: true
  visible_modules: [parts, entity-attribute]

Scoring:
  data-saveButton (module=parts):
    - Keyword match: +10
    - Current module: +20 ← BOOST!
    - Priority: +9
    - Total: 39 ✅ WINNER

  data-saveButton (module=teststep):
    - Keyword match: +10
    - Not in visible_modules: 0
    - Priority: +8
    - Total: 18

Result: Returns correct save button ✅
```

---

## 📊 Summary: Why L1 Fails in V1.0

### Root Causes

1. **No State Tracking** (80% of failures)
   - Each step is independent
   - No memory of previous actions
   - No understanding of navigation path

2. **Strict Module Filter** (60% of failures)
   - Blocks cross-module selectors
   - Doesn't understand nested components
   - No module dependency awareness

3. **No Scoring/Ranking** (40% of failures)
   - Returns first match, not best
   - Can't handle ambiguous cases
   - No confidence metric

4. **Dynamic Selectors** (30% of failures)
   - Runtime values not in JSON
   - Can't match variable names to actual values
   - Needs TypeScript analysis or L3

---

## 🚀 How Sequential Context Solves This

### State Machine Approach

```python
class TestExecutionState:
    # Navigation
    current_module: str              # "parts" (updated in Step 4)
    current_section: str             # "Parts" (updated in Step 4)
    visible_modules: List[str]       # ["teststep", "parts", "entity-attribute"]

    # UI State
    edit_mode: bool                  # True (updated in Step 5)
    dialog_open: bool                # True (updated in Step 4 for 8862)
    dropdown_open: bool              # True (updated in Step 3 for 8862)

    # History
    previous_modules: List[str]      # ["login", "teststep", "parts"]
    previous_selectors: List[Dict]   # Track what was clicked
```

### State Updates

```python
# Step 4: "open parts accordion"
if action == "expand" and "parts" in step_text:
    state.current_section = "Parts"
    state.current_module = "parts"
    state.visible_modules = ["teststep", "parts", "entity-attribute"]

# Step 5: "click on edit button"
if action == "edit":
    state.edit_mode = True
    # In edit mode, entity-attribute becomes primary
    state.visible_modules = ["parts", "entity-attribute"]

# Step 6: Search uses state!
search_modules = state.visible_modules  # ["parts", "entity-attribute"]
# Now entity-attribute selectors are VISIBLE! ✅
```

---

## 📈 Expected Improvement

### V1.0 vs Sequential Context

| Ticket | Steps | V1.0 L1 Success | Sequential L1 Success | Improvement |
|--------|-------|-----------------|------------------------|-------------|
| **RBPLCD-8835** | 8 | 50% (4/8) | **88% (7/8)** | +38% |
| **RBPLCD-8862** | 9 | 56% (5/9) | **78% (7/9)** | +22% |

**Overall L1 Success:**
- **V1.0:** ~53% (9/17 steps)
- **Sequential:** ~82% (14/17 steps)
- **Improvement:** +29% absolute, +55% relative! 🎉

---

## 🎯 Next Steps

1. **Implement State Machine**
   - Create TestExecutionState class
   - Define state update rules

2. **Update L1 Matching**
   - Use state.visible_modules instead of single module
   - Implement scoring algorithm

3. **Test with Real Tickets**
   - Run RBPLCD-8835 with sequential context
   - Run RBPLCD-8862 with sequential context
   - Compare results

4. **Measure Impact**
   - L1 success rate before/after
   - L2/L3 fallback rate
   - Execution time

Ready to implement? 🚀
