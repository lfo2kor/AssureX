# Enriched Selector File Structure - Complete Guide

## 🎯 Purpose: Why Each Field Exists

Every field in the enriched selector solves a **specific problem** in test automation.

---

## 📋 Complete Field List with Purpose

### **EXAMPLE: Real Selector from Your Codebase**

```json
{
  // ============================================
  // BASIC IDENTIFICATION FIELDS
  // ============================================

  "id": "selector_042",
  "type": "DYNAMIC_LOOP",

  // ============================================
  // SELECTOR MATCHING FIELDS (L1 Core)
  // ============================================

  "attr": "data-openCreateDialogDropDown",
  "value": "{{button}}",

  // ============================================
  // MODULE & LOCATION FIELDS
  // ============================================

  "module": "CreateNew",
  "filePath": "src/app/create-new/create-new.component.html",
  "lineNumber": 27,

  // ============================================
  // DYNAMIC VALUE FIELDS (Critical!)
  // ============================================

  "isDynamic": true,
  "dynamicType": "LOOP",
  "dynamicVariable": "button",
  "loopSource": "options",
  "possibleValues": [
    "aeName.StructureLevel.name",
    "aeName.Test.name",
    "aeName.TestStep.name",
    "aeName.UnitUnderTest.name",
    "aeName.TestEquipment.name",
    "aeName.TestSequence.name"
  ],
  "dynamicValueSource": "create-new.component.ts:43",
  "loopContext": "@for (button of options; track button)",

  // ============================================
  // CONTEXT ENRICHMENT FIELDS (LLM Matching)
  // ============================================

  "context": [
    "button",
    "menu-item",
    "dropdown-option",
    "create",
    "dialog",
    "clickable",
    "dynamic-content",
    "loop-generated"
  ],

  // ============================================
  // ELEMENT STRUCTURE FIELDS
  // ============================================

  "elementType": "button",
  "angularComponent": "mat-menu-item",
  "angularDirectives": [],
  "parentElement": "mat-menu",
  "parentSelector": "[data-ShowMoreVerticalBtn]",

  // ============================================
  // PRIORITY & SCORING FIELDS
  // ============================================

  "priority": 8,
  "confidence": "high",

  // ============================================
  // SEMANTIC UNDERSTANDING FIELDS
  // ============================================

  "usage_scenario": "Dropdown menu item for creating different entity types (Project, Test, TestStep, Part, Equipment, Sequence). Dynamically generated from options array.",

  "actionIntent": "create_entity",
  "targetType": "menu_item_action",

  // ============================================
  // INTERACTION FIELDS
  // ============================================

  "interactionType": "click",
  "eventHandler": "openCreateDialog(button)",
  "expectedAction": "Opens create dialog for selected entity type",

  // ============================================
  // VISIBILITY & CONDITIONAL FIELDS
  // ============================================

  "conditionalRendering": "@if ((!_detailViewCondition) && (isCreateBtnEnable || options.length))",
  "visibilityContext": "Visible when NOT in detail view AND (create button enabled OR options exist)",
  "requiresPermission": true,
  "permissionCheck": "isCreate*Allowed",

  // ============================================
  // FORM & STATE FIELDS
  // ============================================

  "isFormField": false,
  "formControlName": null,
  "stateDependent": true,
  "requiredState": {
    "detailView": false,
    "createEnabled": true
  },

  // ============================================
  // HTML REFERENCE FIELDS
  // ============================================

  "htmlSnippet": "<button (click)=\"openCreateDialog(button)\" [attr.data-openCreateDialogDropDown]=\"button\" data-openCreateDialogBtnAddIcon=\"addIcon\" mat-menu-item [value]=\"button\">{{button | translate}}</button>",

  "surroundingContext": {
    "before": "<mat-menu #dropdownMenu=\"matMenu\">",
    "after": "</mat-menu>"
  },

  // ============================================
  // METADATA FIELDS
  // ============================================

  "extractedDate": "2025-11-03T15:30:00Z",
  "framework": "Angular",
  "frameworkVersion": "14+",
  "uiLibrary": "Angular Material",

  // ============================================
  // USAGE STATISTICS (Optional - for learning)
  // ============================================

  "usageCount": 0,
  "successRate": 0.0,
  "lastUsed": null,
  "commonStepPatterns": []
}
```

---

## 🔍 Field-by-Field Explanation

### 1. **Basic Identification**

```json
"id": "selector_042"
```
**Purpose:** Unique identifier for tracking and debugging
**Problem Solved:** Can reference specific selector in logs
**Example Use:** "L1 failed for selector_042, check logs"

```json
"type": "DYNAMIC_LOOP"
```
**Purpose:** Categorize selector behavior
**Values:** `STATIC` | `DYNAMIC` | `DYNAMIC_LOOP`
**Problem Solved:** Executor knows how to handle selector
- `STATIC`: Direct match `[data-save="saveBtn"]`
- `DYNAMIC`: Variable value `[data-action="{{actionType}}"]`
- `DYNAMIC_LOOP`: Multiple values `[data-item="{{item}}"]` from array

---

### 2. **Selector Matching Fields** ⭐ **CRITICAL**

```json
"attr": "data-openCreateDialogDropDown"
```
**Purpose:** The HTML attribute to search for
**Problem Solved:** What to look for in the page
**Used By:** L1 selector builder: `[data-openCreateDialogDropDown]`

```json
"value": "{{button}}"
```
**Purpose:** The attribute value (static or dynamic marker)
**Problem Solved:**
- Static: Direct match `[data-save="saveBtn"]`
- Dynamic: Indicates runtime value `[data-action="{{var}}"]`

---

### 3. **Module & Location**

```json
"module": "CreateNew",
"filePath": "src/app/create-new/create-new.component.html",
"lineNumber": 27
```
**Purpose:** Where selector came from
**Problem Solved:**
- **Module filtering** - Search only in visible modules (sequential context)
- **Debugging** - Developer can find exact HTML line
- **Maintenance** - Update selector when code changes

**Example:**
```python
# Sequential Context Usage
if selector['module'] in visible_modules:
    score += 15  # Boost selectors in visible modules
```

---

### 4. **Dynamic Value Fields** ⭐ **SOLVES DYNAMIC SELECTOR PROBLEM**

```json
"isDynamic": true,
"dynamicType": "LOOP",
"dynamicVariable": "button",
"loopSource": "options",
"possibleValues": [
  "aeName.StructureLevel.name",
  "aeName.Test.name",
  "aeName.TestStep.name",
  "aeName.UnitUnderTest.name",
  "aeName.TestEquipment.name",
  "aeName.TestSequence.name"
],
"dynamicValueSource": "create-new.component.ts:43"
```

**Purpose:** Handle selectors with runtime values
**Problem Solved:** The BIGGEST problem in your current system!

#### **Problem Scenario:**

**HTML:**
```html
@for (button of options; track button) {
  <button [attr.data-openCreateDialogDropDown]="button">
}
```

**At Runtime:** `button` could be ANY of 6 values!

**Without Dynamic Handling:**
```python
# OLD V1.0 approach
selector = "[data-openCreateDialogDropDown]"  # ❌ INCOMPLETE!
page.click(selector)  # ❌ Finds multiple elements, fails!
```

**With Dynamic Handling:**
```python
# NEW V2.0 approach
JIRA Step: "Click on create Test from dropdown"

LLM Analysis:
  Intent: Create "Test"

  Selector: data-openCreateDialogDropDown
  Possible values: [
    "aeName.StructureLevel.name",
    "aeName.Test.name",           ← Contains "Test"!
    "aeName.TestStep.name",
    ...
  ]

  Best match: "aeName.Test.name" (contains "Test")

  Final selector: [data-openCreateDialogDropDown="aeName.Test.name"]
  ✅ SUCCESS!
```

---

### 5. **Context Enrichment Fields** ⭐ **SOLVES POOR NAMING PROBLEM**

```json
"context": [
  "button",
  "menu-item",
  "dropdown-option",
  "create",
  "dialog",
  "clickable",
  "dynamic-content",
  "loop-generated"
]
```

**Purpose:** Semantic keywords for intelligent matching
**Problem Solved:** Developer named selector poorly, LLM understands purpose

#### **Problem Scenario:**

**JIRA Step:**
```
"Select create Test from the dropdown menu"
```

**Selector Name:**
```
data-openCreateDialogDropDown  ← Technical name
```

**Keywords from JIRA:**
```
["select", "create", "test", "dropdown", "menu"]
```

**Direct Matching:**
```python
# V1.0 approach
keywords = ["select", "create", "test", "dropdown", "menu"]
selector_name = "openCreateDialogDropDown"

matches = [
  "create" in "openCreateDialogDropDown" ✓  (1 match)
  "dropdown" in "openCreateDialogDropDown" ✓ (partial)
]

Score: LOW (only 1.5 keywords match)
```

**Context-Based Matching:**
```python
# V2.0 approach with context
keywords = ["select", "create", "test", "dropdown", "menu"]
context = ["button", "menu-item", "dropdown-option", "create", "dialog"]

matches = [
  "create" in context ✓
  "dropdown" → "dropdown-option" in context ✓
  "menu" → "menu-item" in context ✓
  "select" → "menu-item" in context ✓
]

Score: HIGH (4/5 keywords match via context)
✅ FOUND!
```

---

### 6. **Element Structure Fields**

```json
"elementType": "button",
"angularComponent": "mat-menu-item",
"angularDirectives": [],
"parentElement": "mat-menu",
"parentSelector": "[data-ShowMoreVerticalBtn]"
```

**Purpose:** Understand UI hierarchy and framework patterns
**Problem Solved:**
- **Type validation** - Ensure we're clicking a button, not a div
- **Framework patterns** - Recognize Angular Material components
- **Scoping** - Find element within specific parent

**Example:**
```python
# If looking for menu item, scope search within menu
if selector['parentElement'] == 'mat-menu':
    full_selector = f"{selector['parentSelector']} >> {selector['attr']}"
    # Result: [data-ShowMoreVerticalBtn] >> [data-openCreateDialogDropDown="..."]
    # ✅ More specific, faster, more reliable
```

---

### 7. **Priority & Scoring Fields**

```json
"priority": 8,
"confidence": "high"
```

**Purpose:** Rank selectors when multiple matches found
**Problem Solved:** Return BEST match, not FIRST match

**Priority Scale (0-10):**
```
10 = Primary action (Save, Submit, Create)
9  = Important action (Edit, Delete)
8  = Secondary action (Menu items, dropdowns)
7  = Navigation (Links, tabs)
6  = Icon buttons
5  = Generic elements
3  = Container divs
0  = Decorative elements
```

**Example:**
```python
# Multiple selectors match keyword "create"
matches = [
  {"attr": "data-create", "priority": 10},      # Primary button
  {"attr": "data-createIcon", "priority": 6},   # Icon
  {"attr": "data-createLabel", "priority": 3}   # Label text
]

# Sort by priority
best = max(matches, key=lambda x: x['priority'])
# Result: data-create (priority 10) ✅
```

---

### 8. **Semantic Understanding Fields** ⭐ **LLM INTELLIGENCE**

```json
"usage_scenario": "Dropdown menu item for creating different entity types (Project, Test, TestStep, Part, Equipment, Sequence). Dynamically generated from options array.",

"actionIntent": "create_entity",
"targetType": "menu_item_action"
```

**Purpose:** Help LLM understand what selector does
**Problem Solved:** Match by INTENT, not just keywords

**Example:**
```python
JIRA Step: "Create a new project from the options menu"

LLM Analysis:
  Intent: "create" + "project" + "options menu"

  Selector A:
    usage_scenario: "Dropdown menu item for creating different entity types (Project, Test...)"
    ✅ PERFECT MATCH!

  Selector B:
    usage_scenario: "Button to open project settings dialog"
    ❌ Wrong intent (settings, not create)

Result: Selector A selected with high confidence
```

---

### 9. **Interaction Fields**

```json
"interactionType": "click",
"eventHandler": "openCreateDialog(button)",
"expectedAction": "Opens create dialog for selected entity type"
```

**Purpose:** Understand what happens when element is clicked
**Problem Solved:**
- **Validation** - Verify correct action occurred
- **Timing** - Know if dialog will open (need to wait)
- **Debugging** - Understand what should happen

**Example:**
```python
# Execute action
page.click(selector)

# Check expected action
if selector['expectedAction'] == "Opens create dialog":
    # Wait for dialog to appear
    page.wait_for_selector(".mat-dialog", timeout=5000)
    ✅ Validated!
```

---

### 10. **Visibility & Conditional Fields** ⭐ **SOLVES CONDITIONAL RENDERING**

```json
"conditionalRendering": "@if ((!_detailViewCondition) && (isCreateBtnEnable || options.length))",
"visibilityContext": "Visible when NOT in detail view AND (create button enabled OR options exist)",
"requiresPermission": true,
"permissionCheck": "isCreate*Allowed"
```

**Purpose:** Know when selector is visible/available
**Problem Solved:** Don't try to click invisible elements!

**Example:**
```python
Sequential Context State:
  current_view = "detail_view"

Selector A:
  visibilityContext: "Visible when NOT in detail view"
  ❌ NOT VISIBLE in detail view → Skip this selector

Selector B:
  visibilityContext: "Visible in detail view"
  ✅ VISIBLE → Try this selector
```

---

### 11. **Form & State Fields**

```json
"isFormField": false,
"formControlName": null,
"stateDependent": true,
"requiredState": {
  "detailView": false,
  "createEnabled": true
}
```

**Purpose:** Understand state requirements
**Problem Solved:** Match selectors only when state is correct

**Example:**
```python
Current State:
  edit_mode = True
  dialog_open = True

Selector for Save button:
  requiredState: {"edit_mode": true, "dialog_open": true}
  ✅ State matches → High priority

Selector for Create button:
  requiredState: {"edit_mode": false}
  ❌ State mismatch → Skip or low priority
```

---

### 12. **HTML Reference Fields**

```json
"htmlSnippet": "<button (click)=\"openCreateDialog(button)\" [attr.data-openCreateDialogDropDown]=\"button\" mat-menu-item>{{button | translate}}</button>",

"surroundingContext": {
  "before": "<mat-menu #dropdownMenu=\"matMenu\">",
  "after": "</mat-menu>"
}
```

**Purpose:** Developer reference and debugging
**Problem Solved:**
- **Debugging** - See exact HTML that was parsed
- **Validation** - Verify extraction was correct
- **Maintenance** - Developer can find and fix issues

---

## 🎯 Why These Fields Solve Real Problems

### **Problem 1: Poor Developer Naming**

**Selector Name:** `data-ShowMoreVerticalBtn`
**JIRA Says:** "Click on more options menu"

**Solution:**
```json
"context": ["menu-trigger", "dropdown", "options", "show-more"],
"usage_scenario": "Dropdown menu trigger for more options"
```
✅ LLM matches "more options menu" → "menu-trigger", "dropdown", "options"

---

### **Problem 2: Dynamic Runtime Values**

**HTML:** `[attr.data-create]="buttonType"`
**Runtime:** `buttonType` could be "Project", "Test", "TestStep"

**Solution:**
```json
"possibleValues": ["Project", "Test", "TestStep"],
"dynamicValueSource": "component.ts:45"
```
✅ LLM matches JIRA "Create Test" → Uses value "Test"

---

### **Problem 3: Module Mismatch**

**Current Module:** "Teststep"
**Selector Module:** "Parts"
**Action:** "Click parts accordion"

**Solution:**
```json
"module": "Parts",
"visibilityContext": "Visible when teststep detail view is open"
```
✅ Sequential context knows Parts is visible → Selector allowed

---

### **Problem 4: Conditional Rendering**

**Selector:** Create button
**Condition:** Only visible when NOT in detail view

**Solution:**
```json
"conditionalRendering": "@if (!_detailViewCondition)",
"visibilityContext": "Visible when NOT in detail view",
"requiredState": {"detailView": false}
```
✅ If in detail view → Skip this selector (not visible)

---

### **Problem 5: Multiple Matches**

**Keyword:** "save"
**Matches Found:**
- Save button (priority: 10)
- Save icon (priority: 6)
- "Saved successfully" label (priority: 3)

**Solution:**
```json
{"attr": "data-save", "priority": 10, "elementType": "button"}
{"attr": "data-saveIcon", "priority": 6, "elementType": "mat-icon"}
{"attr": "data-saveMsg", "priority": 3, "elementType": "span"}
```
✅ Priority sorting returns Save button (priority: 10)

---

## ⚠️ Possible Issues & Solutions

### **Issue 1: Dynamic Value Extraction Fails**

**Problem:**
```typescript
// TypeScript is too complex
this.buttonName = this.getButtonNameFromService();
```

**Impact:** `possibleValues` will be empty or incomplete

**Solution:**
```json
"possibleValues": [],
"dynamicExtractionFailed": true,
"fallbackStrategy": "use_L2_patterns"
```

**Runtime Handling:**
```python
if selector['dynamicExtractionFailed']:
    # Don't use L1, go directly to L2 or L3
    try_level_2()
```

---

### **Issue 2: Context Extraction Too Generic**

**Problem:**
```json
"context": ["button", "action", "click"]  ← Too generic!
```

**Impact:** Many selectors have same context, poor matching

**Solution:**
- Add more specific keywords from surrounding HTML
- Use parent element context
- Include nearby text content

**Better:**
```json
"context": [
  "button",
  "action",
  "click",
  "create",           // From nearby text
  "primary-action",   // From CSS class
  "dialog-trigger",   // From parent context
  "mat-raised-button" // From Angular directive
]
```

---

### **Issue 3: Usage Scenario Not Clear**

**Problem:**
```json
"usage_scenario": "Button for action"  ← Not helpful!
```

**Impact:** LLM can't distinguish between similar selectors

**Solution:**
- Use LLM to generate detailed scenarios
- Include what the button does
- Include when it's visible

**Better:**
```json
"usage_scenario": "Primary create button that opens dialog for creating new Test, Project, or TestStep. Button text changes based on current module context. Only visible when user has create permission and NOT in detail view."
```

---

### **Issue 4: Module Name Mismatch**

**Problem:**
```
Folder: "entity-attribute"
Module extracted: "entity-attribute"
But JIRA says: "EntityAttribute"
```

**Impact:** Module filtering may not work correctly

**Solution:**
```json
"module": "EntityAttribute",
"moduleAliases": ["entity-attribute", "entityAttribute", "EntityAttribute"],
"modulePath": "entity-attribute"
```

**Runtime:**
```python
def module_matches(selector_module, current_module):
    return (
        selector_module == current_module or
        current_module in selector['moduleAliases'] or
        selector_module in selector['moduleAliases']
    )
```

---

### **Issue 5: File Structure Changes**

**Problem:**
Developer refactors code, moves files

**Impact:**
```json
"filePath": "src/app/old-location/component.html"  ← File no longer exists!
```

**Solution:**
- Re-run extractor when codebase changes
- Add timestamp to track extraction date
- Version control for enriched selectors

```json
"extractedDate": "2025-11-03T15:30:00Z",
"codebaseVersion": "git-commit-hash-abc123",
"needsUpdate": false
```

**Maintenance:**
```bash
# After code changes
git diff --name-only HEAD~1 HEAD | grep "\.html$"
# If HTML files changed, re-run extractor
python enriched_selector_extractor.py --incremental
```

---

### **Issue 6: Too Many Selectors**

**Problem:**
Large codebase → 5000+ selectors → Slow search

**Impact:**
- L1 search takes long
- Memory usage high
- LLM token limits exceeded

**Solution:**

**A. Module-based filtering:**
```python
# Only load selectors for current + visible modules
current_modules = ["Teststep", "Parts", "EntityAttribute"]
selectors = load_selectors_for_modules(current_modules)
# Result: 150 selectors instead of 5000
```

**B. Priority filtering:**
```python
# Only consider high-priority selectors first
high_priority = [s for s in selectors if s['priority'] >= 7]
# Try high priority first, fall back to all if needed
```

**C. Indexing:**
```python
# Pre-index selectors by keyword
index = {
  "create": [selector_1, selector_5, selector_42],
  "save": [selector_8, selector_12],
  ...
}
# Fast keyword lookup
```

---

### **Issue 7: Conflicting Selectors**

**Problem:**
Multiple selectors match the same element

**Example:**
```json
Selector A: {"attr": "data-save", "value": "button"}
Selector B: {"attr": "data-saveBtn", "value": "save"}
```
Both point to the same Save button!

**Impact:** Duplicate selectors, confusion

**Solution:**
```json
"relatedSelectors": ["selector_008", "selector_042"],
"isPrimarySelector": true,
"alternativeFor": null
```

**Runtime:**
```python
if selector['isPrimarySelector']:
    # Use this one
    return selector
else:
    # Use primary instead
    return get_selector(selector['alternativeFor'])
```

---

## 📊 File Size Estimation

### **For Your Codebase:**

```
Components: 82 HTML files
Estimated selectors per file: 15-20
Total selectors: ~1200-1600

Per selector (enriched): ~1.5 KB JSON
Total file size: 1200 × 1.5 KB = 1.8 MB

With compression: ~400-500 KB
```

**Is this a problem?**
- ❌ No! Modern systems handle this easily
- Loading time: <100ms
- Memory usage: <10 MB

---

## ✅ Summary: Why These Fields?

| Field Category | Purpose | Problem Solved |
|---------------|---------|----------------|
| **Basic ID** | Tracking | Debugging, logging |
| **Selector** | Matching | Find element in page |
| **Module/Location** | Context | Filter by visible modules |
| **Dynamic Values** | Runtime | Handle variable selectors ⭐ |
| **Context** | Semantic | Poor naming → Smart matching ⭐ |
| **Element Structure** | Hierarchy | Scoping, validation |
| **Priority** | Ranking | Best match, not first match |
| **Usage Scenario** | LLM Intelligence | Intent-based matching ⭐ |
| **Interaction** | Behavior | What happens when clicked |
| **Visibility** | State | Skip invisible selectors |
| **Form/State** | Requirements | State-aware matching |
| **HTML Reference** | Debugging | Developer maintenance |

**The 3 MOST IMPORTANT:** (⭐)
1. **Dynamic Values** - Solves dynamic selector problem
2. **Context** - Solves poor naming problem
3. **Usage Scenario** - Enables LLM intelligence

---

**Questions?**
- Want to see the actual extraction algorithm?
- Need clarification on any field?
- Want to add/remove any fields?
