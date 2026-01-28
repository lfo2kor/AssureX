# Refined PHASE 1: Automated Enriched Selector Extraction

## 🎯 Goal: ONE-STEP Extraction from Codebase → Enriched Selectors

**OLD APPROACH (2 steps):**
```
Codebase → Basic Extractor → selectors.json (basic)
                                    ↓
                           enrich_selectors.py
                                    ↓
                     selectors_enriched.json (enriched)
```

**NEW APPROACH (1 step):**
```
Codebase → Enriched Extractor → selectors_enriched.json (fully enriched)
           (direct, intelligent)
```

---

## 📥 REFINED PHASE 1: INPUTS

### What Testing Team Provides (One-Time Setup)

```yaml
1. ✅ Web Application Details
   - URL: https://your-application.com
   - Login credentials:
       username: testuser@company.com
       password: ********

2. ✅ Codebase Access
   - Path: C:/path/to/codebase/client/src/app
   - Type: Angular / React / Vue
   - Structure: Component-based (HTML + TypeScript/JSX files)

3. ✅ Framework Information (Optional - auto-detected)
   - Framework: Angular
   - UI Library: Angular Material
   - Version: 14+ (optional)
```

**Note: JIRA tickets are NOT needed in Phase 1**
- Phase 1 = Setup (one-time)
- JIRA tickets = Runtime (per test execution)

---

## 🔧 PHASE 1: ENRICHED SELECTOR EXTRACTION

### Run Command

```bash
python enriched_selector_extractor.py \
    --codebase-path "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src/app" \
    --framework angular \
    --output "Selectors_Folder/selectors_enriched_all_modules.json"
```

### What the Extractor Does (Automatically)

```
┌───────────────────────────────────────────────────────────────┐
│         ENRICHED SELECTOR EXTRACTOR (Smart Engine)            │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 1: Codebase Discovery & Module Detection               │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  • Recursively scan: C:/path/to/codebase/src/app             │
│  • Find all HTML component files                             │
│  • Detect module from folder structure:                      │
│      /app/create-new/ → Module: "CreateNew"                  │
│      /app/parts/ → Module: "Parts"                           │
│      /app/entity-attribute/ → Module: "EntityAttribute"      │
│                                                               │
│  • Find corresponding TypeScript files (.ts)                 │
│  • Auto-detect framework: Angular / React / Vue              │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 2: STATIC Selector Extraction                          │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  HTML:                                                        │
│    <button data-ShowMoreVerticalBtn="ShowMoreVerticalBtn">   │
│                                                               │
│  EXTRACT:                                                     │
│    ✓ Attribute: data-ShowMoreVerticalBtn                     │
│    ✓ Value: "ShowMoreVerticalBtn" (fixed)                    │
│    ✓ Element type: button                                    │
│    ✓ Angular directive: mat-raised-button                    │
│    ✓ Event: (click)="openCreateDialog()"                     │
│    ✓ Context: matMenuTriggerFor → menu-trigger               │
│                                                               │
│  CATEGORIZE:                                                  │
│    Type: STATIC (value is hardcoded string)                  │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 3: DYNAMIC Selector Extraction                         │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  HTML:                                                        │
│    <button [attr.data-openCreateDialog]="buttonName">        │
│                                                               │
│  DETECT:                                                      │
│    ✓ Dynamic binding: [attr.data-openCreateDialog]           │
│    ✓ Variable: buttonName (runtime value)                    │
│                                                               │
│  ANALYZE TypeScript:                                          │
│    • Find: create-new.component.ts                           │
│    • Search for: buttonName declaration                      │
│    • Extract possible values:                                │
│                                                               │
│      buttonName: string;                                      │
│      ...                                                      │
│      if (condition) {                                         │
│        this.buttonName = 'aeName.Test.name';                 │
│      } else if (other) {                                      │
│        this.buttonName = 'aeName.TestStep.name';             │
│      }                                                        │
│                                                               │
│  EXTRACT POSSIBLE VALUES:                                     │
│    ['aeName.Test.name', 'aeName.TestStep.name']              │
│                                                               │
│  CATEGORIZE:                                                  │
│    Type: DYNAMIC                                              │
│    Possible values: [extracted array]                        │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 4: LOOP-BASED Dynamic Selector Extraction              │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  HTML:                                                        │
│    @for (button of options; track button) {                  │
│      <button [attr.data-openCreateDialogDropDown]="button">  │
│    }                                                          │
│                                                               │
│  DETECT:                                                      │
│    ✓ Loop variable: button                                   │
│    ✓ Loop source: options                                    │
│    ✓ Dynamic attribute: data-openCreateDialogDropDown        │
│                                                               │
│  ANALYZE TypeScript:                                          │
│    • Find: options array declaration                         │
│                                                               │
│      options: string[] = [                                    │
│        'aeName.StructureLevel.name',                          │
│        'aeName.Test.name',                                    │
│        'aeName.TestStep.name',                                │
│        'aeName.UnitUnderTest.name',                           │
│        'aeName.TestEquipment.name',                           │
│        'aeName.TestSequence.name'                             │
│      ];                                                       │
│                                                               │
│  EXTRACT ALL POSSIBLE VALUES:                                 │
│    [                                                          │
│      'aeName.StructureLevel.name',                            │
│      'aeName.Test.name',                                      │
│      'aeName.TestStep.name',                                  │
│      'aeName.UnitUnderTest.name',                             │
│      'aeName.TestEquipment.name',                             │
│      'aeName.TestSequence.name'                               │
│    ]                                                          │
│                                                               │
│  CATEGORIZE:                                                  │
│    Type: DYNAMIC_LOOP                                         │
│    Loop variable: button                                     │
│    Possible values: [array extracted]                        │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 5: Context Enrichment (Automatic Semantic Analysis)    │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  For EACH extracted selector:                                │
│                                                               │
│  A. ELEMENT TYPE ANALYSIS                                     │
│     HTML: <button mat-raised-button>                         │
│     → Element: button                                         │
│     → Framework component: mat-raised-button                  │
│     → Context: ["button", "action", "primary"]               │
│                                                               │
│  B. ANGULAR DIRECTIVE ANALYSIS                                │
│     matMenuTriggerFor → ["menu-trigger", "dropdown"]         │
│     mat-expansion-panel → ["accordion", "expansion"]         │
│     formControlName → ["input", "form-field"]                │
│     mat-select → ["dropdown", "select", "option"]            │
│                                                               │
│  C. PARENT/SIBLING CONTEXT                                    │
│     <mat-menu>                                               │
│       <button data-menuItem="item">                          │
│     → Context: ["menu-item", "dropdown-option"]              │
│                                                               │
│     <mat-expansion-panel>                                    │
│       <div data-panelContent="content">                      │
│     → Context: ["accordion-content", "expansion-panel"]      │
│                                                               │
│  D. ATTRIBUTE NAME ANALYSIS (Semantic Extraction)            │
│     data-ShowMoreVerticalBtn → ["show", "more", "vertical", "button"]│
│     data-openCreateDialog → ["open", "create", "dialog"]     │
│     data-partspanel → ["parts", "panel"]                     │
│                                                               │
│  E. EVENT ANALYSIS                                            │
│     (click)="openCreateDialog()" → ["clickable", "interactive"]│
│     (change)="onValueChange()" → ["changeable", "input"]     │
│                                                               │
│  F. CONDITIONAL RENDERING ANALYSIS                            │
│     @if (_detailViewCondition)                               │
│     → Usage scenario: "Visible only in detail view"          │
│                                                               │
│     @if (isCreateBtnEnable)                                  │
│     → Usage scenario: "Conditionally enabled create button"  │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 6: Priority Calculation                                │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  Scoring Algorithm:                                           │
│                                                               │
│  Priority = 0                                                 │
│                                                               │
│  IF primary action button: +3                                │
│  IF unique data attribute: +2                                │
│  IF in Angular Material component: +2                        │
│  IF has clear semantic name: +1                              │
│  IF in form/dialog: +1                                       │
│  IF frequently used pattern: +1                              │
│                                                               │
│  Scale: 0-10 (10 = highest priority)                         │
│                                                               │
│  Examples:                                                    │
│    • Save button (primary action): 10                        │
│    • Create button: 9                                        │
│    • Edit button in row: 8                                   │
│    • Menu item: 7                                            │
│    • Icon button: 6                                          │
│    • Generic div: 3                                          │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 7: Usage Scenario Generation (LLM-Enhanced)            │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  For complex selectors, use LLM to generate description:     │
│                                                               │
│  Input to LLM:                                                │
│    HTML snippet:                                              │
│      <button [matMenuTriggerFor]="dropdownMenu"              │
│        data-ShowMoreVerticalBtn="ShowMoreVerticalBtn">       │
│        <mat-icon>Bosch-Ic-show-more-vertical</mat-icon>      │
│      </button>                                                │
│                                                               │
│    Context: CreateNew module                                  │
│    Element type: button with menu trigger                    │
│                                                               │
│  LLM Output:                                                  │
│    "Dropdown menu trigger button showing more options        │
│     for creating new entities (Project, Test, TestStep)"     │
│                                                               │
│  Usage: Helps LLM selector matcher understand purpose        │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STEP 8: Output Generation                                   │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  Generate enriched JSON with ALL information                 │
└───────────────────────────────────────────────────────────────┘
```

---

## 📤 REFINED PHASE 1: OUTPUTS

### Output File: `selectors_enriched_all_modules.json`

**Structure:**

```json
{
  "metadata": {
    "extraction_date": "2025-11-03T14:30:00Z",
    "codebase_path": "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src/app",
    "framework": "Angular",
    "ui_library": "Angular Material",
    "total_selectors": 1247,
    "static_selectors": 986,
    "dynamic_selectors": 261,
    "modules_scanned": 45
  },
  "selectors": [
    // EXAMPLE 1: STATIC SELECTOR
    {
      "id": "selector_001",
      "type": "STATIC",
      "attr": "data-ShowMoreVerticalBtn",
      "value": "ShowMoreVerticalBtn",
      "module": "CreateNew",
      "filePath": "create-new/create-new.component.html",
      "lineNumber": 20,

      // ENRICHED CONTEXT
      "context": [
        "button",
        "menu-trigger",
        "dropdown",
        "show-more",
        "vertical",
        "options-menu",
        "primary-action"
      ],

      // ELEMENT INFORMATION
      "elementType": "button",
      "angularComponent": "mat-raised-button",
      "angularDirectives": ["matMenuTriggerFor"],

      // SEMANTIC ANALYSIS
      "priority": 9,
      "usage_scenario": "Dropdown menu trigger button for creating new entities (Project, Test, TestStep, etc.). Shows vertical three-dot icon.",

      // INTERACTION
      "interactionType": "click",
      "eventHandler": "Triggers dropdown menu",

      // VISIBILITY CONDITIONS
      "conditionalRendering": "@if (!_detailViewCondition && (isCreateBtnEnable || options.length))",
      "visibilityContext": "Visible when NOT in detail view AND create is enabled",

      // HTML SNIPPET (for reference)
      "htmlSnippet": "<button [matMenuTriggerFor]=\"dropdownMenu\" color=\"primary\" data-ShowMoreVerticalBtn=\"ShowMoreVerticalBtn\" mat-raised-button class=\"mdc-icon-button-theme\">"
    },

    // EXAMPLE 2: DYNAMIC SELECTOR (Single Variable)
    {
      "id": "selector_002",
      "type": "DYNAMIC",
      "attr": "data-openCreateDialog",
      "value": "{{buttonName}}",  // Dynamic
      "module": "CreateNew",
      "filePath": "create-new/create-new.component.html",
      "lineNumber": 11,

      // DYNAMIC VALUE INFORMATION
      "isDynamic": true,
      "dynamicVariable": "buttonName",
      "possibleValues": [
        "aeName.StructureLevel.name",
        "aeName.Test.name",
        "aeName.TestStep.name"
      ],
      "dynamicValueSource": "create-new.component.ts:line 79",

      // ENRICHED CONTEXT
      "context": [
        "button",
        "create",
        "dialog",
        "primary-action",
        "dynamic-content"
      ],

      "elementType": "button",
      "angularComponent": "mat-raised-button",

      "priority": 10,
      "usage_scenario": "Primary create button that opens dialog. Button text changes based on current module (Project/Test/TestStep).",

      "interactionType": "click",
      "eventHandler": "openCreateDialog(buttonName)",

      "conditionalRendering": "@if (!_detailViewCondition && isCreateBtnEnable)",
      "visibilityContext": "Visible when NOT in detail view AND create is enabled",

      "htmlSnippet": "<button (click)=\"openCreateDialog(buttonName)\" [attr.data-openCreateDialog]=\"buttonName\" color=\"primary\" mat-raised-button>"
    },

    // EXAMPLE 3: DYNAMIC LOOP SELECTOR
    {
      "id": "selector_003",
      "type": "DYNAMIC_LOOP",
      "attr": "data-openCreateDialogDropDown",
      "value": "{{button}}",  // From loop
      "module": "CreateNew",
      "filePath": "create-new/create-new.component.html",
      "lineNumber": 27,

      // LOOP INFORMATION
      "isDynamic": true,
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
      "dynamicValueSource": "create-new.component.ts:line 43 (options array)",
      "loopContext": "@for (button of options; track button)",

      // ENRICHED CONTEXT
      "context": [
        "button",
        "menu-item",
        "dropdown-option",
        "create",
        "dialog",
        "loop-generated"
      ],

      "elementType": "button",
      "angularComponent": "mat-menu-item",
      "parentElement": "mat-menu",

      "priority": 8,
      "usage_scenario": "Menu item in dropdown for creating different entity types. Dynamically generated from options array.",

      "interactionType": "click",
      "eventHandler": "openCreateDialog(button)",

      "htmlSnippet": "<button (click)=\"openCreateDialog(button)\" [attr.data-openCreateDialogDropDown]=\"button\" data-openCreateDialogBtnAddIcon=\"addIcon\" mat-menu-item [value]=\"button\">"
    },

    // EXAMPLE 4: FORM FIELD SELECTOR
    {
      "id": "selector_150",
      "type": "STATIC",
      "attr": "data-type-field",
      "value": "typeDropdown",
      "module": "Parts",
      "filePath": "parts/edit-part-dialog.component.html",
      "lineNumber": 45,

      "context": [
        "dropdown",
        "select",
        "form-field",
        "type",
        "mat-select",
        "input"
      ],

      "elementType": "mat-select",
      "angularComponent": "mat-select",
      "angularDirectives": ["formControlName"],
      "formControlName": "partType",

      "priority": 9,
      "usage_scenario": "Dropdown field for selecting part type in edit dialog",

      "interactionType": "select",
      "eventHandler": "Form control binding",

      "conditionalRendering": "@if (editMode)",
      "visibilityContext": "Visible only in edit mode",

      "htmlSnippet": "<mat-select data-type-field=\"typeDropdown\" formControlName=\"partType\">"
    }
  ]
}
```

---

## 🎯 Key Features of Enriched Extractor

### 1. **Handles STATIC Selectors**
```html
<button data-save="saveBtn">Save</button>
```
**Extracted:**
- Type: STATIC
- Value: "saveBtn" (fixed)
- No dynamic analysis needed

---

### 2. **Handles DYNAMIC Selectors (Single Variable)**
```html
<button [attr.data-action]="actionName">Do Action</button>
```

**TypeScript Analysis:**
```typescript
// Searches .ts file for:
actionName: string;
this.actionName = 'create';  // Found!
this.actionName = 'edit';    // Found!
this.actionName = 'delete';  // Found!
```

**Extracted:**
- Type: DYNAMIC
- Variable: actionName
- Possible values: ['create', 'edit', 'delete']

---

### 3. **Handles DYNAMIC LOOP Selectors**
```html
@for (item of itemList; track item) {
  <button [attr.data-item]="item">{{item}}</button>
}
```

**TypeScript Analysis:**
```typescript
// Searches .ts file for:
itemList: string[] = ['Project', 'Test', 'TestStep'];
```

**Extracted:**
- Type: DYNAMIC_LOOP
- Loop variable: item
- Loop source: itemList
- Possible values: ['Project', 'Test', 'TestStep']

---

### 4. **Automatic Context Extraction**

**From element structure:**
```html
<mat-expansion-panel data-parts="accordion">
  <mat-expansion-panel-header>Parts</mat-expansion-panel-header>
</mat-expansion-panel>
```

**Context extracted:**
- Element type: mat-expansion-panel
- Angular component: expansion panel
- Context: ["accordion", "expansion", "parts", "panel", "collapsible"]
- Priority: 9 (important UI element)

---

### 5. **Framework-Agnostic Patterns**

**Angular Material:**
```
mat-button → ["button", "action"]
mat-select → ["dropdown", "select"]
mat-expansion-panel → ["accordion", "expansion"]
matMenuTriggerFor → ["menu-trigger", "dropdown"]
```

**React Material UI:**
```
MuiButton-root → ["button", "action"]
MuiSelect-root → ["dropdown", "select"]
MuiAccordion-root → ["accordion", "expansion"]
```

**Vue Vuetify:**
```
v-btn → ["button", "action"]
v-select → ["dropdown", "select"]
v-expansion-panel → ["accordion", "expansion"]
```

**Works on ANY framework automatically!**

---

## 🔄 How Dynamic Selector Matching Works at Runtime

### Problem: Dynamic selectors have variable values

**HTML:**
```html
<button [attr.data-openCreateDialog]="buttonName">Create</button>
```

**At runtime, buttonName could be:**
- "aeName.Test.name"
- "aeName.TestStep.name"
- "aeName.StructureLevel.name"

### Solution: LLM Matches by Intent + Possible Values

**JIRA Step:**
```
"Click on create Project button"
```

**LLM Analysis:**
```python
Step intent: Create a "Project"

Selector candidates:
  1. {
       "attr": "data-openCreateDialog",
       "value": "{{buttonName}}",  # Dynamic
       "possibleValues": [
         "aeName.StructureLevel.name",  ← Contains "StructureLevel" (Project!)
         "aeName.Test.name",
         "aeName.TestStep.name"
       ],
       "usage_scenario": "Create button for different entity types"
     }

LLM reasoning:
  - User wants to create "Project"
  - Selector has possible value "aeName.StructureLevel.name"
  - StructureLevel = Project in this application
  - MATCH! This is the correct selector

At runtime:
  Try selector: [data-openCreateDialog="aeName.StructureLevel.name"]
  ✅ SUCCESS!
```

---

## 📊 Comparison: Old vs New Approach

| Aspect | OLD (2-step) | NEW (1-step Enriched) |
|--------|--------------|------------------------|
| **Initial input** | Manual selectors.json | Just codebase path |
| **Steps** | 2 (extract → enrich) | 1 (direct enrichment) |
| **Static selectors** | ✅ Extracted | ✅ Extracted + Enriched |
| **Dynamic selectors** | ❌ Not handled | ✅ Fully handled |
| **Loop-based selectors** | ❌ Not handled | ✅ Extracted from code |
| **Context extraction** | Manual HTML parsing | Automatic semantic analysis |
| **Priority calculation** | Manual | Automatic scoring |
| **Usage scenarios** | Manual | LLM-generated |
| **Framework support** | Limited | Framework-agnostic |
| **Setup time** | 30 min | 5 min (run script) |
| **Maintenance** | High (manual updates) | Low (auto re-extract) |

---

## 🚀 REFINED PHASE 1 SUMMARY

### INPUTS (What Testing Team Provides)

```yaml
Initial Setup Information:
  1. Web Application URL
  2. Login credentials (username, password)
  3. Codebase path (e.g., /path/to/src/app)
  4. Framework type (Angular/React/Vue - optional, auto-detected)

NOT NEEDED in Phase 1:
  ❌ JIRA tickets (only needed at runtime)
  ❌ Manual selector list
  ❌ Historical test data
  ❌ Training data
```

### PROCESS

```bash
# Single command
python enriched_selector_extractor.py \
    --codebase-path "C:/path/to/src/app" \
    --output "selectors_enriched_all_modules.json"

# Runtime: 3-5 minutes for typical codebase
# Output: Fully enriched selectors ready for testing
```

### OUTPUTS

```
✅ selectors_enriched_all_modules.json
   - ALL selectors (static + dynamic)
   - Enriched with context keywords
   - Priority scores calculated
   - Usage scenarios generated
   - Dynamic possible values extracted
   - Framework patterns recognized
   - Module mapping complete

✅ extraction_report.txt
   - Total selectors: 1247
   - Static: 986
   - Dynamic: 261
   - Modules scanned: 45
   - Warnings: 12 (selectors with incomplete info)

✅ module_map.json
   - Module hierarchy
   - Module dependencies
   - Cross-module selectors identified
```

---

## 📋 Next Steps After Phase 1

### Phase 1 Complete → Ready for Testing

```bash
# Phase 1 output: selectors_enriched_all_modules.json ✅

# Now testing team provides JIRA tickets (runtime)
Jira_Tickets/RBPLCD-8835.txt
Jira_Tickets/RBPLCD-8862.txt

# Run tests
python main.py
> Enter ticket: RBPLCD-8835

# System uses enriched selectors automatically
# L1 success rate: 75-85% (vs 25% without enrichment)
```

---

## 🎯 Benefits of One-Step Enriched Extraction

1. **✅ No Manual Work**
   - No need to create initial selectors.json
   - Direct extraction from codebase

2. **✅ Handles Dynamic Selectors**
   - Extracts possible values from TypeScript
   - Loop-based selectors supported
   - Runtime value matching

3. **✅ Automatic Context**
   - Semantic analysis from HTML structure
   - Framework patterns recognized
   - Priority calculated automatically

4. **✅ Scalable to Any Project**
   - Framework-agnostic
   - No hardcoding
   - Works on Angular, React, Vue

5. **✅ Low Maintenance**
   - Re-run extractor when codebase changes
   - Automatically updates enriched selectors
   - No manual editing needed

---

## ❓ Questions This Solves

### Q: "Do I need to manually create selectors.json?"
**A:** No! Just provide codebase path. Extractor finds all selectors automatically.

### Q: "What about dynamic selectors with runtime values?"
**A:** Extractor analyzes TypeScript files and extracts all possible values automatically.

### Q: "How does it handle loops generating multiple buttons?"
**A:** Detects loop patterns (`@for`, `*ngFor`, `map()`) and extracts array values from TypeScript.

### Q: "Will this work for React or Vue projects?"
**A:** Yes! Framework-agnostic. Recognizes React (JSX), Vue (templates), Angular (HTML).

### Q: "Do I need JIRA tickets for Phase 1?"
**A:** No! Phase 1 = Setup (one-time). JIRA tickets only needed at runtime for actual test execution.

---

## 🔧 Implementation Status

```bash
# File to create:
enriched_selector_extractor.py

# Components needed:
1. HTML parser (BeautifulSoup / lxml)
2. TypeScript parser (typescript AST parser)
3. Framework pattern detector
4. Semantic context analyzer
5. Dynamic value extractor
6. Priority calculator
7. LLM usage scenario generator (optional)

# Estimated development time: 2-3 days
```

---

**Ready to implement the enriched selector extractor?**
