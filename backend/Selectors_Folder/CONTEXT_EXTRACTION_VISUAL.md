# Context Extraction - Visual Flow Diagram

## 🔄 **Complete Extraction Process**

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: HTML File                                           │
│  C:/.../ create-new.component.html                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Read HTML Line by Line                            │
│  ─────────────────────────────────────────────────────      │
│  Line 20:                                                   │
│  <button [matMenuTriggerFor]="dropdownMenu"                │
│          color="primary"                                    │
│          data-ShowMoreVerticalBtn="ShowMoreVerticalBtn">   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Find data-* Attributes                            │
│  ─────────────────────────────────────────────────────      │
│  Pattern: data-([a-zA-Z0-9\-_]+)="([^"]*)"                 │
│  ─────────────────────────────────────────────────────      │
│  Found:                                                     │
│    attr  = "data-ShowMoreVerticalBtn"                      │
│    value = "ShowMoreVerticalBtn"                           │
│    line_number = 20                                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Extract Element Type                              │
│  ─────────────────────────────────────────────────────      │
│  Pattern: <([a-z\-]+)                                      │
│  ─────────────────────────────────────────────────────      │
│  Found: <button                                            │
│  Result: elementType = "button"                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Analyze HTML for Context (20+ Rules)              │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴─────────────────┬──────────────────┐
        │                                    │                  │
        ▼                                    ▼                  ▼
┌──────────────────┐              ┌──────────────────┐  ┌─────────────────┐
│ Rule Category 1: │              │ Rule Category 2: │  │ Rule Category 3:│
│ Element Type     │              │ Angular/Material │  │ Click Handlers  │
└──────────────────┘              └──────────────────┘  └─────────────────┘
        │                                    │                  │
        │                                    │                  │
┌───────▼────────────────────────────────────▼──────────────────▼──────────┐
│  Context Extraction Rules Applied:                                       │
│  ───────────────────────────────────────────────────────────────────     │
│                                                                           │
│  ✅ Rule 1.1: Element Type Detection                                     │
│     HTML: <button                                                        │
│     → Add "button" to context                                            │
│                                                                           │
│  ✅ Rule 2.1: Material Directive Detection                               │
│     HTML: [matMenuTriggerFor]="dropdownMenu"                            │
│     → Add "menu-trigger" to context                                      │
│     → Add "dropdown" to context                                          │
│                                                                           │
│  ✅ Rule 2.2: Material Button Variant                                    │
│     HTML: color="primary"                                                │
│     → Add "primary-action" to context                                    │
│                                                                           │
│  ✅ Rule 4.1: Attribute Name Analysis - "show"                           │
│     attr: data-ShowMoreVerticalBtn                                       │
│     → "show" found → Add "show" to context                               │
│                                                                           │
│  ✅ Rule 4.2: Attribute Name Analysis - "more"                           │
│     attr: data-ShowMoreVerticalBtn                                       │
│     → "more" found → Add "more-options" to context                       │
│                                                                           │
│  ✅ Rule 4.3: Attribute Name Analysis - "vertical"                       │
│     attr: data-ShowMoreVerticalBtn                                       │
│     → "vertical" found → Add "more-vertical" to context                  │
│                                                                           │
│  RESULT:                                                                 │
│  context = ["button", "menu-trigger", "dropdown", "primary-action",     │
│             "show", "more-options", "more-vertical"]                     │
└───────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Calculate Priority (1-10)                         │
│  ─────────────────────────────────────────────────────      │
│  Input: context, attr, elementType                         │
│                                                             │
│  Algorithm:                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │ START: priority = 5 (default)               │           │
│  │                                              │           │
│  │ IF 'menu-trigger' in context:                │           │
│  │   priority = 10  ✅ (matched!)              │           │
│  │                                              │           │
│  │ ELSE IF 'primary-action' in context:         │           │
│  │   priority = 9                               │           │
│  │                                              │           │
│  │ ELSE IF 'dialog' in context:                 │           │
│  │   priority = 9                               │           │
│  │                                              │           │
│  │ ... (more rules)                             │           │
│  │                                              │           │
│  │ RETURN: priority = 10                        │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  RESULT: priority = 10 (Critical dropdown trigger)         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Build Usage Scenario                              │
│  ─────────────────────────────────────────────────────      │
│  Input: context, elementType                               │
│                                                             │
│  Logic:                                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │ parts = []                                   │           │
│  │                                              │           │
│  │ IF 'button' in context:                      │           │
│  │   IF 'primary-action' in context:            │           │
│  │     parts.append('Primary')  ✅             │           │
│  │                                              │           │
│  │   IF 'menu-trigger' in context:              │           │
│  │     parts.append('Dropdown menu trigger') ✅ │           │
│  │                                              │           │
│  │ IF 'create' in context:                      │           │
│  │   parts.append('create')  ❌ (not present)  │           │
│  │                                              │           │
│  │ RESULT: ' '.join(parts)                      │           │
│  │       = "Primary Dropdown menu trigger"      │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  RESULT: usage_scenario = "Primary Dropdown menu trigger"  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Assemble Complete Selector Object                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Enriched Selector                                  │
│  ═════════════════════════════════════════════════════      │
│  {                                                          │
│    "attr": "data-ShowMoreVerticalBtn",                     │
│    "value": "ShowMoreVerticalBtn",                         │
│    "module": "create-new",                                 │
│    "context": [                                             │
│      "button",                                             │
│      "menu-trigger",                                       │
│      "dropdown",                                           │
│      "primary-action",                                     │
│      "show",                                               │
│      "more-options",                                       │
│      "more-vertical"                                       │
│    ],                                                       │
│    "priority": 10,                                         │
│    "usage_scenario": "Primary Dropdown menu trigger",      │
│    "parentComponent": "create-new",                        │
│    "filePath": "src/app/create-new/create-new.component...",│
│    "dynamic": false,                                       │
│    "label": "",                                            │
│    "lineNumber": 20,                                       │
│    "elementType": "button"                                 │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 **How Priority Works in Matching**

### **Scenario: Find button for "click ... + dropdown"**

```
┌─────────────────────────────────────────────────────────────┐
│  TEST STEP: "Force Click on ... + button"                  │
│  ───────────────────────────────────────────────────────    │
│  Extracted Keywords: ['click', 'button', 'dropdown', '+']  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  SEARCH IN SELECTORS.JSON                                   │
│  (Assuming we have 50 selectors in create-new module)       │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴─────────────────┬──────────────────┐
        │                                    │                  │
        ▼                                    ▼                  ▼
┌──────────────────┐              ┌──────────────────┐  ┌─────────────────┐
│ Candidate 1:     │              │ Candidate 2:     │  │ Candidate 3:    │
│ data-createnew   │              │ data-ShowMoreBtn │  │ data-button     │
└──────────────────┘              └──────────────────┘  └─────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  SCORING ALGORITHM                                                         │
│  ────────────────────────────────────────────────────────────────────      │
│                                                                            │
│  For each selector:                                                        │
│    score = 0                                                               │
│                                                                            │
│    # Check keyword matches in context                                     │
│    for keyword in step_keywords:                                          │
│      if keyword in selector['context']:                                   │
│        score += 5                                                          │
│                                                                            │
│    # Add priority bonus                                                   │
│    score += selector['priority']                                          │
│                                                                            │
│    # Sort by score (highest first)                                        │
│    return selector with highest score                                     │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE 1: data-createnew                                               │
│  ────────────────────────────────────────────────────────────────────      │
│  context: ["container", "create"]                                         │
│  priority: 8                                                               │
│                                                                            │
│  Scoring:                                                                  │
│  ┌────────────────────────────────────────┐                               │
│  │ Keyword "click"    → Not in context ❌ │ +0                            │
│  │ Keyword "button"   → Not in context ❌ │ +0                            │
│  │ Keyword "dropdown" → Not in context ❌ │ +0                            │
│  │ Priority bonus     →                   │ +8                            │
│  │ ──────────────────────────────────────│                               │
│  │ TOTAL SCORE:                           │ 8                             │
│  └────────────────────────────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE 2: data-ShowMoreVerticalBtn                                     │
│  ────────────────────────────────────────────────────────────────────      │
│  context: ["button", "menu-trigger", "dropdown", "more-options", ...]     │
│  priority: 10                                                              │
│                                                                            │
│  Scoring:                                                                  │
│  ┌────────────────────────────────────────┐                               │
│  │ Keyword "click"    → Not in context ❌ │ +0                            │
│  │ Keyword "button"   → In context ✅     │ +5                            │
│  │ Keyword "dropdown" → In context ✅     │ +5                            │
│  │ Priority bonus     →                   │ +10                           │
│  │ ──────────────────────────────────────│                               │
│  │ TOTAL SCORE:                           │ 20  ✅ HIGHEST               │
│  └────────────────────────────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE 3: data-button                                                  │
│  ────────────────────────────────────────────────────────────────────      │
│  context: ["button", "clickable"]                                         │
│  priority: 5                                                               │
│                                                                            │
│  Scoring:                                                                  │
│  ┌────────────────────────────────────────┐                               │
│  │ Keyword "click"    → Not exact match ❌│ +0                            │
│  │ Keyword "button"   → In context ✅     │ +5                            │
│  │ Keyword "dropdown" → Not in context ❌ │ +0                            │
│  │ Priority bonus     →                   │ +5                            │
│  │ ──────────────────────────────────────│                               │
│  │ TOTAL SCORE:                           │ 10                            │
│  └────────────────────────────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘

                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RESULT: Select Highest Score                              │
│  ═══════════════════════════════════════════════════       │
│                                                             │
│  Scores:                                                    │
│    Candidate 1: 8                                          │
│    Candidate 2: 20  ✅ WINNER                              │
│    Candidate 3: 10                                         │
│                                                             │
│  SELECTED: data-ShowMoreVerticalBtn                        │
│  Confidence: HIGH (20 points, 2x higher than next best)    │
│  Time: 50ms (L1 success, no fallback needed)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 **Priority Scale Visual**

```
┌────────────────────────────────────────────────────────────────┐
│  PRIORITY LEVELS (1-10)                                        │
└────────────────────────────────────────────────────────────────┘

Priority 10 ████████████████████████████████████████ CRITICAL
  ├─ Test-specific attributes (data-testid, data-cy)
  ├─ Primary action buttons (create, save, submit)
  └─ Menu triggers (dropdown, ... + button)

Priority 9  ████████████████████████████████████ VERY HIGH
  ├─ Important actions (detail view, edit, delete)
  ├─ Menu items (dropdown options)
  └─ Dialog triggers

Priority 8  ███████████████████████████████ HIGH
  ├─ Containers (wrappers, panels)
  ├─ Action icons (edit icon, add icon)
  └─ Secondary buttons

Priority 7  ██████████████████████████ MEDIUM-HIGH
  ├─ Decorative icons (+ icon, arrow icon)
  └─ Supporting elements

Priority 5  ████████████████ DEFAULT
  └─ Unknown/unclassified elements

Priority 3  ████████ LOW
  └─ Low confidence matches

Priority 1  ██ VERY LOW
  └─ Fallback/last resort
```

---

## 🔄 **Complete Workflow: HTML → Enriched Selector**

```
HTML Source Code
        │
        ▼
┌───────────────────┐
│ Read HTML File    │
│ Line by Line      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Find data-*       │
│ Attributes        │
└───────────────────┘
        │
        ├─────────────────┬─────────────────┬──────────────────┐
        │                 │                 │                  │
        ▼                 ▼                 ▼                  ▼
┌─────────────┐   ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
│Extract      │   │Extract       │  │Extract      │  │Extract       │
│Element Type │   │Context (20+  │  │Priority     │  │Usage         │
│             │   │rules)        │  │(algorithm)  │  │Scenario      │
└─────────────┘   └──────────────┘  └─────────────┘  └──────────────┘
        │                 │                 │                  │
        └─────────────────┴─────────────────┴──────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Combine into    │
                  │ Selector Object │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Save to JSON    │
                  │ (enriched)      │
                  └─────────────────┘
```

---

## ✅ **Key Takeaways**

### **Context:**
- **Extracted from:** HTML structure, Angular directives, attribute names
- **NOT from:** JIRA tickets, test scenarios, manual input
- **Purpose:** Keyword matching for intelligent search
- **Example:** `["button", "dropdown", "menu-trigger"]`

### **Priority:**
- **Scale:** 1-10 (10 = most important)
- **Purpose:** Break ties when multiple selectors match keywords
- **Calculation:** Based on element importance, context, directives
- **Example:** Menu trigger = 10, Icon = 7

### **Result:**
- **Better Matching:** Context + Priority → Best selector selected
- **Faster Execution:** L1 success (50ms) vs L3 fallback (2-3s)
- **No Maintenance:** All automatic from HTML source

**All logic is in the Python script - runs automatically on any HTML file!**
