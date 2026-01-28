# Sequential Context Approach for L1 Selector Matching

## 🎯 Core Concept

**Instead of treating each step independently, build a STATE MACHINE that tracks context across steps.**

---

## 📊 Problem Illustration: RBPLCD-8835

### Current Independent Execution (FAILS)

```
Step 4: "Expand Parts accordion"
  - Search in: module=teststep
  - Find: data-parts-accordion
  - Module filter: ❌ Blocked (accordion is in parts module)
  - Result: L1 FAILS

Step 5: "Click edit button"
  - Search in: module=teststep
  - Find: data-editButton
  - Module filter: ❌ Blocked (edit button is in parts module)
  - Result: L1 FAILS

Step 6: "Select Type dropdown"
  - Search in: module=teststep
  - Find: data-attribute="Type"
  - Module filter: ❌ Blocked (attribute is in entity-attribute module)
  - Result: L1 FAILS

Total L1 Success: 0/3 steps = 0%
```

### With Sequential Context (SUCCESS)

```
INITIAL STATE:
  current_module: null
  current_section: null
  edit_mode: false
  visible_modules: []

Step 4: "Expand Parts accordion"
  - Keyword: "Parts", "accordion", "expand"
  - ACTION: Detect navigation to Parts section
  - UPDATE STATE:
      current_section: "Parts"
      current_module: "parts"  ← Context switch!
      visible_modules: ["teststep", "parts"]
  - Search in: module=parts OR visible_modules
  - Find: data-parts-accordion ✅
  - Result: L1 SUCCESS

Step 5: "Click edit button"
  - STATE: current_module=parts (from Step 4)
  - Keyword: "edit", "button"
  - ACTION: Enter edit mode
  - UPDATE STATE:
      edit_mode: true
      visible_modules: ["parts", "entity-attribute"]  ← Parts uses entity-attribute!
  - Search in: module=parts
  - Find: data-editButton ✅
  - Result: L1 SUCCESS

Step 6: "Select Type dropdown"
  - STATE:
      current_module=parts (from Step 4)
      edit_mode=true (from Step 5)
      visible_modules=["parts", "entity-attribute"] (from Step 5)
  - Keyword: "Type", "dropdown", "select"
  - Search in: visible_modules=["parts", "entity-attribute"]
  - Find: data-attribute="Type" in entity-attribute module ✅
  - Result: L1 SUCCESS

Total L1 Success: 3/3 steps = 100%! 🎉
```

---

## 🏗️ State Machine Design

### State Object

```python
class TestExecutionState:
    def __init__(self):
        # Navigation state
        self.current_module = None          # "teststep", "parts", etc.
        self.current_section = None         # "Parts", "Attributes", etc.
        self.navigation_path = []           # ["Login", "Teststep", "Parts"]

        # UI state
        self.visible_modules = []           # Modules currently visible/active
        self.edit_mode = False              # In edit mode?
        self.dialog_open = False            # Dialog/modal open?
        self.expanded_sections = []         # Expanded accordions

        # Context history
        self.previous_selectors = []        # Last N selectors used
        self.previous_modules = []          # Module history

        # Module dependencies
        self.module_dependencies = {
            "parts": ["entity-attribute"],       # Parts uses entity-attribute
            "teststep": ["entity-attribute"],    # Teststep uses entity-attribute
            "attributes": ["entity-attribute"],
        }

    def update_from_step(self, step_text, selected_selector):
        """Update state based on executed step"""
        # Analyze step action
        action = self._detect_action(step_text)

        if action == "expand":
            # Expanding accordion → switching section
            section = self._extract_section_name(step_text)
            self._enter_section(section)

        elif action == "navigate":
            # Navigation → switching module
            module = self._extract_module_name(step_text)
            self._navigate_to_module(module)

        elif action == "edit":
            # Edit button → enter edit mode
            self._enter_edit_mode()

        elif action == "close":
            # Close dialog/section
            self._exit_current_context()

        # Track selector history
        self.previous_selectors.append(selected_selector)
        self.previous_modules.append(self.current_module)
```

---

## 🔍 Context-Aware Selector Search

### Before (Independent)

```python
def find_selector_L1(step_text, jira_module):
    keywords = extract_keywords(step_text)

    # Search ONLY in JIRA module
    for selector in selectors:
        if selector['module'] != jira_module:
            continue  # ❌ BLOCKS cross-module selectors

        if keywords_match(selector, keywords):
            return selector

    return None  # L1 FAILS
```

### After (Sequential Context)

```python
def find_selector_L1_with_context(step_text, jira_module, state):
    keywords = extract_keywords(step_text)

    # Build search scope from state
    search_modules = state.get_search_scope(jira_module)
    # Returns: ["teststep", "parts", "entity-attribute"] based on state

    candidates = []
    for selector in selectors:
        if selector['module'] not in search_modules:
            continue

        score = calculate_score(selector, keywords, state)
        candidates.append((selector, score))

    # Return highest scored
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None
```

---

## 🎯 State-Aware Scoring

```python
def calculate_score(selector, keywords, state):
    score = 0

    # 1. Keyword matching (base score)
    for keyword in keywords:
        if keyword in selector['context']:
            score += 5

    # 2. Module relevance (state-based)
    if selector['module'] == state.current_module:
        score += 20  # ✅ Current module (high boost)
    elif selector['module'] in state.visible_modules:
        score += 15  # ✅ Visible module (good boost)
    elif selector['module'] == state.previous_modules[-1]:
        score += 10  # ✅ Recent module (medium boost)

    # 3. Section relevance
    if state.current_section and state.current_section.lower() in selector['value'].lower():
        score += 10  # ✅ Section match

    # 4. Edit mode context
    if state.edit_mode:
        if 'edit' in selector['context'] or 'input' in selector['context']:
            score += 8  # ✅ Edit-related selectors

    # 5. Dialog context
    if state.dialog_open:
        if 'dialog' in selector['context']:
            score += 8  # ✅ Dialog selectors

    # 6. Priority (from V2.0 enrichment)
    score += selector.get('priority', 5)

    return score
```

---

## 🧠 State Transition Rules

### Rule 1: Accordion Expansion → Section Switch

```python
Step: "Expand Parts accordion"

Detection:
  - Keywords: ["expand", "parts", "accordion"]
  - Action: expand

State Update:
  state.current_section = "Parts"
  state.current_module = "parts"
  state.expanded_sections.append("Parts")

  # Add dependent modules
  deps = state.module_dependencies.get("parts", [])
  state.visible_modules = ["teststep", "parts"] + deps
  # → ["teststep", "parts", "entity-attribute"]

Search Scope:
  Next step will search in: ["teststep", "parts", "entity-attribute"]
```

---

### Rule 2: Edit Button → Edit Mode + Module Dependencies

```python
Step: "Click edit button"

Detection:
  - Keywords: ["click", "edit", "button"]
  - Action: edit

State Update:
  state.edit_mode = True

  # In edit mode, entity-attribute selectors become visible
  if state.current_module == "parts":
      state.visible_modules.append("entity-attribute")

Search Scope:
  Next step will search in: ["parts", "entity-attribute"]
  → Now "Type" dropdown from entity-attribute is findable!
```

---

### Rule 3: Dialog Open → Dialog Context

```python
Step: "Click create button"

Detection:
  - Selector used: data-opencreatedialog
  - Selector context: ["create", "dialog"]
  - Action: open_dialog

State Update:
  state.dialog_open = True
  state.dialog_type = "create"
  state.visible_modules.append("create-new")

Search Scope:
  Next step will search in: [current_module, "create-new"]
```

---

### Rule 4: Navigation → Module Switch

```python
Step: "Navigate to Teststep"

Detection:
  - Keywords: ["navigate", "teststep"]
  - Action: navigate

State Update:
  state.current_module = "teststep"
  state.navigation_path.append("teststep")
  state.visible_modules = ["teststep"]

Search Scope:
  Next step will search in: ["teststep"]
```

---

## 📈 Expected Impact

### Current L1 Success Rate: ~20%

**Breakdown:**
- Simple steps (login, navigate): 60% success
- Cross-module steps (expand accordion): 10% success ❌
- Nested context steps (edit in section): 5% success ❌

### With Sequential Context: ~70-85%

**Breakdown:**
- Simple steps: 80% success (improved keyword matching)
- Cross-module steps: 75% success ✅ (state tracking)
- Nested context steps: 70% success ✅ (visible_modules)

---

## 🔧 Implementation Plan

### Phase 1: State Machine (Core)

1. **Create state object** to track:
   - current_module
   - current_section
   - visible_modules
   - edit_mode
   - dialog_open

2. **Define state transitions** for:
   - Accordion expansion
   - Edit mode entry
   - Dialog open/close
   - Navigation

3. **Update after each step:**
   ```python
   selector = find_selector_L1(step, state)
   execute_step(selector)
   state.update_from_step(step, selector)  # ← KEY!
   ```

---

### Phase 2: Module Dependency Map

Define which modules are visible from each context:

```python
MODULE_DEPENDENCIES = {
    "teststep": {
        "base": ["teststep"],
        "edit": ["teststep", "entity-attribute"],
        "parts_section": ["teststep", "parts", "entity-attribute"],
    },
    "parts": {
        "base": ["parts"],
        "edit": ["parts", "entity-attribute"],
    },
    "project": {
        "base": ["project"],
        "edit": ["project", "entity-attribute"],
    },
}
```

This can be **extracted automatically** by analyzing:
- HTML imports
- Component dependencies
- Shared component usage

---

### Phase 3: Context-Aware Search

Update `find_selector_L1()`:

```python
def find_selector_L1(step_text, jira_module, state):
    # OLD: search_modules = [jira_module]
    # NEW:
    search_modules = state.get_search_scope(jira_module)

    # Rest remains same (score-based matching)
    ...
```

---

### Phase 4: Step Action Detection

Classify steps into action types:

```python
def detect_action(step_text):
    step_lower = step_text.lower()

    if any(kw in step_lower for kw in ["expand", "open accordion"]):
        return "expand"

    elif any(kw in step_lower for kw in ["navigate", "go to", "open"]):
        return "navigate"

    elif any(kw in step_lower for kw in ["click edit", "edit button"]):
        return "edit"

    elif "close" in step_lower or "cancel" in step_lower:
        return "close"

    elif "select" in step_lower or "choose" in step_lower:
        return "select"

    else:
        return "interact"  # Generic action
```

---

## 🎯 Example: RBPLCD-8835 Execution Flow

### Full Ticket with Sequential Context

```python
# Initialize state
state = TestExecutionState()

# Step 1: Login to the application
step_1 = "Login to the application"
selector_1 = find_selector_L1(step_1, "login", state)
# → Finds: data-username, data-password, data-loginBtn
execute(selector_1)
state.update_from_step(step_1, selector_1)
# STATE: current_module="login", navigation_path=["login"]

# Step 2: Navigate to Teststep
step_2 = "Navigate to Teststep"
selector_2 = find_selector_L1(step_2, "teststep", state)
# → Finds: data-teststep-link
execute(selector_2)
state.update_from_step(step_2, selector_2)
# STATE: current_module="teststep", navigation_path=["login", "teststep"]

# Step 3: Click on teststep from listing page
step_3 = "Click on teststep from listing page"
selector_3 = find_selector_L1(step_3, "teststep", state)
# → Finds: data-teststep-row (in teststep module)
execute(selector_3)
state.update_from_step(step_3, selector_3)
# STATE: current_module="teststep", viewing_details=true

# Step 4: Expand Parts accordion ← STATE CHANGE!
step_4 = "Expand Parts accordion"
selector_4 = find_selector_L1(step_4, "teststep", state)
# Search in: ["teststep", "parts"] (state knows Parts is a section)
# → Finds: data-parts-accordion (in parts module) ✅
execute(selector_4)
state.update_from_step(step_4, selector_4)
# STATE:
#   current_section="Parts"
#   current_module="parts"
#   visible_modules=["teststep", "parts", "entity-attribute"]
#   expanded_sections=["Parts"]

# Step 5: Click edit button ← Uses STATE!
step_5 = "Click edit button"
selector_5 = find_selector_L1(step_5, "teststep", state)
# Search in: ["teststep", "parts", "entity-attribute"] (from state)
# → Finds: data-editButton (in parts module) ✅
execute(selector_5)
state.update_from_step(step_5, selector_5)
# STATE:
#   current_module="parts"
#   edit_mode=True
#   visible_modules=["parts", "entity-attribute"]

# Step 6: Select Type dropdown ← Uses STATE!
step_6 = "Select Type dropdown"
selector_6 = find_selector_L1(step_6, "teststep", state)
# Search in: ["parts", "entity-attribute"] (from state)
# → Finds: data-attribute="Type" (in entity-attribute module) ✅
execute(selector_6)
state.update_from_step(step_6, selector_6)

# Step 7: Click Save button
step_7 = "Click Save button"
selector_7 = find_selector_L1(step_7, "teststep", state)
# Search in: ["parts", "entity-attribute"]
# → Finds: data-saveButton ✅
execute(selector_7)
state.update_from_step(step_7, selector_7)
# STATE: edit_mode=False (save closes edit mode)

# Step 8: Verify success message
step_8 = "Verify success message"
selector_8 = find_selector_L1(step_8, "teststep", state)
# → Finds: data-success-message ✅

# RESULT: 8/8 steps L1 SUCCESS = 100%! 🎉
```

---

## 🔄 State Update Examples

### After "Expand Parts accordion"

```python
Before:
  current_module: "teststep"
  visible_modules: ["teststep"]

After:
  current_module: "parts"
  current_section: "Parts"
  visible_modules: ["teststep", "parts", "entity-attribute"]
  expanded_sections: ["Parts"]
```

---

### After "Click edit button"

```python
Before:
  edit_mode: False
  visible_modules: ["teststep", "parts", "entity-attribute"]

After:
  edit_mode: True
  visible_modules: ["parts", "entity-attribute"]  # Focus on edit context
```

---

### After "Click Save button"

```python
Before:
  edit_mode: True

After:
  edit_mode: False
  visible_modules: ["teststep", "parts"]  # Back to view mode
```

---

## 🎯 Benefits of Sequential Context

### 1. **Cross-Module Selectors Work** ✅
- Parts accordion (in parts module) found from teststep context
- Entity-attribute selectors found when editing Parts

### 2. **Context Accumulation** ✅
- Step 4 expands Parts → Step 5 knows we're in Parts
- Step 5 clicks edit → Step 6 knows edit mode active

### 3. **Intelligent Scope** ✅
- Not limited to single JIRA module
- Not searching ALL modules (too noisy)
- Searches RELEVANT modules based on state

### 4. **Action-Based State** ✅
- Edit button → edit_mode=True → prioritize input/dropdown selectors
- Accordion → current_section changes → search in that section's module

### 5. **Fallback Path** ✅
- If L1 fails with state-based search, fall back to L2/L3
- But L1 success rate dramatically improved

---

## 📊 Comparison: V1 vs V2 vs Sequential

| Approach | L1 Success | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| **V1.0 (Current)** | ~20% | Simple | Strict module filter blocks cross-module |
| **V2.0 (HTML Context)** | ~70-80% | Keyword-rich context | Still independent steps |
| **V3.0 (Sequential)** | ~85-95% | State tracking across steps | More complex logic |

---

## 🚀 Next Steps

### Option 1: V2.0 THEN Sequential
1. Implement V2.0 (HTML context enrichment) ← DONE!
2. Measure L1 improvement
3. Add sequential state tracking on top
4. Measure additional improvement

**Benefit:** Incremental approach, easier to debug

---

### Option 2: V2.0 + Sequential Together
1. Use enriched selectors WITH sequential context
2. Combine keyword scoring + state-based scope
3. Best of both worlds

**Benefit:** Maximum L1 success rate from day 1

---

## 🎯 Recommendation

**Implement BOTH:**

1. **V2.0 (HTML Context)** → Improves keyword matching
2. **Sequential State** → Improves module scope

**Combined Algorithm:**
```python
def find_selector_L1_v3(step_text, jira_module, state):
    # Get search scope from sequential state
    search_modules = state.get_search_scope(jira_module)

    # Extract keywords
    keywords = extract_keywords(step_text)

    # Score selectors using BOTH context AND state
    candidates = []
    for selector in selectors:
        if selector['module'] not in search_modules:
            continue

        # V2.0 contribution: keyword + context matching
        keyword_score = score_keywords(selector, keywords)

        # V3.0 contribution: state-based relevance
        state_score = score_state_relevance(selector, state)

        # Combined score
        total_score = keyword_score + state_score + selector['priority']
        candidates.append((selector, total_score))

    return max(candidates, key=lambda x: x[1])[0]
```

**Expected Result:** 85-95% L1 success rate! 🎉

---

## 📝 Summary

**Sequential Context solves the fundamental problem:**

❌ Old: "Each step is independent → loses context → cross-module fails"

✅ New: "Each step builds on previous → tracks context → cross-module works"

**Key Insight:**
> The test steps form a JOURNEY through the application.
> By tracking that journey, we know WHERE we are and WHAT modules are relevant.

This is the missing piece for high L1 success rate! 🚀
