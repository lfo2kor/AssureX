# Hardcoding Analysis: L1, L2, L3 - Make It Scalable!

## 🔴 PROBLEM: Hardcoding Everywhere!

You're absolutely right - there's **extensive hardcoding** in L1 and L2 that makes it NOT scalable across projects.

---

## Current Hardcoding Issues

### ❌ LEVEL 1: Hardcoded Keyword Extraction

**File:** `utils/selector_loader.py` (Lines 165-214)

**Hardcoded Code:**
```python
def _extract_keywords(self, step_text: str) -> List[str]:
    keywords = []

    # HARDCODED project-specific keywords
    if '... +' in step_text or 'more' in step_lower:
        keywords.extend(['showmoreverticalbtn', 'showmore', 'vertical', 'more'])
    if 'project' in step_lower or 'product' in step_lower:
        keywords.extend(['selectproject', 'project', 'product'])
    if 'accordion' in step_lower:
        keywords.extend(['accordion', 'panel'])
    if 'parts' in step_lower:
        keywords.append('parts')
    if 'teststep' in step_lower:
        keywords.append('teststep')
    # ... more hardcoded patterns
```

**Problems:**
- ❌ Keywords like `'showmoreverticalbtn'`, `'selectproject'` are PLCD-specific
- ❌ Keywords like `'parts'`, `'teststep'` are domain-specific
- ❌ Code must be changed for every new project
- ❌ Not reusable across different applications

---

### ❌ LEVEL 2: Hardcoded Generic Patterns

**File:** `utils/step_executor.py` (Lines 62-106)

**Hardcoded Code:**
```python
# Generic patterns for Level 2
self.generic_patterns = {
    'button_click': [
        "button:has-text('{text}')",
        "a:has-text('{text}')",
        "[role='button']:has-text('{text}')",
        "button[type='submit']",
        "input[type='submit']",
        "button",
    ],
    'dropdown_select': [
        "[data-attribute='{text}']",  # PLCD-specific!
        "[data-basicattribute='attribute'][aria-label*='{text}']",  # PLCD-specific!
        "input.mat-mdc-autocomplete-trigger[data-attribute='{text}']",  # Angular Material!
        "label:has-text('{text}') .mat-select",  # Angular Material!
        ".mat-select",  # Angular Material!
        "select",
    ],
    'accordion_expand': [
        "[role='button'][aria-expanded='false']",
        ".mat-expansion-panel-header:has-text('{text}')",  # Angular Material!
        ".accordion-header",
    ],
    'verify_message': [
        ":has-text('Successfully edited')",  # PLCD-specific message!
        ".mat-snack-bar-container",  # Angular Material!
        "[role='alert']",
        ".notification",
        ":text('Successfully')",  # PLCD-specific!
        "*",
    ],
}
```

**Problems:**
- ❌ **Angular Material-specific:** `.mat-select`, `.mat-expansion-panel-header`, `.mat-snack-bar-container`
  - Won't work in React, Vue, plain HTML projects!
- ❌ **PLCD-specific:** `[data-attribute]`, `[data-basicattribute]`
  - Won't work in projects that don't use these attributes!
- ❌ **Hardcoded messages:** `'Successfully edited'`, `'Successfully'`
  - Won't work in projects with different success messages!
- ❌ **Code must be changed** for every new project

---

### ❌ LEVEL 2: Hardcoded Action Detection

**File:** `utils/step_executor.py` (Lines 276-393)

**Hardcoded Code:**
```python
# Detect action type
if 'should be displayed' in step_lower or 'message' in step_lower:
    action_type = 'verify_message'
    # Extract message text from quotes
    match = re.search(r'^"([^"]*)"', step_text)
    # ...

elif 'dropdown' in step_lower or 'select' in step_lower:
    action_type = 'dropdown_select'
    # ...

elif 'navigate' in step_lower and self.module:
    extracted_text = self.web_module_name
    action_type = 'button_click'
    # ...

elif 'button' in step_lower or 'btn' in step_lower or 'click' in step_lower:
    action_type = 'button_click'
    # Extract button text
    for word in ['save', 'edit', 'close', 'cancel', 'submit', 'login', 'add']:
        if word in step_lower:
            extracted_text = word.capitalize()
            break
    # ...

elif 'accordion' in step_lower:
    action_type = 'accordion_expand'
    if 'parts' in step_lower:
        extracted_text = 'Parts'  # HARDCODED!
```

**Problems:**
- ❌ **Hardcoded action keywords:** `'save'`, `'edit'`, `'close'`, `'cancel'`, etc.
- ❌ **Hardcoded UI elements:** `'Parts'` accordion
- ❌ **Hardcoded patterns:** `'should be displayed'`, `'navigate'`
- ❌ Different projects use different wording in test steps!

---

### ✅ LEVEL 3: No Hardcoding (Good!)

**File:** `utils/step_executor.py` (Lines 429-478)

```python
def _try_level3_cv_guided(self, step_text: str, screenshot: bytes) -> tuple:
    # Uses CV vision API - fully dynamic
    cv_result = self.vision_client.identify_step_selector(
        screenshot, step_text, custom_selector, module_context
    )
    # No hardcoding! ✅
```

**Status:** ✅ Level 3 is completely dynamic - no hardcoding!

---

## 📊 Summary of Hardcoding

| Level | Component | Hardcoded? | Impact |
|-------|-----------|------------|--------|
| **L1** | Selector JSON file | ✅ Configurable | Good - project-specific file |
| **L1** | Keyword extraction | ❌ HARDCODED | Bad - must change code per project |
| **L2** | Generic patterns | ❌ HARDCODED | Bad - framework-specific |
| **L2** | Action detection | ❌ HARDCODED | Bad - project-specific keywords |
| **L3** | CV-guided | ✅ Dynamic | Good - no hardcoding |

**Overall:** L1 and L2 are NOT scalable due to extensive hardcoding!

---

## ✅ SOLUTION: Make It Scalable!

### Proposed Architecture

```
Program Code (Python files)
  ├─ Generic, reusable logic
  └─ NO project-specific details

Project Configuration (JSON/YAML files)
  ├─ selectors_enriched_all_modules.json  (L1 selectors)
  ├─ l2_generic_patterns.json             (L2 patterns) ← NEW!
  ├─ keyword_mappings.json                (L1 keywords) ← NEW!
  └─ action_detection_rules.json          (L2 detection) ← NEW!
```

**Benefits:**
- ✅ Program code NEVER changes between projects
- ✅ Only configuration files change
- ✅ Easy to add new projects (just create new config files)
- ✅ Fully scalable across frameworks (Angular, React, Vue, etc.)

---

## 📝 Detailed Solution

### 1. Extract L1 Keywords to Config File

**Create:** `keyword_mappings.json`

```json
{
  "metadata": {
    "project": "PLCD",
    "framework": "Angular Material",
    "version": "1.0.0"
  },
  "keyword_patterns": [
    {
      "trigger": ["... +", "more", "vertical"],
      "keywords": ["showmoreverticalbtn", "showmore", "vertical", "more"]
    },
    {
      "trigger": ["project", "product"],
      "keywords": ["selectproject", "project", "product"]
    },
    {
      "trigger": ["accordion"],
      "keywords": ["accordion", "panel"]
    },
    {
      "trigger": ["parts"],
      "keywords": ["parts"]
    },
    {
      "trigger": ["teststep"],
      "keywords": ["teststep"]
    }
  ],
  "action_keywords": {
    "save": ["save", "btn", "button"],
    "edit": ["edit", "btn", "button"],
    "delete": ["delete", "btn", "button"],
    "remove": ["remove", "btn", "button"],
    "close": ["close", "cancel", "btn"],
    "navigate": ["navigate", "nav", "menu"],
    "dropdown": ["dropdown", "select", "type"]
  }
}
```

**Updated Code:** `utils/selector_loader.py`

```python
class SelectorLoader:
    def __init__(self,
                 selectors_file: str = "Selectors_Folder/selectors_enriched_all_modules.json",
                 keywords_file: str = "Config/keyword_mappings.json"):  # NEW!
        self.logger = logging.getLogger("TA_AI_Project")
        self.selectors_file = Path(selectors_file)
        self.keywords_file = Path(keywords_file)
        self.selectors = []
        self.keyword_config = {}
        self.load_selectors()
        self.load_keyword_config()  # NEW!

    def load_keyword_config(self):
        """Load keyword mappings from config file."""
        try:
            if not self.keywords_file.exists():
                self.logger.warning(f"Keyword config not found: {self.keywords_file}")
                self.keyword_config = {"keyword_patterns": [], "action_keywords": {}}
                return

            with open(self.keywords_file, 'r', encoding='utf-8') as f:
                self.keyword_config = json.load(f)

            self.logger.info(f"Loaded keyword config from {self.keywords_file}")

        except Exception as e:
            self.logger.error(f"Error loading keyword config: {e}")
            self.keyword_config = {"keyword_patterns": [], "action_keywords": {}}

    def _extract_keywords(self, step_text: str) -> List[str]:
        """
        Extract keywords from step text using configuration.
        NO HARDCODING - reads from keyword_mappings.json
        """
        step_lower = step_text.lower()
        keywords = []

        # Use configured patterns (from JSON, not hardcoded!)
        for pattern in self.keyword_config.get('keyword_patterns', []):
            triggers = pattern.get('trigger', [])
            pattern_keywords = pattern.get('keywords', [])

            # Check if any trigger matches
            for trigger in triggers:
                if trigger.lower() in step_lower:
                    keywords.extend(pattern_keywords)
                    break

        # Use configured action keywords (from JSON, not hardcoded!)
        action_keywords = self.keyword_config.get('action_keywords', {})
        for action, action_kw_list in action_keywords.items():
            if action.lower() in step_lower:
                keywords.extend(action_kw_list)

        return keywords
```

---

### 2. Extract L2 Generic Patterns to Config File

**Create:** `l2_generic_patterns.json`

```json
{
  "metadata": {
    "project": "PLCD",
    "framework": "Angular Material",
    "version": "1.0.0",
    "description": "Generic HTML patterns for Level 2 selector fallback"
  },
  "action_patterns": {
    "button_click": [
      "button:has-text('{text}')",
      "a:has-text('{text}')",
      "[role='link']:has-text('{text}')",
      "[role='button']:has-text('{text}')",
      "button[type='submit']",
      "input[type='submit']",
      "button"
    ],
    "input_fill": [
      "input[type='text']",
      "input:not([type='hidden'])",
      "textarea"
    ],
    "dropdown_select": [
      "[data-attribute='{text}']",
      "[data-basicattribute='attribute'][aria-label*='{text}']",
      "input.mat-mdc-autocomplete-trigger[data-attribute='{text}']",
      "label:has-text('{text}') .mat-select",
      "label:has-text('{text}') ~ .mat-select",
      "label:has-text('{text}') input.mat-mdc-autocomplete-trigger",
      "div:has-text('{text}') >> .mat-select",
      "div:has-text('{text}') >> input[role='combobox']",
      ":text('{text}') >> xpath=.. >> .mat-select",
      "label:has-text('{text}') ~ select",
      "[role='combobox']",
      ".mat-select",
      "select"
    ],
    "accordion_expand": [
      "[role='button'][aria-expanded='false']",
      ".mat-expansion-panel-header:has-text('{text}')",
      ".accordion-header"
    ],
    "verify_message": [
      ":has-text('Successfully edited')",
      "div:has-text('Successfully edited')",
      ".mat-snack-bar-container",
      "[role='alert']",
      ".notification",
      ":text('Successfully')",
      "*"
    ]
  }
}
```

**Updated Code:** `utils/step_executor.py`

```python
class StepExecutor:
    def __init__(
        self,
        page: Page,
        vision_client: AzureVisionClient,
        selector_loader: SelectorLoader,
        config: Dict,
        logger: logging.Logger,
        module: Optional[str] = None,
        l2_patterns_file: str = "Config/l2_generic_patterns.json"  # NEW!
    ):
        self.page = page
        self.vision_client = vision_client
        self.selector_loader = selector_loader
        self.config = config
        self.logger = logger
        self.module = module
        self.module_mapper = ModuleMapper(config)

        # Load L2 patterns from config file (NO HARDCODING!)
        self.generic_patterns = self._load_l2_patterns(l2_patterns_file)

        if module:
            self.web_module_name = self.module_mapper.get_web_name(module)

    def _load_l2_patterns(self, patterns_file: str) -> Dict:
        """
        Load Level 2 generic patterns from config file.
        NO HARDCODING - reads from JSON!
        """
        try:
            patterns_path = Path(patterns_file)
            if not patterns_path.exists():
                self.logger.warning(f"L2 patterns file not found: {patterns_file}")
                return {}

            with open(patterns_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            patterns = data.get('action_patterns', {})
            self.logger.info(f"Loaded {len(patterns)} L2 action patterns from {patterns_file}")
            return patterns

        except Exception as e:
            self.logger.error(f"Error loading L2 patterns: {e}")
            return {}
```

---

### 3. Extract L2 Action Detection to Config File

**Create:** `action_detection_rules.json`

```json
{
  "metadata": {
    "project": "PLCD",
    "version": "1.0.0",
    "description": "Rules for detecting action types from step text"
  },
  "detection_rules": [
    {
      "action_type": "verify_message",
      "triggers": ["should be displayed", "message"],
      "exclusions": ["click"],
      "value_extraction": {
        "method": "quoted_text",
        "regex": "^\"([^\"]*)\""
      }
    },
    {
      "action_type": "dropdown_select",
      "triggers": ["dropdown", "select", "items per page"],
      "exclusions": ["accordion"],
      "priority": 1
    },
    {
      "action_type": "button_click",
      "triggers": ["navigate"],
      "requires_module": true,
      "value_extraction": {
        "method": "use_module_name"
      }
    },
    {
      "action_type": "button_click",
      "triggers": ["button", "btn", "click"],
      "button_keywords": ["save", "edit", "close", "cancel", "submit", "login", "add"]
    },
    {
      "action_type": "accordion_expand",
      "triggers": ["accordion"],
      "value_extraction": {
        "method": "keyword_match",
        "mappings": {
          "parts": "Parts"
        }
      }
    },
    {
      "action_type": "input_fill",
      "triggers": ["type", "enter", "fill"]
    }
  ]
}
```

---

### 4. Create Config Folder Structure

```
TA_AI_Project/
├── Config/                                    ← NEW FOLDER!
│   ├── keyword_mappings.json                 ← L1 keywords
│   ├── l2_generic_patterns.json              ← L2 patterns
│   ├── action_detection_rules.json           ← L2 action detection
│   └── README.md                             ← Documentation
├── Selectors_Folder/
│   └── selectors_enriched_all_modules.json   ← L1 selectors
├── utils/
│   ├── selector_loader.py                    ← Updated (no hardcoding)
│   └── step_executor.py                      ← Updated (no hardcoding)
└── run_test.py                               ← No changes needed
```

---

## 🎯 Benefits of This Approach

### Before (Hardcoded):
```
❌ Want to test a React app?
   → Must modify selector_loader.py code
   → Must modify step_executor.py code
   → Must rewrite generic_patterns dictionary
   → 2-3 hours of code changes!

❌ Want to test a different Angular app?
   → Must modify keyword extraction logic
   → Must update hardcoded patterns
   → 1-2 hours of code changes!
```

### After (Configurable):
```
✅ Want to test a React app?
   → Create new l2_generic_patterns.json (React selectors)
   → Create new keyword_mappings.json (React keywords)
   → Run selectors extractor on React codebase
   → 15-30 minutes of config file creation!
   → NO CODE CHANGES! ✅

✅ Want to test a different Angular app?
   → Create new keyword_mappings.json (app-specific)
   → Run selectors extractor on new codebase
   → 10-15 minutes of config file creation!
   → NO CODE CHANGES! ✅
```

**Improvement:** 2-3 hours → 15 minutes (10x faster!)

---

## 📊 Comparison

| Aspect | Before (Hardcoded) | After (Configurable) |
|--------|-------------------|----------------------|
| **Add new project** | Change Python code (2-3 hours) | Create config files (15 min) |
| **Change patterns** | Edit .py file, redeploy (1 hour) | Edit JSON file (5 min) |
| **Framework change** | Rewrite code (4-5 hours) | New config files (30 min) |
| **Scalability** | ❌ Not scalable | ✅ Fully scalable |
| **Maintenance** | ❌ Hard (code changes) | ✅ Easy (config changes) |
| **Reusability** | ❌ PLCD-specific | ✅ Works for any project |

---

## 🚀 Implementation Plan

### Phase 1: Create Config Files (30 min)
1. Create `Config/` folder
2. Extract current hardcoded values to JSON files:
   - `keyword_mappings.json`
   - `l2_generic_patterns.json`
   - `action_detection_rules.json`
3. Save files

### Phase 2: Update Python Code (45 min)
1. Update `selector_loader.py`:
   - Add `load_keyword_config()` method
   - Update `_extract_keywords()` to use config
2. Update `step_executor.py`:
   - Add `_load_l2_patterns()` method
   - Update `__init__()` to load patterns from file
   - (Optional) Add action detection from config

### Phase 3: Test (30 min)
1. Test with RBPLCD-8835
2. Test with RBPLCD-8862
3. Verify NO regression (should work same as before)

### Phase 4: Document (15 min)
1. Create `Config/README.md` with examples
2. Document how to create config files for new projects

**Total Time:** ~2 hours

---

## 📝 Config/README.md Example

```markdown
# Configuration Files for Test Automation

## Overview
This folder contains project-specific configuration files that control L1 and L2 selector strategies.

**NO CODE CHANGES NEEDED** - just update these JSON files for new projects!

## Files

### 1. keyword_mappings.json
Controls keyword extraction for L1 selector matching.

**When to update:**
- New project with different UI terminology
- Different domain-specific keywords

### 2. l2_generic_patterns.json
Controls generic HTML patterns for L2 fallback selectors.

**When to update:**
- Different framework (React, Vue, etc.)
- Different CSS class naming conventions

### 3. action_detection_rules.json
Controls how step text is analyzed to detect action types.

**When to update:**
- Different test step wording
- Different action types

## Creating Config for New Project

1. Copy existing config files
2. Update framework-specific selectors (e.g., `.mat-select` → `.MuiSelect-root` for Material-UI)
3. Update domain keywords (e.g., `parts` → `products`)
4. Run extractor to generate new selectors.json
5. Test!

**No Python code changes required!** ✅
```

---

## ✅ Summary

### Current State (Hardcoded):
- ❌ L1 keyword extraction: HARDCODED in Python
- ❌ L2 generic patterns: HARDCODED in Python
- ❌ L2 action detection: HARDCODED in Python
- ❌ Must change code for each new project
- ❌ Not scalable

### Proposed State (Configurable):
- ✅ L1 keyword extraction: Config file (JSON)
- ✅ L2 generic patterns: Config file (JSON)
- ✅ L2 action detection: Config file (JSON)
- ✅ NO code changes for new projects
- ✅ Fully scalable

**Next Steps:**
1. Get your approval for this approach
2. Create Config/ folder and JSON files
3. Update selector_loader.py and step_executor.py
4. Test with RBPLCD-8835 and RBPLCD-8862

**Ready to proceed?**
