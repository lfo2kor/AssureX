# Safe Upgrade Implementation - Option 2 (Quick Win)

## ✅ What Will Be Modified

### Files to Modify (Only 2!):
```
utils/selector_loader.py       → Update to load new JSON format
utils/step_executor.py         → Improve L1 logic for dynamic selectors
```

### Files That Will NOT Change:
```
run_test.py                    → NO CHANGES ✅
vision_executor_agent.py       → NO CHANGES ✅
All other files                → NO CHANGES ✅
```

---

## 🛡️ Safety Backups Created

**Backup Files (Can restore anytime):**
```
utils/selector_loader_BACKUP_OLD.py     ✅ Created
utils/step_executor_BACKUP_OLD.py       ✅ Created
```

**How to Rollback if Needed:**
```bash
# If new approach fails, restore from backup:
cp utils/selector_loader_BACKUP_OLD.py utils/selector_loader.py
cp utils/step_executor_BACKUP_OLD.py utils/step_executor.py
```

---

## 📂 Selector Files

**New File (Will use):**
```
Selectors_Folder/selectors_enriched_all_modules.json
  - 884 selectors
  - Static: 595 (67.3%)
  - Dynamic: 289 (32.7%)
  - Dynamic with possibleValues: 7
```

**Old File (Kept as backup):**
```
Selectors_Folder/selectors.json
  - 888 selectors
  - Has broken dynamic selectors (attr.data-X format)
  - Will keep unchanged as fallback
```

---

## 🔄 How It Works

### Before Changes:
```
run_test.py
  → vision_executor_agent.py
     → StepExecutor (step_executor.py)
        → SelectorLoader (selector_loader.py)
           → Loads: Selectors_Folder/selectors.json (OLD)
```

### After Changes:
```
run_test.py  (NO CHANGE!)
  → vision_executor_agent.py  (NO CHANGE!)
     → StepExecutor (step_executor.py - UPDATED!)
        → SelectorLoader (selector_loader.py - UPDATED!)
           → Loads: Selectors_Folder/selectors_enriched_all_modules.json (NEW!)
```

**Result:** run_test.py works exactly the same way, just uses better selectors!

---

## 📝 Changes Details

### Change 1: selector_loader.py

**Line 19 (Change default file path):**
```python
# OLD:
def __init__(self, selectors_file: str = "Selectors_Folder/selectors.json"):

# NEW:
def __init__(self, selectors_file: str = "Selectors_Folder/selectors_enriched_all_modules.json"):
```

**Lines 31-45 (Handle new JSON format):**
```python
# OLD:
def load_selectors(self):
    with open(self.selectors_file, 'r', encoding='utf-8') as f:
        self.selectors = json.load(f)
    self.logger.info(f"Loaded {len(self.selectors)} selectors")

# NEW:
def load_selectors(self):
    with open(self.selectors_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # New format: object with metadata + selectors
        if isinstance(data, dict) and 'selectors' in data:
            self.selectors = data['selectors']
            self.metadata = data.get('metadata', {})
            self.logger.info(f"Loaded enriched selectors (Phase: {self.metadata.get('phase', 'unknown')})")
        else:
            # Old format: array
            self.selectors = data
            self.metadata = {}

    self.logger.info(f"Loaded {len(self.selectors)} selectors")
```

**Lines 101-131 (Update build_selector to handle new format):**
```python
# OLD:
def build_selector(self, selector_obj: Dict[str, Any]) -> str:
    attr = selector_obj.get('attr', '')
    value = selector_obj.get('value', '')
    is_dynamic = selector_obj.get('dynamic', False)

    if attr.startswith('attr.'):
        attr = attr.replace('attr.', '')
        if is_dynamic:
            return f"[{attr}]"
        else:
            return f'[{attr}="{value}"]'
    else:
        if is_dynamic:
            return f"[{attr}]"
        else:
            return f'[{attr}="{value}"]'

# NEW:
def build_selector(self, selector_obj: Dict[str, Any], value_override: str = None) -> str:
    """
    Build selector with support for possibleValues.

    Args:
        selector_obj: Selector from JSON
        value_override: Optional value to use (for trying possibleValues)
    """
    attr = selector_obj.get('attr', '')
    value = value_override or selector_obj.get('value', '')

    # New format uses 'isDynamic' instead of 'dynamic'
    is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

    # New format has clean attr names (no 'attr.' prefix needed)
    if is_dynamic and not value_override:
        # Dynamic without specific value - wildcard
        return f"[{attr}]"
    else:
        # Static OR dynamic with specific value
        clean_value = value.replace('{{', '').replace('}}', '')
        return f'[{attr}="{clean_value}"]'
```

---

### Change 2: step_executor.py

**Lines 212-266 (Improve L1 to handle possibleValues):**
```python
# OLD:
def _try_level1_custom_selectors(self, step_text: str) -> tuple:
    selector_obj = self.selector_loader.find_best_selector(step_text, self.module)
    if selector_obj:
        selector_str = self.selector_loader.build_selector(selector_obj)
        count = self.page.locator(selector_str).count()
        if count == 1:
            return self._execute_action(step_text, selector_str)
    return (False, "")

# NEW:
def _try_level1_custom_selectors(self, step_text: str) -> tuple:
    """
    L1 with improved dynamic selector handling.

    Strategy:
    1. Static selectors → try exact match
    2. Dynamic with possibleValues → try each value
    3. Dynamic without possibleValues → try wildcard
    """
    # Check if row scoping needed (skip L1)
    import re
    if 'named as' in step_text.lower() or 'of part' in step_text.lower():
        return (False, "")

    # Find best selector
    selector_obj = self.selector_loader.find_best_selector(step_text, self.module)
    if not selector_obj:
        return (False, "")

    is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

    if not is_dynamic:
        # STATIC SELECTOR - exact match
        selector_str = self.selector_loader.build_selector(selector_obj)
        self.logger.info(f"Trying static selector: {selector_str}")

        count = self.page.locator(selector_str).count()
        if count == 1:
            return self._execute_action(step_text, selector_str)
        elif count > 1:
            self.logger.warning(f"Multiple matches ({count})")
        return (False, "")

    else:
        # DYNAMIC SELECTOR
        possible_values = selector_obj.get('possibleValues', [])

        if possible_values:
            # Try each possible value
            self.logger.info(f"Dynamic selector with {len(possible_values)} possible values")
            for value in possible_values:
                selector_str = self.selector_loader.build_selector(selector_obj, value_override=value)
                self.logger.info(f"Trying: {selector_str}")

                count = self.page.locator(selector_str).count()
                if count > 0:
                    return self._execute_action(step_text, selector_str)

            return (False, "")

        else:
            # Dynamic without values - wildcard
            selector_str = self.selector_loader.build_selector(selector_obj)
            self.logger.info(f"Trying dynamic wildcard: {selector_str}")

            count = self.page.locator(selector_str).count()
            if count == 1:
                return self._execute_action(step_text, selector_str)
            elif count > 1:
                self.logger.warning(f"Multiple matches ({count})")
            return (False, "")
```

---

## 🎯 Expected Improvements

### L1 Success Rate:
```
Before: ~25%
  - Static: ~70% (works OK)
  - Dynamic: ~5% (BROKEN - wrong attr names)

After: ~70%
  - Static: ~85-90% (improved coverage)
  - Dynamic with possibleValues: ~70-80% (NEW!)
  - Dynamic wildcard: ~40-50% (improved from 5%)

Improvement: 3x better! ✅
```

### Execution Speed:
```
L1: 50-100ms (fast)
L2: 200-500ms (slower)
L3: 2-5 seconds (slowest)

More L1 success = Faster overall execution ✅
```

---

## 🧪 Testing Plan

### Test 1: RBPLCD-8835
```bash
python run_test.py RBPLCD-8835
```

**Check:**
- How many steps use L1? (expect 60-70%)
- How many steps fall to L2? (expect 20-30%)
- How many steps fall to L3? (expect 5-10%)
- Overall success rate?

### Test 2: RBPLCD-8862
```bash
python run_test.py RBPLCD-8862
```

**Check:**
- Same metrics as Test 1
- Compare with previous test results

---

## 📊 How to Measure Success

**Look at log file after running test:**

```
Logs/RBPLCD-8835_*.log
```

**Search for:**
```
"Level 1 succeeded"   → Count these (L1 success)
"Level 2 succeeded"   → Count these (L2 fallback)
"Level 3 succeeded"   → Count these (L3 fallback)
```

**Calculate:**
```
L1 Success Rate = (L1 successes / Total steps) * 100%

Target: 60-70%+ (currently ~25%)
```

---

## 🔄 Rollback Instructions (If Needed)

**If new approach causes issues:**

```bash
# Step 1: Restore old files
cp utils/selector_loader_BACKUP_OLD.py utils/selector_loader.py
cp utils/step_executor_BACKUP_OLD.py utils/step_executor.py

# Step 2: Test again
python run_test.py RBPLCD-8835

# Done! Back to working state
```

**Note:** run_test.py was NEVER changed, so no need to restore it!

---

## 📋 Implementation Steps

### Step 1: Update selector_loader.py ✅ (Ready to execute)
- Change default file path
- Handle new JSON format
- Update build_selector() method

### Step 2: Update step_executor.py ✅ (Ready to execute)
- Improve _try_level1_custom_selectors()
- Add possibleValues logic
- Add dynamic wildcard logic

### Step 3: Test with RBPLCD-8835 ⏳
- Run test
- Check L1/L2/L3 usage in logs
- Measure success rate

### Step 4: Test with RBPLCD-8862 ⏳
- Run test
- Compare results

### Step 5: Analyze and Document ⏳
- Calculate improvements
- Document findings

---

## ✅ Ready to Proceed?

**What happens next:**
1. I'll update `selector_loader.py` (3 changes)
2. I'll update `step_executor.py` (1 change)
3. You run: `python run_test.py RBPLCD-8835`
4. We check if L1 success rate improved

**Safety:**
- ✅ Backups created
- ✅ run_test.py unchanged
- ✅ Can rollback anytime
- ✅ Old selector file preserved

**Shall I proceed with updating the 2 files?**
