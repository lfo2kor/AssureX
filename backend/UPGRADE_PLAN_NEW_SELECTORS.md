# Upgrade Plan: New Selector File Integration

## Objective
Upgrade L1/L2/L3 system to use the new `selectors_enriched_all_modules.json` file to improve L1 success rate from ~25% to ~70%.

## Current Architecture

### Files Involved:
1. **run_test.py** - Main test runner
2. **vision_executor_agent.py** - Orchestrates browser automation
3. **step_executor.py** - Implements 3-level selector strategy
4. **selector_loader.py** - Loads and searches selectors from JSON

### Current L1/L2/L3 Flow:
```
Step Execution
    ↓
Level 1: Try custom selectors from selectors.json
    ↓ (if fails)
Level 2: Try generic HTML patterns (hardcoded)
    ↓ (if fails)
Level 3: Use CV-guided selector discovery
```

### Current Selector File:
- **File:** `Selectors_Folder/selectors.json` (OLD)
- **Format:** Array of selectors with wrong attr names for dynamic selectors
- **Problems:**
  - Dynamic selectors have `"attr": "attr.data-X"` (BROKEN!)
  - No possible values for dynamic selectors
  - No element types
  - L1 success rate: ~25%

---

## New Architecture

### New Selector File:
- **File:** `Selectors_Folder/selectors_enriched_all_modules.json` (NEW)
- **Format:** Object with metadata + selectors array
- **Benefits:**
  - Correct attr names: `"attr": "data-X"` (FIXED!)
  - Has `possibleValues` array for 7 dynamic selectors
  - Has `elementType` for validation
  - Has `isDynamic` flag
  - L1 expected success rate: ~70%

### Updated L1/L2/L3 Flow:
```
Step Execution
    ↓
Level 1: Try ENRICHED custom selectors
    ├─ Try static selectors (exact match)
    ├─ Try dynamic selectors with possibleValues (try each value)
    └─ Try dynamic selectors with wildcard (if unique)
    ↓ (if fails)
Level 2: Try generic HTML patterns (unchanged)
    ↓ (if fails)
Level 3: Use CV-guided selector discovery (unchanged)
```

---

## Changes Required

### 1. Update selector_loader.py

**Current Code (Line 19):**
```python
def __init__(self, selectors_file: str = "Selectors_Folder/selectors.json"):
```

**New Code:**
```python
def __init__(self, selectors_file: str = "Selectors_Folder/selectors_enriched_all_modules.json"):
```

**Current Code (Line 38-39):**
```python
with open(self.selectors_file, 'r', encoding='utf-8') as f:
    self.selectors = json.load(f)
```

**New Code:**
```python
with open(self.selectors_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    # New format has metadata + selectors array
    if isinstance(data, dict) and 'selectors' in data:
        self.selectors = data['selectors']
        self.metadata = data['metadata']
        self.logger.info(f"Loaded enriched selectors (Phase: {self.metadata.get('phase', 'unknown')})")
    else:
        # Fallback to old format (array)
        self.selectors = data
        self.metadata = {}
```

**Current Code (Line 101-130):**
```python
def build_selector(self, selector_obj: Dict[str, Any]) -> str:
    attr = selector_obj.get('attr', '')
    value = selector_obj.get('value', '')
    is_dynamic = selector_obj.get('dynamic', False)

    # Handle attribute selectors
    if attr.startswith('attr.'):
        # Dynamic attribute - just use attribute name
        attr = attr.replace('attr.', '')
        if is_dynamic:
            # Dynamic value - just check attribute exists
            return f"[{attr}]"
        else:
            # Static value
            return f'[{attr}="{value}"]'
    else:
        # Regular data attribute
        if is_dynamic:
            return f"[{attr}]"
        else:
            return f'[{attr}="{value}"]'
```

**New Code:**
```python
def build_selector(self, selector_obj: Dict[str, Any], value_override: str = None) -> str:
    """
    Build Playwright selector string from selector object.

    Args:
        selector_obj: Selector dictionary from JSON
        value_override: Optional value to use instead of selector's value
                       (used when trying possibleValues)

    Returns:
        CSS selector string for Playwright
    """
    attr = selector_obj.get('attr', '')
    value = value_override or selector_obj.get('value', '')

    # New format uses 'isDynamic' instead of 'dynamic'
    is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

    # New format has clean attribute names (no 'attr.' prefix)
    # OLD: "attr": "attr.data-X" -> BROKEN
    # NEW: "attr": "data-X" -> CORRECT

    if is_dynamic and not value_override:
        # Dynamic selector without specific value - use wildcard
        return f"[{attr}]"
    else:
        # Static selector OR dynamic with specific value
        # Remove {{...}} markers if present
        clean_value = value.replace('{{', '').replace('}}', '')
        return f'[{attr}="{clean_value}"]'
```

---

### 2. Update step_executor.py - Improve L1

**Current Code (Line 212-266):**
```python
def _try_level1_custom_selectors(self, step_text: str) -> tuple:
    # ... current logic (doesn't handle possibleValues)
```

**New Code:**
```python
def _try_level1_custom_selectors(self, step_text: str) -> tuple:
    """
    Level 1: Try custom selectors from enriched JSON.

    Strategy:
    1. Find best matching selector
    2. If static -> try exact match
    3. If dynamic with possibleValues -> try each value
    4. If dynamic without possibleValues -> try wildcard

    Returns:
        Tuple of (success, selector_string)
    """
    try:
        # Check if step requires row scoping (skip Level 1, use Level 2)
        import re
        step_lower = step_text.lower()
        row_identifier = None

        if 'named as' in step_lower:
            match = re.search(r'named as\s+(\S+)', step_text, re.IGNORECASE)
            if match:
                row_identifier = match.group(1)
        elif 'of part' in step_lower or 'of testobject' in step_lower:
            match = re.search(r'of (?:part|testobject)\s+(\S+)', step_text, re.IGNORECASE)
            if match:
                row_identifier = match.group(1)

        if row_identifier:
            self.logger.info(f"Row scoping required - skipping Level 1")
            return (False, "")

        # Find best matching selector
        selector_obj = self.selector_loader.find_best_selector(step_text, self.module)

        if not selector_obj:
            self.logger.info("No matching selector found in JSON")
            return (False, "")

        # Check if selector is dynamic
        is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

        if not is_dynamic:
            # STATIC SELECTOR - Try exact match
            selector_str = self.selector_loader.build_selector(selector_obj)
            self.logger.info(f"Trying static selector: {selector_str}")

            count = self.page.locator(selector_str).count()
            self.logger.info(f"Selector count: {count}")

            if count == 1:
                return self._execute_action(step_text, selector_str)
            elif count > 1:
                self.logger.warning(f"Multiple matches ({count}) - skipping to Level 3")
                return (False, "")
            else:
                return (False, "")

        else:
            # DYNAMIC SELECTOR
            possible_values = selector_obj.get('possibleValues', [])

            if possible_values:
                # DYNAMIC WITH POSSIBLE VALUES - Try each value
                self.logger.info(f"Dynamic selector with {len(possible_values)} possible values")

                for value in possible_values:
                    selector_str = self.selector_loader.build_selector(selector_obj, value_override=value)
                    self.logger.info(f"Trying dynamic selector with value '{value}': {selector_str}")

                    count = self.page.locator(selector_str).count()
                    self.logger.info(f"Selector count: {count}")

                    if count > 0:
                        return self._execute_action(step_text, selector_str)

                self.logger.info("No possible values matched")
                return (False, "")

            else:
                # DYNAMIC WITHOUT VALUES - Try wildcard
                selector_str = self.selector_loader.build_selector(selector_obj)
                self.logger.info(f"Trying dynamic selector (wildcard): {selector_str}")

                count = self.page.locator(selector_str).count()
                self.logger.info(f"Selector count: {count}")

                if count == 1:
                    # Unique match with wildcard
                    return self._execute_action(step_text, selector_str)
                elif count > 1:
                    self.logger.warning(f"Multiple matches ({count}) - skipping to Level 3")
                    return (False, "")
                else:
                    return (False, "")

    except Exception as e:
        self.logger.error(f"Level 1 error: {e}")

    return (False, "")
```

---

### 3. No Changes Needed

**Files that DON'T need changes:**
- ✅ `run_test.py` - Already calls step_executor correctly
- ✅ `vision_executor_agent.py` - Already initializes StepExecutor correctly
- ✅ Level 2 logic in `step_executor.py` - Generic patterns still work
- ✅ Level 3 logic in `step_executor.py` - CV-guided still works

---

## Expected Improvements

### Before (Old selectors.json):
```
L1 Success Rate: ~25%
  - Static selectors: ~70% (works OK)
  - Dynamic selectors: ~5% (BROKEN - wrong attr names)

L2 Success Rate: ~40%
L3 Success Rate: ~80%

Overall: L1 (25%) → L2 (40%) → L3 (80%)
```

### After (New selectors_enriched_all_modules.json):
```
L1 Success Rate: ~70% ✅
  - Static selectors: ~85-90% (better coverage)
  - Dynamic with possibleValues: ~70-80% (NEW!)
  - Dynamic wildcard: ~40-50% (improved from 5%)

L2 Success Rate: ~40% (unchanged)
L3 Success Rate: ~80% (unchanged)

Overall: L1 (70%) → L2 (40%) → L3 (80%)
```

**Key Improvement:** L1 goes from 25% → 70% (3x better!)

---

## Testing Plan

### Test with RBPLCD-8835:
1. Run with OLD selector file
   - Measure L1/L2/L3 usage
   - Record success rate per step
2. Switch to NEW selector file
   - Measure L1/L2/L3 usage
   - Record success rate per step
3. Compare results

### Test with RBPLCD-8862:
1. Run with NEW selector file
   - Measure L1/L2/L3 usage
   - Record success rate per step
2. Verify improvements

### Metrics to Track:
- **L1 success rate:** % of steps that succeeded at Level 1
- **L2 fallback rate:** % of steps that fell through to Level 2
- **L3 fallback rate:** % of steps that fell through to Level 3
- **Overall success rate:** % of all steps that passed
- **Execution time:** Average time per step (L1 should be faster)

---

## Implementation Steps

### Step 1: Backup Old Files
```bash
cp utils/selector_loader.py utils/selector_loader.py.backup
cp utils/step_executor.py utils/step_executor.py.backup
```

### Step 2: Update selector_loader.py
- Change default file path
- Update load_selectors() to handle new format
- Update build_selector() to support value_override

### Step 3: Update step_executor.py
- Improve _try_level1_custom_selectors()
- Add logic for possibleValues
- Add logic for dynamic wildcards

### Step 4: Test with RBPLCD-8835
```bash
python run_test.py RBPLCD-8835
```

### Step 5: Analyze Results
- Check logs for L1/L2/L3 usage
- Compare with previous runs

### Step 6: Test with RBPLCD-8862
```bash
python run_test.py RBPLCD-8862
```

### Step 7: Document Results
- Create comparison report
- Update documentation

---

## Risk Assessment

### Low Risk:
- ✅ Backward compatible (can fall back to old file if needed)
- ✅ L2 and L3 unchanged (safety net)
- ✅ Can revert changes easily (have backups)

### Medium Risk:
- ⚠️ New selector format might have edge cases
- ⚠️ Dynamic selector logic needs testing

### Mitigation:
- Test with multiple tickets (RBPLCD-8835, RBPLCD-8862)
- Keep old selector file as backup
- Monitor logs closely during testing

---

## Success Criteria

### Must Have:
- ✅ L1 success rate improves from 25% to at least 60%
- ✅ No regression in L2/L3 performance
- ✅ All steps in RBPLCD-8835 pass
- ✅ All steps in RBPLCD-8862 pass

### Nice to Have:
- ✅ L1 success rate reaches 70%+
- ✅ Faster execution time (L1 is faster than L2/L3)
- ✅ Better logging for debugging

---

## Rollback Plan

If new selector file doesn't work:

**Option 1: Revert selector_loader.py**
```python
# Change line 19 back to:
def __init__(self, selectors_file: str = "Selectors_Folder/selectors.json"):
```

**Option 2: Use both files**
```python
# Try new file first, fall back to old:
def __init__(self):
    new_file = "Selectors_Folder/selectors_enriched_all_modules.json"
    old_file = "Selectors_Folder/selectors.json"

    if Path(new_file).exists():
        self.load_new_format(new_file)
    else:
        self.load_old_format(old_file)
```

---

## Timeline

### Immediate (Next 30-45 minutes):
1. Update selector_loader.py (10 min)
2. Update step_executor.py (15 min)
3. Test with RBPLCD-8835 (10 min)
4. Review and fix issues (10 min)

### Follow-up (Next session):
1. Test with RBPLCD-8862
2. Analyze performance improvements
3. Document results
4. Optimize further if needed

---

## Next Steps

1. **Get user approval** for the plan
2. **Update selector_loader.py** with new logic
3. **Update step_executor.py** with improved L1
4. **Test with RBPLCD-8835**
5. **Test with RBPLCD-8862**
6. **Compare and document results**

---

**Ready to proceed with implementation?**
