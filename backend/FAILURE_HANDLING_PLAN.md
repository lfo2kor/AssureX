# Failure Handling Plan - Fail Fast with Actionable Guidance

## Current Problem

When L1, L2, and L3 ALL fail:
- ❌ Test continues to next step (doesn't terminate)
- ❌ No clear reason for failure
- ❌ No actionable recommendations
- ❌ User doesn't know what to fix

## Solution Design

### **Step 1: Detect Complete Failure**

In `step_executor.py`, add else clause after line 157:

```python
if success:
    result['status'] = 'PASSED'
    result['selector_used'] = selector_used
    result['level_used'] = level_used
    result['confidence'] = 0.95
else:
    # ALL 3 LEVELS FAILED - ANALYZE AND TERMINATE
    result['status'] = 'FAILED'
    result['failure_analysis'] = self._analyze_failure(step_text, step_num)

    # Log detailed failure
    self.logger.error("="*80)
    self.logger.error(f"CRITICAL FAILURE - Step {step_num} failed at ALL levels")
    self.logger.error("="*80)
    self.logger.error(f"Step: {step_text}")
    self.logger.error(f"\nFailure Analysis:")
    for key, value in result['failure_analysis'].items():
        self.logger.error(f"  {key}: {value}")

    # TERMINATE TEST IMMEDIATELY
    raise Exception(f"Step {step_num} failed at all levels. See failure analysis above.")
```

---

### **Step 2: Failure Analysis Method**

Add new method to analyze WHY it failed:

```python
def _analyze_failure(self, step_text: str, step_num: int) -> dict:
    """
    Analyze why all 3 levels failed and provide actionable recommendations.

    Returns:
        Dictionary with:
        - failure_reason: Specific reason (missing selector, wrong module, etc.)
        - l1_details: Why L1 failed
        - l2_details: Why L2 failed
        - l3_details: Why L3 failed
        - recommendations: List of specific actions to fix
        - missing_selector_template: JSON template to add
    """
    analysis = {
        'step_number': step_num,
        'step_text': step_text,
        'failure_reason': 'Unknown',
        'l1_details': {},
        'l2_details': {},
        'l3_details': {},
        'recommendations': [],
        'missing_selector_template': None
    }

    # Analyze L1 failure
    analysis['l1_details'] = self._analyze_l1_failure(step_text)

    # Analyze L2 failure
    analysis['l2_details'] = self._analyze_l2_failure(step_text)

    # Analyze L3 failure
    analysis['l3_details'] = self._analyze_l3_failure(step_text)

    # Determine primary failure reason
    if analysis['l1_details']['reason'] == 'no_selectors_found':
        analysis['failure_reason'] = 'MISSING_SELECTOR_IN_JSON'
        analysis['recommendations'].append(
            "1. Add the missing selector to JSON file manually"
        )
        analysis['recommendations'].append(
            "2. Or run: python extract_runtime_selectors.py <TICKET_ID>"
        )
        analysis['missing_selector_template'] = self._generate_selector_template(step_text)

    elif analysis['l1_details']['reason'] == 'selector_ambiguous':
        analysis['failure_reason'] = 'AMBIGUOUS_SELECTOR'
        analysis['recommendations'].append(
            f"1. Selector [{analysis['l1_details']['selector']}] matches multiple elements"
        )
        analysis['recommendations'].append(
            "2. Add tag/class to make it unique (e.g., input.mat-input[data-attr='value'])"
        )

    elif analysis['l1_details']['reason'] == 'selector_not_on_page':
        analysis['failure_reason'] = 'SELECTOR_NOT_VISIBLE'
        analysis['recommendations'].append(
            "1. Check if element is in wrong module (check sequential context)"
        )
        analysis['recommendations'].append(
            f"2. Current module: {self.module}, Expected: check Jira step"
        )

    elif analysis['l2_details']['reason'] == 'no_generic_pattern':
        analysis['failure_reason'] = 'NO_GENERIC_PATTERN_AVAILABLE'
        analysis['recommendations'].append(
            "1. This element type doesn't have a generic L2 pattern"
        )
        analysis['recommendations'].append(
            "2. You MUST add a custom selector to JSON for this element"
        )

    elif analysis['l3_details']['reason'] == 'cv_failed':
        analysis['failure_reason'] = 'ELEMENT_NOT_DETECTABLE'
        analysis['recommendations'].append(
            "1. Element might be hidden/invisible or in iframe"
        )
        analysis['recommendations'].append(
            "2. Check screenshot for visual clues"
        )
        analysis['recommendations'].append(
            "3. Element might need wait time or scroll"
        )

    return analysis

def _analyze_l1_failure(self, step_text: str) -> dict:
    """Analyze why L1 failed."""
    details = {
        'tried': False,
        'reason': 'not_tried',
        'keywords_extracted': [],
        'selectors_searched': 0,
        'selectors_found': 0,
        'selector': None,
        'count_on_page': 0
    }

    # Extract keywords that would be used
    keywords = self.selector_loader._extract_keywords(step_text)
    details['keywords_extracted'] = keywords

    # Find what L1 would search for
    selector_obj = self.selector_loader.find_best_selector(step_text, self.module)

    if not selector_obj:
        details['reason'] = 'no_selectors_found'
        details['tried'] = True
        return details

    details['tried'] = True
    details['selectors_found'] = 1

    # Build selector and check count
    selector_str = self.selector_loader.build_selector(selector_obj)
    details['selector'] = selector_str

    try:
        count = self.page.locator(selector_str).count()
        details['count_on_page'] = count

        if count == 0:
            details['reason'] = 'selector_not_on_page'
        elif count > 1:
            details['reason'] = 'selector_ambiguous'
        else:
            details['reason'] = 'unknown'  # Found 1, but still failed?
    except:
        details['reason'] = 'invalid_selector'

    return details

def _analyze_l2_failure(self, step_text: str) -> dict:
    """Analyze why L2 failed."""
    details = {
        'tried': True,
        'reason': 'unknown',
        'patterns_tried': [],
        'best_match_count': 0
    }

    # Determine what type of action this is
    step_lower = step_text.lower()

    if 'navigate' in step_lower or 'click on' in step_lower and 'button' in step_lower:
        details['patterns_tried'] = ['button:has-text', 'a:has-text', '[role="link"]']
    elif 'select' in step_lower and 'dropdown' in step_lower:
        details['patterns_tried'] = ['[data-attribute]', 'input.mat-autocomplete', '.mat-select']
    elif 'enter' in step_lower or 'type' in step_lower:
        details['patterns_tried'] = ['input[name]', 'input[placeholder]', 'input[type="text"]']
    else:
        details['patterns_tried'] = ['Generic button/link patterns']

    # L2 failed means no pattern matched with count=1
    details['reason'] = 'no_generic_pattern' if not details['patterns_tried'] else 'pattern_ambiguous_or_not_found'

    return details

def _analyze_l3_failure(self, step_text: str) -> dict:
    """Analyze why L3 (CV) failed."""
    details = {
        'tried': True,
        'reason': 'cv_failed',
        'cv_response': None,
        'selectors_tried': []
    }

    # L3 failure means CV couldn't find it or CV selector didn't work
    # This is usually element visibility or iframe issues

    return details

def _generate_selector_template(self, step_text: str) -> dict:
    """Generate JSON template for missing selector."""

    # Extract action type
    step_lower = step_text.lower()

    if 'button' in step_lower or 'click' in step_lower:
        tag = 'button'
        context = ['btn', 'button', 'click']
    elif 'input' in step_lower or 'enter' in step_lower or 'type' in step_lower:
        tag = 'input'
        context = ['input', 'field']
    elif 'dropdown' in step_lower or 'select' in step_lower:
        tag = 'mat-select'
        context = ['dropdown', 'select']
    else:
        tag = 'UNKNOWN'
        context = []

    template = {
        "attr": "data-FIXME",
        "value": "FIXME",
        "tagName": tag,
        "className": "",
        "module": self.module,
        "context": context,
        "priority": 20,
        "isClickable": True,
        "isVisible": True,
        "source": "manual_fix",
        "label": f"FIXME: {step_text[:50]}"
    }

    return template
```

---

### **Step 3: Failure Report Generation**

When test terminates, save detailed failure report:

```python
def _save_failure_report(self, step_num: int, analysis: dict):
    """Save detailed failure report to file."""

    report_path = Path('Failure_Reports') / f'failure_step_{step_num}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Also create human-readable text report
    text_path = report_path.with_suffix('.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"FAILURE ANALYSIS - Step {step_num}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Step Text: {analysis['step_text']}\n\n")
        f.write(f"Failure Reason: {analysis['failure_reason']}\n\n")

        f.write("L1 Analysis:\n")
        f.write(f"  Keywords extracted: {analysis['l1_details']['keywords_extracted']}\n")
        f.write(f"  Reason: {analysis['l1_details']['reason']}\n")
        if analysis['l1_details']['selector']:
            f.write(f"  Selector tried: {analysis['l1_details']['selector']}\n")
            f.write(f"  Count on page: {analysis['l1_details']['count_on_page']}\n")
        f.write("\n")

        f.write("L2 Analysis:\n")
        f.write(f"  Patterns tried: {analysis['l2_details']['patterns_tried']}\n")
        f.write(f"  Reason: {analysis['l2_details']['reason']}\n\n")

        f.write("L3 Analysis:\n")
        f.write(f"  Reason: {analysis['l3_details']['reason']}\n\n")

        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n")
        for i, rec in enumerate(analysis['recommendations'], 1):
            f.write(f"{i}. {rec}\n")

        if analysis['missing_selector_template']:
            f.write("\n" + "="*80 + "\n")
            f.write("MISSING SELECTOR TEMPLATE:\n")
            f.write("="*80 + "\n")
            f.write("Add this to your JSON file:\n\n")
            f.write(json.dumps(analysis['missing_selector_template'], indent=2))
            f.write("\n")

    self.logger.error(f"\nFailure report saved to: {text_path}")
    return str(text_path)
```

---

## Expected Behavior After Fix

### When Step Fails at All Levels:

```
[ERROR] ========================================================================
[ERROR] CRITICAL FAILURE - Step 4 failed at ALL levels
[ERROR] ========================================================================
[ERROR] Step: Select "Project" from the drop down and click on it.
[ERROR]
[ERROR] Failure Analysis:
[ERROR]   failure_reason: MISSING_SELECTOR_IN_JSON
[ERROR]   step_number: 4
[ERROR]
[ERROR] L1 Analysis:
[ERROR]   Keywords extracted: ['dropdown', 'select', 'project']
[ERROR]   Reason: no_selectors_found
[ERROR]   Selectors searched: 1331
[ERROR]   Selectors found: 0
[ERROR]
[ERROR] L2 Analysis:
[ERROR]   Patterns tried: ['[data-attribute]', 'input.mat-autocomplete', '.mat-select']
[ERROR]   Reason: pattern_ambiguous_or_not_found
[ERROR]
[ERROR] L3 Analysis:
[ERROR]   Reason: cv_failed
[ERROR]
[ERROR] RECOMMENDATIONS:
[ERROR]   1. Add the missing selector to JSON file manually
[ERROR]   2. Or run: python extract_runtime_selectors.py RBPLCD-8862
[ERROR]
[ERROR] MISSING SELECTOR TEMPLATE:
[ERROR]   {
[ERROR]     "attr": "data-opencreatedialogdropdown",
[ERROR]     "value": "aeName.StructureLevel.name",
[ERROR]     "tagName": "mat-select",
[ERROR]     "module": "Teststep",
[ERROR]     "context": ["dropdown", "select"],
[ERROR]     "priority": 20
[ERROR]   }
[ERROR]
[ERROR] Failure report saved to: Failure_Reports/failure_step_4_20251104_162000.txt
[ERROR] ========================================================================
[ERROR] TEST TERMINATED - Fix the issue above and restart
[ERROR] ========================================================================

Exception: Step 4 failed at all levels. See failure analysis above.
```

---

## Implementation Files

### Files to Modify:
1. **`utils/step_executor.py`**
   - Add else clause after line 157
   - Add `_analyze_failure()` method
   - Add `_analyze_l1_failure()` method
   - Add `_analyze_l2_failure()` method
   - Add `_analyze_l3_failure()` method
   - Add `_generate_selector_template()` method
   - Add `_save_failure_report()` method

2. **Create `Failure_Reports/` directory**
   - Store JSON and TXT failure reports

---

## Benefits

### For User:
1. ✅ **Immediate termination** - Don't waste time on remaining steps
2. ✅ **Clear failure reason** - Know exactly why it failed
3. ✅ **Actionable guidance** - Know exactly what to fix
4. ✅ **Selector template** - Copy-paste ready JSON to add
5. ✅ **Failure history** - All failures logged for analysis

### For Debugging:
1. ✅ **L1/L2/L3 breakdown** - See which level failed and why
2. ✅ **Keywords extracted** - Verify L1 search logic
3. ✅ **Selector count** - Know if ambiguous (2+) or missing (0)
4. ✅ **Screenshot available** - Visual confirmation of page state

---

## Example Failure Scenarios

### Scenario 1: Missing Selector
```
Failure Reason: MISSING_SELECTOR_IN_JSON
L1: no_selectors_found (searched: 1331, found: 0)
L2: pattern_ambiguous_or_not_found
L3: cv_failed

Recommendation:
1. Add selector to JSON manually
2. Template provided below

{
  "attr": "data-opencreatedialogdropdown",
  "value": "aeName.StructureLevel.name",
  ...
}
```

### Scenario 2: Ambiguous Selector
```
Failure Reason: AMBIGUOUS_SELECTOR
L1: selector_ambiguous (count: 9)
    Selector: [data-test="sidebar-nav-item-undefined"]
L2: pattern_ambiguous_or_not_found
L3: cv_failed

Recommendation:
1. Selector matches 9 elements - too generic
2. Add tag/class: mat-list-item[data-test="sidebar-nav-item-undefined"]
3. Or use more specific value instead of "undefined"
```

### Scenario 3: Wrong Module
```
Failure Reason: SELECTOR_NOT_VISIBLE
L1: selector_not_on_page (count: 0)
    Selector: [data-savebtn="SaveBtn"]
    Current module: CreateNew
    Expected module: DetailView
L2: pattern_ambiguous_or_not_found
L3: cv_failed

Recommendation:
1. Selector exists but wrong module
2. Check sequential context tracking
3. Previous step may not have transitioned state correctly
```

---

## Timeline

**Immediate (< 30 minutes):**
1. Add else clause with failure detection
2. Add basic failure logging
3. Add test termination

**Short-term (< 2 hours):**
1. Implement full failure analysis methods
2. Add selector template generation
3. Add failure report saving

**Would you like me to implement this now?**
