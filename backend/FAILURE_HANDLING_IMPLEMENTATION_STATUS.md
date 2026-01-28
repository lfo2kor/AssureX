# Failure Handling Implementation - Current Status

## What Was Completed (✅)

### Phase 1: Fail-Fast Termination ✅ DONE
**Files Modified:**
1. `utils/step_executor.py` (lines 158-182)
   - Added else clause after line 157
   - Logs detailed failure message when all 3 levels fail
   - Sets result['status'] = 'FAILED'
   - Sets result['error'] = 'All 3 selector levels (L1, L2, L3) failed'

2. `agents/vision_executor_agent.py` (lines 163-176)
   - Detects when error is "All 3 selector levels (L1, L2, L3) failed"
   - Logs termination message
   - Raises Exception to stop test execution immediately

**Result:** Test now terminates immediately when step fails at all levels

---

### Phase 2: Failure Analyzer ✅ CREATED
**New File Created:**
- `utils/failure_analyzer.py` (519 lines)

**Contains:**
1. `FailureAnalyzer` class with methods:
   - `analyze_failure()` - Main analysis orchestrator
   - `_analyze_l1_failure()` - Why L1 failed (no selectors, ambiguous, not visible)
   - `_analyze_l2_failure()` - Why L2 failed (no generic pattern)
   - `_analyze_l3_failure()` - Why L3/CV failed
   - `_generate_selector_template()` - Creates JSON template
   - `save_failure_report()` - Saves JSON and TXT reports

**Features:**
- Extracts keywords that L1 would use
- Checks selector count on page (0, 1, or multiple)
- Identifies failure reason: MISSING_SELECTOR_IN_JSON | AMBIGUOUS_SELECTOR | SELECTOR_NOT_VISIBLE
- Generates actionable recommendations
- Creates copy-paste ready JSON template
- Saves reports to Failure_Reports/ directory

**Status:** File created, BUT NOT YET INTEGRATED into step_executor.py

---

## What Needs To Be Done (⏳)

### Phase 2 Completion: Integrate FailureAnalyzer
**TODO:**
1. Import FailureAnalyzer in step_executor.py ✅ DONE (line 19)
2. Initialize FailureAnalyzer in StepExecutor.__init__() ❌ NOT DONE
3. Call analyzer in step_executor.py else clause (around line 158) ❌ NOT DONE
4. Pass analysis results to vision_executor_agent.py ❌ NOT DONE

**Code to add in StepExecutor.__init__() (~line 40):**
```python
# Initialize failure analyzer
self.failure_analyzer = FailureAnalyzer(
    selector_loader=self.selector_loader,
    page=self.page,
    module=self.module,
    logger=self.logger
)
```

**Code to add in step_executor.py else clause (~line 161):**
```python
else:
    # ALL 3 LEVELS FAILED - CRITICAL FAILURE
    result['status'] = 'FAILED'
    result['error'] = 'All 3 selector levels (L1, L2, L3) failed'

    # ANALYZE WHY IT FAILED
    analysis = self.failure_analyzer.analyze_failure(step_text, step_num)
    result['failure_analysis'] = analysis

    # Save detailed report
    report_path = self.failure_analyzer.save_failure_report(
        step_num,
        analysis,
        screenshot_path=result.get('screenshot_after')
    )

    # Log critical failure with details
    self.logger.error("="*80)
    self.logger.error(f"CRITICAL FAILURE - Step {step_num} failed at ALL levels")
    self.logger.error("="*80)
    self.logger.error(f"Step Text: {step_text}")
    self.logger.error(f"Failure Reason: {analysis['failure_reason']}")
    self.logger.error("")

    # L1 Details
    l1 = analysis['l1_details']
    self.logger.error("L1 (Custom Selectors):")
    self.logger.error(f"  Keywords: {l1.get('keywords_extracted', [])}")
    self.logger.error(f"  Reason: {l1.get('reason')}")
    if l1.get('selector'):
        self.logger.error(f"  Tried: {l1['selector']}")
        self.logger.error(f"  Count: {l1.get('count_on_page', 0)}")

    # L2 Details
    l2 = analysis['l2_details']
    self.logger.error("")
    self.logger.error("L2 (Generic Patterns):")
    self.logger.error(f"  Type: {l2.get('action_type')}")
    self.logger.error(f"  Reason: {l2.get('reason')}")

    # L3 Details
    self.logger.error("")
    self.logger.error("L3 (CV-Guided):")
    self.logger.error(f"  Reason: {analysis['l3_details'].get('reason')}")

    # Recommendations
    self.logger.error("")
    self.logger.error("="*80)
    self.logger.error("RECOMMENDATIONS:")
    self.logger.error("="*80)
    for i, rec in enumerate(analysis['recommendations'], 1):
        self.logger.error(f"{i}. {rec}")

    # Selector template
    if analysis.get('missing_selector_template'):
        self.logger.error("")
        self.logger.error("="*80)
        self.logger.error("MISSING SELECTOR TEMPLATE (copy to JSON):")
        self.logger.error("="*80)
        import json
        self.logger.error(json.dumps(analysis['missing_selector_template'], indent=2))

    self.logger.error("")
    self.logger.error(f"Detailed report: {report_path}")
    self.logger.error("="*80)
    self.logger.error("TEST WILL TERMINATE - Fix above issue and restart")
    self.logger.error("="*80)
```

---

### Phase 3: Not Started
- Create Failure_Reports/ directory
- Test the complete flow
- Verify termination works
- Verify reports are generated

---

## Testing Plan

Once Phase 2 is integrated:

1. **Test with missing selector:**
   ```bash
   python run_test.py RBPLCD-8862
   ```
   - Should fail at Step 4 (Select "Project" dropdown)
   - Should detect: MISSING_SELECTOR_IN_JSON
   - Should generate template for data-opencreatedialogdropdown

2. **Verify output:**
   - Check logs for detailed failure analysis
   - Check Failure_Reports/ for saved report
   - Verify JSON template is correct

3. **Add selector and retry:**
   - Add template to JSON
   - Re-run test
   - Verify Step 4 now passes

---

## Files Modified Summary

**Modified:**
1. `utils/step_executor.py` - Added fail-fast logic (lines 158-182)
2. `agents/vision_executor_agent.py` - Added termination logic (lines 163-176)

**Created:**
1. `utils/failure_analyzer.py` - Complete failure analysis system (519 lines)

**Imported:**
1. `utils/step_executor.py` line 19 - Added FailureAnalyzer import

**Still Need:**
- Initialize FailureAnalyzer in StepExecutor.__init__()
- Call analyzer in failure handling code
- Create Failure_Reports/ directory
- Test and verify

---

## Key Features Implemented

✅ Immediate termination when all 3 levels fail
✅ Detailed L1/L2/L3 failure breakdown
✅ Root cause identification (missing, ambiguous, not visible)
✅ Actionable recommendations
✅ Copy-paste ready JSON selector template
✅ Save failure reports (JSON + TXT)

---

## Next Session Tasks

1. Complete Phase 2 integration (15 minutes)
2. Create Failure_Reports/ directory
3. Test with RBPLCD-8862
4. Verify reports generated correctly
5. Document user workflow

**Estimated time to complete:** 30 minutes
