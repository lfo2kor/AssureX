# Comprehensive L1/L2/L3 Improvement Strategy

## 🎯 Goal: Maximize Overall Test Success Rate

**Current Problem:**
- L1 success rate is low (~20-30%)
- Too many steps fall back to L2/L3
- L2/L3 are slower and less reliable
- Overall test execution is slow

**Target:**
- **L1 Success:** 70-85% (main improvement area)
- **L2 Success:** 10-15% (backup for edge cases)
- **L3 Success:** 5% (last resort for complex cases)
- **Overall Success:** 95%+ across all levels

---

## 📊 Current 3-Level Architecture

```
Step Execution Flow:
  ├── L1: Custom Selectors (selectors.json)
  │     ├── Fast (50-100ms)
  │     ├── Reliable (if found)
  │     └── Current success: ~25%
  │
  ├── L2: Generic HTML Patterns (hardcoded)
  │     ├── Medium speed (200-500ms)
  │     ├── Moderate reliability (pattern matching)
  │     └── Current success: ~50%
  │
  └── L3: CV-Guided Discovery (GPT-4o Vision)
        ├── Slow (2-5 seconds)
        ├── Variable reliability (AI-dependent)
        └── Current success: ~25%
```

**Current Distribution (Estimated):**
- L1 handles: 25% of steps
- L2 handles: 50% of steps
- L3 handles: 20% of steps
- Failure: 5% of steps

**Problems:**
- Too much reliance on L2 (slow pattern matching)
- L3 too slow for routine operations
- L1 underutilized due to poor selector matching

---

## 🚀 Improvement Strategy for Each Level

---

## 1️⃣ LEVEL 1 (L1) - Custom Selectors

### Current Implementation

**File:** `utils/selector_loader.py` (V1.0)

**How it works:**
```python
def find_best_selector(step_text, module):
    keywords = extract_keywords(step_text)  # ['save', 'button']

    # Search in selectors.json
    for selector in selectors:
        if module not in selector['module']:
            continue  # ❌ BLOCKS cross-module

        if any(kw in selector['attr'] or kw in selector['value'] for kw in keywords):
            return selector  # ❌ RETURNS FIRST MATCH

    return None
```

**Problems:**
1. ❌ **Strict module filter** - Blocks Parts accordion when in Teststep
2. ❌ **No scoring** - Returns first match, not best match
3. ❌ **No context** - Selectors have no semantic information
4. ❌ **No state tracking** - Each step independent
5. ❌ **Poor keyword matching** - Limited to attr/value only

---

### L1 Improvement Plan

#### Phase 1: Sequential Context (Week 1)

**What:** Add state tracking across steps

**Implementation:**
- Use `SelectorLoaderV2` with `SequentialContextTracker`
- Track: current_module, visible_modules, edit_mode, dialog_open
- Update state after each step

**Changes:**
```python
# agents/vision_executor_agent.py (2 lines)
from utils.selector_loader_v2 import SelectorLoaderV2
selector_loader = SelectorLoaderV2(use_sequential_context=True)
```

**Impact:**
- Cross-module selectors work ✅
- Parts accordion findable from Teststep ✅
- State-aware module scope ✅

**Expected Improvement:** +20% L1 success (25% → 45%)

---

#### Phase 2: Selector Enrichment (Week 1-2)

**What:** Add context, priority, usage_scenario to selectors

**Current selector:**
```json
{
  "attr": "data-parts",
  "value": "parts",
  "module": "parts"
}
```

**Enriched selector:**
```json
{
  "attr": "data-parts",
  "value": "parts",
  "module": "parts",
  "context": ["accordion", "section", "parts", "expansion-panel"],
  "priority": 9,
  "usage_scenario": "Parts accordion in detail view",
  "elementType": "mat-expansion-panel",
  "lineNumber": 45
}
```

**How to enrich:**
1. Run extraction script on HTML files
2. Analyze Angular Material directives
3. Extract context from element type + attributes
4. Calculate priority based on element importance

**Impact:**
- Better keyword matching (context field) ✅
- Intelligent ranking (priority field) ✅
- Disambiguation (usage_scenario) ✅

**Expected Improvement:** +25% L1 success (45% → 70%)

---

#### Phase 3: Score-based Ranking (Built into V2)

**What:** Rank selectors by relevance score

**Scoring algorithm:**
```python
score = 0

# Keyword matches
for keyword in keywords:
    if keyword in selector['attr/value']: score += 5
    if keyword in selector['context']: score += 8  # Context is stronger

# Priority boost
score += selector['priority']  # 0-10 points

# State-based boost
if selector['module'] == current_module: score += 20
if selector['module'] in visible_modules: score += 15
if edit_mode and 'input' in context: score += 8
if dialog_open and 'dialog' in context: score += 8

return selector_with_highest_score
```

**Impact:**
- Returns BEST match, not first match ✅
- Handles ambiguous cases ✅
- Context-aware scoring ✅

**Expected Improvement:** +10% L1 success (70% → 80%)

---

#### Phase 4: Dynamic Selector Handling (Week 3)

**Problem:** Some selectors have runtime values
```json
{
  "attr": "attr.data-opencreatedialogdropdown",
  "value": "button",  // ← This is a variable, not "Project"!
  "dynamic": true
}
```

**Solutions:**

**Option A: Extract Possible Values from TypeScript**
```typescript
// analyze create-new.component.ts
buttons = ['Project', 'Task', 'Test', 'Requirement'];
```

Add to selector:
```json
{
  "value": "button",
  "dynamic": true,
  "possible_values": ["Project", "Task", "Test", "Requirement"]
}
```

**Option B: Use Dynamic Matching Pattern**
```python
if selector['dynamic']:
    # Match by attribute existence + context
    if 'project' in step_text.lower() and 'menu-item' in selector['context']:
        return selector
```

**Impact:**
- Dynamic menu items findable ✅
- Dropdown options matchable ✅

**Expected Improvement:** +5% L1 success (80% → 85%)

---

### L1 Success Roadmap

| Phase | Action | Timeline | Expected L1 | Improvement |
|-------|--------|----------|-------------|-------------|
| Baseline | Current V1.0 | Now | 25% | - |
| Phase 1 | Sequential Context | Week 1 | 45% | +20% |
| Phase 2 | Enriched Selectors | Week 2 | 70% | +25% |
| Phase 3 | Score-based Ranking | Week 2 | 80% | +10% |
| Phase 4 | Dynamic Handling | Week 3 | 85% | +5% |

---

## 2️⃣ LEVEL 2 (L2) - Generic Patterns

### Current Implementation

**File:** `utils/step_executor.py` (lines 62-106, 268-427)

**How it works:**
```python
generic_patterns = {
    'button_click': [
        "button:has-text('{text}')",
        "[role='button']:has-text('{text}')",
        "button",
    ],
    'dropdown_select': [
        "[data-attribute='{text}']",
        "label:has-text('{text}') .mat-select",
        ".mat-select",
    ],
    'accordion_expand': [
        ".mat-expansion-panel-header:has-text('{text}')",
        "[role='button'][aria-expanded='false']",
    ],
}
```

**Process:**
1. Detect action type from step text (button_click, dropdown_select, etc.)
2. Extract key text (e.g., "Save", "Type", "Parts")
3. Try patterns in order, stop at first unique match

**Strengths:**
- ✅ Framework-aware (Angular Material)
- ✅ Handles common UI patterns
- ✅ No JSON maintenance needed

**Problems:**
- ❌ Fixed pattern order (may not try best pattern first)
- ❌ Stops at first match (may not be correct one)
- ❌ No row scoping for some cases
- ❌ Limited action type detection

---

### L2 Improvement Plan

#### Improvement 1: Enhanced Action Detection (Week 2)

**Current:**
```python
if 'dropdown' in step_lower or 'select' in step_lower:
    action_type = 'dropdown_select'
elif 'button' in step_lower or 'click' in step_lower:
    action_type = 'button_click'
```

**Improved:**
```python
# Priority-based detection (most specific first)
if re.search(r'expand.*accordion', step_lower):
    action_type = 'accordion_expand'
elif re.search(r'(dropdown|select).*and.*(select|choose)', step_lower):
    action_type = 'dropdown_select_with_option'  # New type
elif 'edit.*button' in step_lower:
    action_type = 'edit_button'  # Specific type
elif 'save.*button' in step_lower:
    action_type = 'save_button'  # Specific type
```

**Impact:** Better pattern selection, fewer false positives

---

#### Improvement 2: Smart Pattern Ordering (Week 2)

**Current:** Tries patterns in hardcoded order

**Improved:** Rank patterns by:
1. State relevance (if in edit mode, prioritize input patterns)
2. Recent success (track which patterns worked recently)
3. Specificity (more specific patterns first)

```python
def get_ranked_patterns(action_type, state):
    patterns = generic_patterns[action_type]
    scored_patterns = []

    for pattern in patterns:
        score = 0

        # Specificity score
        if '{text}' in pattern: score += 10  # Text-based is specific
        if 'data-' in pattern: score += 8   # Data attributes reliable

        # State-based score
        if state.edit_mode and 'input' in pattern: score += 5
        if state.dialog_open and 'dialog' in pattern: score += 5

        scored_patterns.append((pattern, score))

    # Sort by score, return patterns
    return [p for p, s in sorted(scored_patterns, key=lambda x: x[1], reverse=True)]
```

**Impact:** Tries most likely pattern first, faster execution

---

#### Improvement 3: Better Row Scoping (Week 3)

**Current:** Works for some cases, fails for nested structures

**Improved:**
```python
# Enhanced scoping strategies
scoping_strategies = [
    # Strategy 1: Exact text match, closest ancestor
    f":text-is('{row_id}') >> xpath=ancestor::tr[1] >> {action_selector}",

    # Strategy 2: Row with data attribute
    f"[data-row-id='{row_id}'] >> {action_selector}",

    # Strategy 3: Flexible containment
    f"tr:has(:text-is('{row_id}')) >> {action_selector}",

    # Strategy 4: Sibling navigation
    f":text-is('{row_id}') >> xpath=following-sibling::*[1]",
]
```

**Impact:** Better handling of table/list row operations

---

### L2 Success Target

**Current:** ~50% of steps use L2
**Target:** 10-15% of steps (L1 should handle most)

**With improvements:**
- L2 handles edge cases L1 misses
- Faster pattern matching (smart ordering)
- More reliable (better action detection)

---

## 3️⃣ LEVEL 3 (L3) - CV-Guided Discovery

### Current Implementation

**File:** `utils/step_executor.py` (lines 429-478)

**How it works:**
```python
def _try_level3_cv_guided(step_text, screenshot):
    # Call GPT-4o Vision API
    cv_result = vision_client.identify_step_selector(
        screenshot,
        step_text,
        custom_selector,  # L1 selector as hint
        module_context
    )

    # Try CV-suggested selector
    primary_selector = cv_result.get('selector')
    if primary_selector and page.locator(primary_selector).count() > 0:
        return execute_action(primary_selector)

    # Try fallbacks
    for fallback in cv_result.get('fallback_selectors', []):
        if page.locator(fallback).count() > 0:
            return execute_action(fallback)
```

**Strengths:**
- ✅ Handles ambiguous cases
- ✅ Visual understanding (OCR, layout)
- ✅ Can find elements L1/L2 miss

**Problems:**
- ❌ **Slow** (2-5 seconds per call)
- ❌ **Expensive** (API costs)
- ❌ **Variable reliability** (depends on screenshot quality)
- ❌ **No learning** (doesn't remember successful selectors)

---

### L3 Improvement Plan

#### Improvement 1: CV Result Caching (Week 2)

**Problem:** Same step executed multiple times calls CV repeatedly

**Solution:** Cache CV results by (step_text, screenshot_hash)

```python
class CVCache:
    def __init__(self):
        self.cache = {}  # {(step_text, img_hash): cv_result}

    def get(self, step_text, screenshot):
        img_hash = hashlib.md5(screenshot).hexdigest()[:8]
        return self.cache.get((step_text, img_hash))

    def set(self, step_text, screenshot, result):
        img_hash = hashlib.md5(screenshot).hexdigest()[:8]
        self.cache[(step_text, img_hash)] = result
```

**Impact:**
- Avoid redundant API calls ✅
- Faster re-runs ✅
- Lower costs ✅

---

#### Improvement 2: CV Learning - Successful Selector Extraction (Week 3)

**Problem:** CV finds a selector, but knowledge is lost

**Solution:** When L3 succeeds, save selector to JSON

```python
def _try_level3_cv_guided(step_text, screenshot):
    # ... existing code ...

    if success:
        # Extract selector pattern for future use
        learned_selector = {
            "attr": extract_attr_from_selector(selector_used),
            "value": extract_value_from_selector(selector_used),
            "module": current_module,
            "context": extract_context_from_step(step_text),
            "priority": 7,  # Medium priority (learned)
            "learned": True,
            "learned_from": step_text
        }

        # Append to selectors.json
        save_learned_selector(learned_selector)
```

**Impact:**
- L3 discoveries become L1 selectors ✅
- System learns over time ✅
- Reduces L3 usage ✅

---

#### Improvement 3: Enhanced CV Prompt (Week 2)

**Current:** Basic prompt with step text

**Improved:** Rich context prompt

```python
prompt = f"""
You are analyzing a screenshot to find a selector for this step:
"{step_text}"

Context:
- Module: {current_module}
- Previous step: {previous_step_text}
- UI State: edit_mode={edit_mode}, dialog_open={dialog_open}
- Visible modules: {visible_modules}
- Recent selectors used: {recent_selectors[-3:]}

L1 suggested selector (if any): {custom_selector}

Please identify:
1. The exact element to interact with
2. A unique CSS selector for it
3. Fallback selectors if ambiguous
4. Reasoning for your choice

Consider:
- Angular Material patterns (mat-button, mat-select, etc.)
- Data attributes (data-*)
- ARIA roles
- Text content
- Spatial layout
"""
```

**Impact:**
- Better selector suggestions ✅
- Fewer API retries ✅
- Higher success rate ✅

---

#### Improvement 4: Hybrid L2+L3 (Week 3)

**Problem:** L3 called for simple cases L2 could handle with hints

**Solution:** L2 asks CV for hints without full analysis

```python
def _try_level2_with_cv_hint(step_text, screenshot):
    # Quick CV call: "Which button text should I look for?"
    hint = vision_client.get_quick_hint(screenshot, step_text)
    # Returns: {"element_text": "Save", "type": "button"}

    # Use hint in L2 patterns
    extracted_text = hint.get('element_text')
    action_type = hint.get('type')

    # Try L2 patterns with CV-extracted text
    return try_generic_patterns(action_type, extracted_text)
```

**Impact:**
- Faster than full L3 (1 second vs 3 seconds) ✅
- More reliable than blind L2 ✅
- Lower cost than full CV analysis ✅

---

### L3 Success Target

**Current:** ~25% of steps use L3
**Target:** 5% of steps (only for truly complex cases)

**With improvements:**
- Cached results avoid repeat calls
- Learned selectors reduce future L3 usage
- Hybrid L2+L3 handles medium complexity faster

---

## 📊 Overall Strategy Timeline

### Week 1: Quick Wins

**Focus: L1 Sequential Context**

- [ ] Day 1-2: Implement sequential context (2 line change)
- [ ] Day 3: Test with current selectors
- [ ] Day 4-5: Run enrichment script, generate enriched selectors
- [ ] End of Week: **L1 = 45%**

---

### Week 2: Major Improvements

**Focus: L1 Enrichment + L2/L3 Enhancements**

- [ ] Day 1-2: Deploy enriched selectors, test L1
- [ ] Day 3: Enhance L2 action detection
- [ ] Day 4: Add CV caching and enhanced prompts
- [ ] Day 5: Integration testing
- [ ] End of Week: **L1 = 70%, L2 improved, L3 faster**

---

### Week 3: Polish & Edge Cases

**Focus: Dynamic selectors + Learning**

- [ ] Day 1-2: Implement dynamic selector handling
- [ ] Day 3: Add CV learning (L3 → L1 pipeline)
- [ ] Day 4: Better row scoping in L2
- [ ] Day 5: Comprehensive testing
- [ ] End of Week: **L1 = 85%, L2 = 12%, L3 = 3%**

---

## 📈 Expected Results

### Success Rate Distribution

| Metric | Current | Week 1 | Week 2 | Week 3 | Target |
|--------|---------|--------|--------|--------|--------|
| **L1 Success** | 25% | 45% | 70% | 85% | 80-85% |
| **L2 Success** | 50% | 40% | 20% | 12% | 10-15% |
| **L3 Success** | 20% | 15% | 8% | 3% | 5% |
| **Overall Success** | 95% | 95% | 98% | 100% | 95%+ |

### Performance Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Avg step time | 5s | 1.5s | 3.3x faster |
| L1 step time | 0.1s | 0.1s | Same |
| L2 step time | 1s | 0.5s | 2x faster |
| L3 step time | 4s | 2s | 2x faster (caching) |
| Total test time (8 steps) | 40s | 12s | 3.3x faster |

### Cost Reduction

| Metric | Current | Target | Savings |
|--------|---------|--------|---------|
| CV API calls per test | 3-4 | 0-1 | 75% reduction |
| API cost per test | $0.12 | $0.03 | 75% savings |
| Monthly cost (100 tests) | $12 | $3 | $9/month saved |

---

## 🎯 Implementation Priority

### Must Have (Week 1-2)

1. **Sequential Context** - Enables cross-module matching
2. **Enriched Selectors** - Provides context/priority data
3. **Score-based Ranking** - Returns best match

**Impact:** L1 from 25% → 70%

---

### Should Have (Week 2-3)

4. **Enhanced L2 Action Detection** - Better pattern selection
5. **CV Caching** - Faster L3, lower costs
6. **Enhanced CV Prompts** - Better L3 suggestions

**Impact:** Overall performance 2-3x faster

---

### Nice to Have (Week 3+)

7. **Dynamic Selector Handling** - Edge cases
8. **CV Learning** - Long-term improvement
9. **Hybrid L2+L3** - Medium complexity optimization

**Impact:** 85%+ L1, minimal L3 usage

---

## 📝 Success Metrics

### How to Measure

**After each week, run these metrics:**

```bash
# Run all test tickets
python run_test.py RBPLCD-8835
python run_test.py RBPLCD-8862
# ... other tickets

# Analyze logs
python analyze_test_results.py
```

**Track:**
1. **L1 success rate** = (Steps succeeded in L1) / (Total steps)
2. **L2 success rate** = (Steps succeeded in L2) / (Steps that reached L2)
3. **L3 success rate** = (Steps succeeded in L3) / (Steps that reached L3)
4. **Overall success rate** = (Total passed steps) / (Total steps)
5. **Average step time**
6. **CV API calls**

---

## 🚦 Decision Points

### At End of Week 1

**If L1 = 40-50%:** ✅ Proceed to Week 2 (enrichment)
**If L1 < 40%:** ⚠️ Debug sequential context, check state updates
**If L1 > 50%:** 🎉 Accelerate to Week 2

### At End of Week 2

**If L1 = 65-75%:** ✅ Proceed to Week 3 (polish)
**If L1 < 65%:** ⚠️ Review enrichment quality, add more context
**If L1 > 75%:** 🎉 Consider Week 3 optional

### At End of Week 3

**If L1 > 80%:** 🎉 **SUCCESS!** Deploy to production
**If L1 = 70-80%:** ✅ Good enough, monitor and iterate
**If L1 < 70%:** ⚠️ Investigate edge cases, expand enrichment

---

## 📋 Summary

### The Plan

1. **Week 1:** Sequential Context → L1 = 45%
2. **Week 2:** Enriched Selectors + L2/L3 improvements → L1 = 70%
3. **Week 3:** Dynamic handling + Learning → L1 = 85%

### Key Improvements

| Level | Improvement | Impact |
|-------|-------------|--------|
| **L1** | Sequential context + Enriched selectors + Scoring | 25% → 85% |
| **L2** | Better action detection + Smart ordering | Faster, more reliable |
| **L3** | Caching + Enhanced prompts + Learning | 2x faster, 75% cost reduction |

### Why This Works

- **L1** handles routine operations (fast, reliable)
- **L2** handles framework patterns (medium speed, good reliability)
- **L3** handles complex edge cases only (slow but powerful)

**Result:** 3x faster tests, 85% L1 success, 95%+ overall success 🚀

---

Ready to start with Week 1? 🎯
