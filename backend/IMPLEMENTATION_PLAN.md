# Implementation Plan: State Tracking + Context-Based Matching

## Overview

This document outlines the complete implementation plan for the State Tracking + Context-Based Matching solution.

---

## **File Structure**

### **New Files to Create:**

```
TA_AI_Project/
├── utils/
│   ├── execution_state_tracker.py          # NEW - State tracking logic
│   ├── context_scorer.py                   # NEW - Context-based scoring
│   └── selector_matcher.py                 # NEW - Unified matching interface
│
├── Selectors_Folder/
│   └── selectors_enriched_all_modules.json # ALREADY EXISTS (400 selectors)
│
└── tests/
    └── test_state_tracker.py               # NEW - Unit tests
```

### **Existing Files to Modify:**

```
TA_AI_Project/
├── utils/
│   ├── step_executor.py                    # MODIFY - Integrate new matching
│   └── selector_loader.py                  # MODIFY - Add context-aware methods
│
├── run_test.py                              # MODIFY - Initialize state tracker
│
└── plcdtest_config.yaml                     # MODIFY - Add new config options
```

---

## **Implementation Phases**

### **Phase 1: Core State Tracking (Priority 1)**
**Files:** `execution_state_tracker.py`
**Time:** 3-4 hours
**Dependencies:** None

### **Phase 2: Context-Based Scoring (Priority 1)**
**Files:** `context_scorer.py`
**Time:** 2-3 hours
**Dependencies:** None

### **Phase 3: Unified Matcher Interface (Priority 2)**
**Files:** `selector_matcher.py`
**Time:** 1-2 hours
**Dependencies:** Phase 1, Phase 2

### **Phase 4: Integration (Priority 3)**
**Files:** Modify `step_executor.py`, `selector_loader.py`, `run_test.py`
**Time:** 2-3 hours
**Dependencies:** Phase 1, 2, 3

### **Phase 5: Testing & Validation (Priority 4)**
**Files:** `test_state_tracker.py`, run actual tests
**Time:** 2-3 hours
**Dependencies:** All previous phases

**Total Time:** 10-15 hours

---

## **Detailed File Specifications**

---

### **NEW FILE 1: `utils/execution_state_tracker.py`**

**Purpose:** Track execution state across test steps

**Size:** ~300 lines

**Key Classes:**
```python
class ExecutionState:
    """Immutable state snapshot"""
    def __init__(self):
        self.current_page: str
        self.module: str
        self.visible_components: List[str]
        self.previous_actions: List[str]
        self.active_accordion: Optional[str]
        self.open_dialogs: List[str]
        self.open_menus: List[str]
        self.current_context: str
        self.editing_entity: Optional[str]
        self.current_list_view: Optional[str]

class ExecutionStateTracker:
    """Main state tracker"""
    def __init__(self):
        self.state: ExecutionState
        self.state_history: List[ExecutionState]

    def update_after_action(self, action_type: str, selector: dict, result: bool):
        """Update state based on action executed"""
        pass

    def get_relevant_selectors(self, all_selectors: List[dict]) -> List[dict]:
        """Filter selectors by current state"""
        pass

    def infer_action_type(self, step_text: str, selector: dict) -> str:
        """Infer what type of action was performed"""
        pass
```

**Methods:**
- `update_after_action()` - Update state after each step
- `get_relevant_selectors()` - Filter selectors by visibility
- `infer_action_type()` - Detect action from step text
- `reset()` - Reset state (for new test)
- `get_state_snapshot()` - Get current state for debugging

**Integration Points:**
- Called by `step_executor.py` after each step
- Uses selector metadata to infer state changes

---

### **NEW FILE 2: `utils/context_scorer.py`**

**Purpose:** Score selectors based on context and keywords

**Size:** ~150 lines

**Key Classes:**
```python
class ContextScorer:
    """Score selectors using context-based matching"""

    def __init__(self, config: dict = None):
        self.keyword_weight = 10
        self.context_weight = 15
        self.priority_weight = 1
        self.module_bonus = 25
        self.parent_component_bonus = 15

    def score_selector(
        self,
        selector: dict,
        keywords: List[str],
        test_module: Optional[str] = None,
        state: Optional[ExecutionState] = None
    ) -> int:
        """Calculate score for a single selector"""
        pass

    def score_selectors(
        self,
        selectors: List[dict],
        keywords: List[str],
        test_module: Optional[str] = None,
        state: Optional[ExecutionState] = None
    ) -> List[Tuple[int, dict]]:
        """Score all selectors and return sorted by score"""
        pass

    def extract_keywords(self, step_text: str) -> List[str]:
        """Extract keywords from step text"""
        pass

    def decompose_attribute_name(self, attr: str) -> List[str]:
        """Break 'data-ShowMoreVerticalBtn' into ['show', 'more', 'vertical', 'btn']"""
        pass
```

**Methods:**
- `score_selector()` - Score single selector
- `score_selectors()` - Score all and sort
- `extract_keywords()` - Parse step text
- `decompose_attribute_name()` - CamelCase → keywords

**Integration Points:**
- Called by `selector_matcher.py`
- Uses enriched selectors with context field

---

### **NEW FILE 3: `utils/selector_matcher.py`**

**Purpose:** Unified interface for L1/L2/L3 selector matching

**Size:** ~200 lines

**Key Classes:**
```python
class SelectorMatcher:
    """Unified selector matching with state tracking"""

    def __init__(
        self,
        selector_loader: SelectorLoader,
        state_tracker: ExecutionStateTracker,
        scorer: ContextScorer,
        config: dict
    ):
        self.selector_loader = selector_loader
        self.state_tracker = state_tracker
        self.scorer = scorer
        self.config = config

    def find_best_selector(
        self,
        step_text: str,
        module: Optional[str] = None
    ) -> Tuple[bool, Optional[dict], str]:
        """
        Find best matching selector using state + context

        Returns:
            (success, selector, level)
            - success: True if found
            - selector: Best matching selector dict
            - level: "L1", "L2", or "L3"
        """
        # Try L1: State-filtered + context-scored
        success, selector = self._try_level1_with_state(step_text, module)
        if success:
            return (True, selector, "L1")

        # Try L2: Generic patterns (fallback)
        success, selector = self._try_level2_patterns(step_text)
        if success:
            return (True, selector, "L2")

        # Try L3: Vision (last resort)
        success, selector = self._try_level3_vision(step_text)
        if success:
            return (True, selector, "L3")

        return (False, None, "FAILED")

    def _try_level1_with_state(self, step_text: str, module: str):
        """Level 1: State tracking + context scoring"""
        # 1. Get state-filtered candidates
        all_selectors = self.selector_loader.load_all_selectors()
        candidates = self.state_tracker.get_relevant_selectors(all_selectors)

        # 2. Extract keywords
        keywords = self.scorer.extract_keywords(step_text)

        # 3. Score candidates
        scored = self.scorer.score_selectors(
            candidates,
            keywords,
            module,
            self.state_tracker.state
        )

        # 4. Check if clear winner
        if scored and scored[0][0] > self.config['min_score_threshold']:
            return (True, scored[0][1])

        return (False, None)

    def _try_level2_patterns(self, step_text: str):
        """Level 2: Generic hardcoded patterns"""
        # Delegate to existing L2 logic in step_executor
        pass

    def _try_level3_vision(self, step_text: str):
        """Level 3: AI Vision"""
        # Delegate to existing L3 logic in step_executor
        pass
```

**Methods:**
- `find_best_selector()` - Main entry point (L1→L2→L3)
- `_try_level1_with_state()` - State + context matching
- `_try_level2_patterns()` - Generic patterns
- `_try_level3_vision()` - Vision fallback

**Integration Points:**
- Called by `step_executor.py` instead of direct selector lookup
- Uses all three components (state, scorer, loader)

---

### **MODIFIED FILE 1: `utils/step_executor.py`**

**Current:** 674 lines
**Changes:** Modify ~50 lines, add ~20 lines

**Modifications:**

```python
# Line 1: Add imports
from utils.execution_state_tracker import ExecutionStateTracker
from utils.context_scorer import ContextScorer
from utils.selector_matcher import SelectorMatcher

class StepExecutor:
    def __init__(self, page, config):
        self.page = page
        self.config = config

        # Existing
        self.selector_loader = SelectorLoader(config['selectors_file'])

        # NEW: Initialize state tracking components
        self.state_tracker = ExecutionStateTracker()
        self.scorer = ContextScorer(config)
        self.matcher = SelectorMatcher(
            self.selector_loader,
            self.state_tracker,
            self.scorer,
            config
        )

    def execute_step(self, step_text, module=None):
        """Execute a single test step"""

        # NEW: Use unified matcher instead of direct lookup
        success, selector, level = self.matcher.find_best_selector(step_text, module)

        if not success:
            return {
                'success': False,
                'level': 'FAILED',
                'error': 'No selector found'
            }

        # Execute action
        result = self._perform_action(step_text, selector)

        # NEW: Update state after action
        action_type = self.state_tracker.infer_action_type(step_text, selector)
        self.state_tracker.update_after_action(action_type, selector, result['success'])

        return {
            'success': result['success'],
            'level': level,
            'selector': selector,
            'state': self.state_tracker.get_state_snapshot()  # For debugging
        }

    # REMOVE or DEPRECATE: Old _execute_three_level_strategy
    # (Replaced by matcher.find_best_selector)
```

**Changes Summary:**
- ✅ Add state tracker initialization
- ✅ Replace direct selector lookup with matcher
- ✅ Update state after each action
- ✅ Add state snapshot to results (debugging)
- ⚠️ Keep L2/L3 logic for fallback (delegated to matcher)

---

### **MODIFIED FILE 2: `utils/selector_loader.py`**

**Current:** 297 lines
**Changes:** Add ~50 lines (new methods)

**Additions:**

```python
class SelectorLoader:
    # Existing methods stay unchanged

    # NEW METHODS:

    def load_enriched_selectors(self, filepath: str = None) -> List[dict]:
        """Load selectors with context field"""
        if filepath is None:
            # Use enriched selectors by default
            filepath = "Selectors_Folder/selectors_enriched_all_modules.json"

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def filter_by_visible_components(
        self,
        selectors: List[dict],
        visible_components: List[str]
    ) -> List[dict]:
        """Filter selectors to only those in visible components"""
        filtered = []
        for selector in selectors:
            if selector.get('parentComponent') in visible_components:
                filtered.append(selector)
        return filtered

    def filter_by_context(
        self,
        selectors: List[dict],
        required_context: List[str]
    ) -> List[dict]:
        """Filter selectors that have all required context keywords"""
        filtered = []
        for selector in selectors:
            selector_context = set(selector.get('context', []))
            if all(ctx in selector_context for ctx in required_context):
                filtered.append(selector)
        return filtered
```

**Changes Summary:**
- ✅ Add method to load enriched selectors
- ✅ Add filtering by visible components
- ✅ Add filtering by context keywords
- ✅ Keep existing methods for backward compatibility

---

### **MODIFIED FILE 3: `run_test.py`**

**Current:** ~100 lines
**Changes:** Modify ~10 lines

**Modifications:**

```python
# Line ~15: Add import
from utils.execution_state_tracker import ExecutionStateTracker

def main():
    # ... existing setup code ...

    # Load config
    state = load_config_node({'ticket_number': ticket_id})

    # Parse JIRA ticket
    state = jira_parser_agent(state)

    # NEW: Initialize state tracker before execution
    state_tracker = ExecutionStateTracker()
    state['state_tracker'] = state_tracker

    # Execute test with state tracking
    state = vision_executor_agent(state)

    # Generate report
    state = report_generator_agent(state)

    # NEW: Save state history for debugging
    if state_tracker.state_history:
        state['execution_state_history'] = [
            s.to_dict() for s in state_tracker.state_history
        ]
```

**Changes Summary:**
- ✅ Initialize state tracker
- ✅ Pass to vision executor
- ✅ Save state history in results

---

### **MODIFIED FILE 4: `plcdtest_config.yaml`**

**Changes:** Add new configuration section

**Additions:**

```yaml
# Existing config stays the same

# NEW: State Tracking Configuration
state_tracking:
  enabled: true
  track_history: true
  max_history_size: 100

# NEW: Context Scoring Configuration
context_scoring:
  enabled: true
  keyword_weight: 10
  context_weight: 15
  priority_weight: 1
  module_bonus: 25
  parent_component_bonus: 15
  min_score_threshold: 30  # Minimum score to accept L1 match

# NEW: Selector Configuration
selectors:
  use_enriched: true  # Use enriched selectors with context
  enriched_file: "Selectors_Folder/selectors_enriched_all_modules.json"
  fallback_file: "Selectors_Folder/selectors.json"  # Fallback if enriched not found
```

---

## **Implementation Order**

### **Step 1: Create Core State Tracker**

**File:** `utils/execution_state_tracker.py`

**Tasks:**
1. Define `ExecutionState` dataclass
2. Implement `ExecutionStateTracker` class
3. Implement state update rules:
   - Login → dashboard
   - Navigate → list view
   - Click accordion → expanded accordion
   - Click edit → edit form
   - etc.
4. Implement `get_relevant_selectors()` filtering

**Test:** Unit test with mock selectors

---

### **Step 2: Create Context Scorer**

**File:** `utils/context_scorer.py`

**Tasks:**
1. Implement keyword extraction
2. Implement attribute name decomposition
3. Implement scoring algorithm:
   - Keyword matches in attr/value
   - Context matches
   - Priority weighting
   - Module bonus
4. Implement `score_selectors()` batch scoring

**Test:** Unit test with sample selectors

---

### **Step 3: Create Unified Matcher**

**File:** `utils/selector_matcher.py`

**Tasks:**
1. Implement L1 with state + context
2. Integrate existing L2 patterns
3. Integrate existing L3 vision
4. Implement scoring threshold logic

**Test:** Integration test with state tracker + scorer

---

### **Step 4: Integrate into step_executor.py**

**File:** `utils/step_executor.py`

**Tasks:**
1. Initialize components in `__init__`
2. Replace `_execute_three_level_strategy` with `matcher.find_best_selector`
3. Add state update after each step
4. Add state snapshot to results

**Test:** Run RBPLCD-8835 and RBPLCD-8862

---

### **Step 5: Update Configuration**

**Files:** `plcdtest_config.yaml`, `run_test.py`

**Tasks:**
1. Add new config options
2. Initialize state tracker in run_test.py
3. Pass state tracker to executor

**Test:** End-to-end test

---

### **Step 6: Scale Enriched Selectors**

**Already Done!**
- Script: `extract_all_modules.py` (exists)
- Output: `selectors_enriched_all_modules.json` (400 selectors exist)

**TODO:**
- Run on remaining modules (if needed)
- Verify all 888 selectors have context

---

## **Testing Strategy**

### **Unit Tests:**

```python
# tests/test_state_tracker.py

def test_state_after_login():
    tracker = ExecutionStateTracker()
    tracker.update_after_action('login', {}, True)
    assert tracker.state.current_page == 'dashboard'

def test_state_after_navigate():
    tracker = ExecutionStateTracker()
    tracker.update_after_action('login', {}, True)
    tracker.update_after_action('navigate', {'module': 'teststep'}, True)
    assert tracker.state.current_page == 'teststep-list'
    assert tracker.state.module == 'teststep'

def test_filter_by_visible_components():
    tracker = ExecutionStateTracker()
    tracker.state.visible_components = ['parts-list', 'command-bar']

    selectors = [
        {'parentComponent': 'parts-list', 'attr': 'data-edit'},
        {'parentComponent': 'login-form', 'attr': 'data-login'},
        {'parentComponent': 'command-bar', 'attr': 'data-save'}
    ]

    filtered = tracker.get_relevant_selectors(selectors)
    assert len(filtered) == 2  # Only parts-list and command-bar

def test_scorer():
    scorer = ContextScorer()
    selector = {
        'attr': 'data-ShowMoreVerticalBtn',
        'value': 'ShowMoreVerticalBtn',
        'context': ['button', 'menu-trigger', 'more-options', 'more-vertical'],
        'priority': 10
    }

    keywords = ['more', 'vertical', 'button']
    score = scorer.score_selector(selector, keywords)

    assert score > 80  # High score due to multiple matches
```

### **Integration Tests:**

```python
# Test with RBPLCD-8835
def test_rbplcd_8835():
    result = run_test('RBPLCD-8835')
    assert result['overall_status'] == 'PASSED'
    assert result['l1_success_rate'] >= 0.85  # At least 85% L1 success

# Test with RBPLCD-8862
def test_rbplcd_8862():
    result = run_test('RBPLCD-8862')
    assert result['overall_status'] == 'PASSED'
    assert result['l1_success_rate'] == 1.0  # 100% L1 success

    # Verify Step 3 succeeds at L1
    step3 = result['steps'][2]  # Step 3 (0-indexed)
    assert step3['level'] == 'L1'
    assert 'ShowMoreVerticalBtn' in step3['selector']['attr']
```

---

## **Rollout Plan**

### **Phase 1: Development (Week 1)**
- Day 1-2: Implement state tracker + scorer
- Day 3: Implement unified matcher
- Day 4: Integration
- Day 5: Testing

### **Phase 2: Validation (Week 2)**
- Test with RBPLCD-8835 ✅
- Test with RBPLCD-8862 ✅
- Test with 5-10 other JIRA tickets
- Fix any issues

### **Phase 3: Production (Week 3)**
- Scale enriched selectors to all modules
- Run full regression suite
- Monitor L1 success rate
- Deploy to production

---

## **Backward Compatibility**

### **Ensure Old Tests Still Work:**

```python
# In selector_loader.py
def load_all_selectors(self):
    """Load selectors (tries enriched first, falls back to old)"""
    try:
        # Try enriched selectors
        if self.config.get('selectors', {}).get('use_enriched', True):
            return self.load_enriched_selectors()
    except FileNotFoundError:
        pass

    # Fallback to old selectors.json
    return self._load_old_format()
```

### **Config Flag to Enable/Disable:**

```yaml
state_tracking:
  enabled: true  # Set to false to use old approach
```

---

## **Success Metrics**

### **Target Metrics:**

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| L1 Success Rate | 0-12% | 85-95% | % steps succeeding at L1 |
| L2 Fallback | 75-88% | 5-15% | % steps needing L2 |
| L3 Fallback | 12-25% | 5-10% | % steps needing L3 |
| Avg Execution Time | 60s | 45s | Seconds per test |
| L3 Vision Costs | $0.01/test | $0.002/test | Dollar cost |

### **Critical Test Cases:**

1. ✅ RBPLCD-8835 Step 6 (Type dropdown) → L1 SUCCESS
2. ✅ RBPLCD-8862 Step 3 ("... +" button) → L1 SUCCESS
3. ✅ Both tests overall → PASSED

---

## **Risk Mitigation**

### **Risk 1: Enriched Selectors Incomplete**

**Mitigation:**
- Keep fallback to old selectors.json
- Gradual rollout (enable enriched per module)

### **Risk 2: State Inference Errors**

**Mitigation:**
- Extensive logging of state transitions
- Manual review of state history
- Fallback to L2 if state seems wrong

### **Risk 3: Performance Degradation**

**Mitigation:**
- Profile scoring algorithm
- Cache scored results
- Limit candidates before scoring

---

## **Documentation**

### **Files to Create:**

1. `docs/STATE_TRACKING_GUIDE.md` - How state tracking works
2. `docs/CONTEXT_SCORING_GUIDE.md` - How scoring works
3. `docs/MIGRATION_GUIDE.md` - How to migrate tests
4. `docs/TROUBLESHOOTING.md` - Common issues

---

## **Summary**

### **New Files (3):**
1. `utils/execution_state_tracker.py` (300 lines)
2. `utils/context_scorer.py` (150 lines)
3. `utils/selector_matcher.py` (200 lines)

### **Modified Files (4):**
1. `utils/step_executor.py` (~70 line changes)
2. `utils/selector_loader.py` (+50 lines)
3. `run_test.py` (~10 line changes)
4. `plcdtest_config.yaml` (+20 lines)

### **Total New Code:** ~650 lines
### **Total Modified Code:** ~130 lines
### **Total Implementation Time:** 10-15 hours

### **Expected Results:**
- ✅ 85-95% L1 success rate
- ✅ 25% faster execution
- ✅ 80% cost reduction (fewer L3 calls)
- ✅ RBPLCD-8862 Step 3 solved!

**Ready to start implementation?**
