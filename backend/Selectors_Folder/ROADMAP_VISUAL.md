# 3-Level Success Improvement Roadmap

## Current State vs Target State

```
CURRENT (Slow, unreliable)
==========================
Test Step → L1 (25% success) → L2 (50% success) → L3 (20% success) → FAIL (5%)
            ↓ 0.1s              ↓ 1s                ↓ 4s

Average: 5 seconds per step
Most steps fall through to L2/L3 (slow!)


TARGET (Fast, reliable)
========================
Test Step → L1 (85% success) ✅ DONE! → L2 (12% success) → L3 (3% success) → FAIL (0%)
            ↓ 0.1s                      ↓ 0.5s             ↓ 2s

Average: 1.5 seconds per step
Most steps succeed in L1 (fast!)
```

---

## 3-Week Transformation

```
WEEK 1: Sequential Context (Quick Win)
========================================

What We Do:
  - Add state tracking across steps
  - 2 line code change in vision_executor_agent.py

Result:
  L1: 25% → 45% (+20%)

Why It Works:
  - Cross-module selectors now accessible
  - "Parts accordion" findable from Teststep context


WEEK 2: Enriched Selectors + L2/L3 Boost
==========================================

What We Do:
  L1: Run enrichment script, add context/priority to selectors
  L2: Better action detection, smart pattern ordering
  L3: Add caching, enhance CV prompts

Result:
  L1: 45% → 70% (+25%)
  L2: Faster by 2x
  L3: Faster by 2x (caching)

Why It Works:
  - Rich context enables intelligent matching
  - L2 tries best pattern first
  - L3 doesn't repeat work (cached)


WEEK 3: Polish & Edge Cases
=============================

What We Do:
  L1: Handle dynamic selectors, TypeScript analysis
  L2: Better row scoping
  L3: Learning (save successful selectors)

Result:
  L1: 70% → 85% (+15%)
  L2: 20% → 12% (L1 takes over)
  L3: 8% → 3% (mostly eliminated)

Why It Works:
  - Dynamic values matched
  - System learns over time
  - Edge cases handled
```

---

## Level-by-Level Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1: Custom Selectors (selectors.json)                 │
│ ─────────────────────────────────────────────────────────── │
│ Speed:  ⚡⚡⚡ 0.1 seconds                                    │
│ Cost:   💰 Free                                             │
│                                                             │
│ IMPROVEMENTS:                                               │
│ Week 1: Sequential Context        → 25% to 45%             │
│ Week 2: Enriched Selectors        → 45% to 70%             │
│ Week 3: Dynamic Handling          → 70% to 85%             │
│                                                             │
│ HOW:                                                        │
│ ✅ Track state (current module, edit mode, etc.)           │
│ ✅ Add context to selectors (keywords, priority)           │
│ ✅ Score-based ranking (best match, not first)             │
│ ✅ Handle dynamic values (TypeScript analysis)             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2: Generic HTML Patterns (hardcoded)                 │
│ ─────────────────────────────────────────────────────────── │
│ Speed:  ⚡⚡ 0.5-1 seconds                                   │
│ Cost:   💰 Free                                             │
│                                                             │
│ IMPROVEMENTS:                                               │
│ Week 2: Better action detection   → More accurate          │
│ Week 2: Smart pattern ordering    → 2x faster              │
│ Week 3: Enhanced row scoping      → Fewer ambiguities      │
│                                                             │
│ HOW:                                                        │
│ ✅ Detect action type more accurately                      │
│ ✅ Rank patterns by relevance                              │
│ ✅ Try most likely pattern first                           │
│ ✅ Better scoping for table rows                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LEVEL 3: CV-Guided Discovery (GPT-4o Vision)               │
│ ─────────────────────────────────────────────────────────── │
│ Speed:  ⚡ 2-4 seconds                                      │
│ Cost:   💰💰💰 $0.03 per call                               │
│                                                             │
│ IMPROVEMENTS:                                               │
│ Week 2: Result caching            → 50% fewer calls         │
│ Week 2: Enhanced prompts          → Better suggestions      │
│ Week 3: Learning (L3 → L1)        → Long-term reduction    │
│                                                             │
│ HOW:                                                        │
│ ✅ Cache results by screenshot hash                        │
│ ✅ Provide richer context to CV                            │
│ ✅ Save successful selectors to JSON                       │
│ ✅ Hybrid L2+CV for medium complexity                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Metrics Evolution

```
CURRENT STATE
=============
Steps per level:
  [█████] L1: 25%   (2 of 8 steps)  → 0.2s total
  [██████████] L2: 50%   (4 of 8 steps)  → 4s total
  [████] L3: 20%   (2 of 8 steps)  → 8s total
  [█] FAIL: 5%   (0.4 of 8 steps)

Total time: 12.2 seconds for 8 steps


AFTER WEEK 1
============
Steps per level:
  [█████████] L1: 45%   (3.6 of 8 steps)  → 0.36s
  [████████] L2: 40%   (3.2 of 8 steps)  → 3.2s
  [███] L3: 15%   (1.2 of 8 steps)  → 4.8s

Total time: 8.4 seconds (-31%)


AFTER WEEK 2
============
Steps per level:
  [██████████████] L1: 70%   (5.6 of 8 steps)  → 0.56s
  [████] L2: 20%   (1.6 of 8 steps)  → 0.8s
  [██] L3: 8%   (0.6 of 8 steps)  → 1.2s
  [█] FAIL: 2%

Total time: 2.6 seconds (-79% from baseline!)


AFTER WEEK 3 (TARGET)
=====================
Steps per level:
  [█████████████████] L1: 85%   (6.8 of 8 steps)  → 0.68s
  [██] L2: 12%   (1.0 of 8 steps)  → 0.5s
  [█] L3: 3%   (0.2 of 8 steps)  → 0.4s

Total time: 1.6 seconds (-87% from baseline!)
```

---

## Implementation Checklist

### WEEK 1: Sequential Context

```
Day 1-2: Code Changes
  [ ] Edit agents/vision_executor_agent.py (2 lines)
      Line 22: from utils.selector_loader_v2 import SelectorLoaderV2
      Line 63: selector_loader = SelectorLoaderV2(use_sequential_context=True)
  [ ] Test with RBPLCD-8835
  [ ] Verify state tracking in logs

Day 3-5: Enrichment Prep
  [ ] Check if enrichment script exists
  [ ] Run enrichment on 3-5 modules (test)
  [ ] Validate enriched selectors (context, priority present)
  [ ] Full enrichment (all 29 modules)

Day 5: Validation
  [ ] Test RBPLCD-8835 with enriched selectors
  [ ] Test RBPLCD-8862 with enriched selectors
  [ ] Measure L1 success: Target 40-50%
```

### WEEK 2: Major Improvements

```
Day 1-2: L1 Enrichment
  [ ] Deploy enriched selectors.json
  [ ] Run 5-10 test tickets
  [ ] Measure L1 success: Target 65-75%

Day 3: L2 Enhancement
  [ ] Improve action detection in step_executor.py
  [ ] Add pattern ranking logic
  [ ] Test pattern selection

Day 4: L3 Enhancement
  [ ] Add CV result caching
  [ ] Enhance CV prompts with state context
  [ ] Test caching effectiveness

Day 5: Integration Test
  [ ] Run full test suite
  [ ] Measure all metrics
  [ ] Document improvements
```

### WEEK 3: Polish

```
Day 1-2: Dynamic Selectors
  [ ] Analyze TypeScript for possible values
  [ ] Add dynamic matching logic
  [ ] Test with menu items/dropdowns

Day 3: CV Learning
  [ ] Add L3 → L1 pipeline (save successful selectors)
  [ ] Test learning mechanism
  [ ] Validate learned selectors

Day 4-5: Final Testing
  [ ] Comprehensive test run (20+ tickets)
  [ ] Performance benchmarking
  [ ] Success rate validation
  [ ] Documentation update
```

---

## Quick Start (Do This Now)

```bash
# 1. Backup current code
cp agents/vision_executor_agent.py agents/vision_executor_agent.py.backup

# 2. Make the 2-line change
# Edit agents/vision_executor_agent.py:
#   Line 22: from utils.selector_loader_v2 import SelectorLoaderV2
#   Line 63: selector_loader = SelectorLoaderV2(use_sequential_context=True)
#            selector_loader.reset_state()

# 3. Test it works
python run_test.py RBPLCD-8835

# 4. Check the logs for state tracking
# Look for lines like: "STATE: module=parts, visible_modules=[...]"

# 5. If working, proceed to enrichment
# If not, debug and fix before continuing
```

---

## Success Criteria Summary

| Metric | Current | Week 1 | Week 2 | Week 3 |
|--------|---------|--------|--------|--------|
| L1 Success Rate | 25% | 45% ✅ | 70% ✅ | 85% ✅ |
| Avg Step Time | 5.0s | 3.5s | 1.8s | 1.5s |
| Test Time (8 steps) | 40s | 28s | 14s | 12s |
| CV API Calls | 3-4 | 2-3 | 1 | 0-1 |

**Target Achievement: 85% L1, 3x faster execution, 75% cost reduction** 🎯
