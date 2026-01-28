# Enriched Selector Extractor - Implementation Plan

## ⏰ Time Estimates & Checkpoint Strategy

### **Total Estimated Time:**
- **Development:** 2-3 hours
- **First Run (Extraction):** 5-10 minutes
- **Testing:** 30 minutes

**Session Management:** Save after each major step to avoid losing work!

---

## 🎯 Implementation Approach

### **Strategy: Modular + Incremental**

```
Build in phases:
  Phase 1: Core extraction (basic) → Save ✅
  Phase 2: Add dynamic handling → Save ✅
  Phase 3: Add context enrichment → Save ✅
  Phase 4: Add priority calculation → Save ✅

Each phase produces working output!
Can stop and resume at any phase.
```

---

## 📋 Detailed Step-by-Step Plan

### **PHASE 1: Basic Extraction (45 minutes)**

**What We Build:**
- HTML parser
- Basic selector extraction (static only)
- Module detection from folder structure
- Save to JSON

**Output:** `selectors_basic.json` (static selectors only)

**Time Breakdown:**
- Setup & imports: 5 min
- HTML parser: 15 min
- Selector extraction: 15 min
- JSON output: 10 min

**Checkpoint:** ✅ Save code + output

---

### **PHASE 2: Dynamic Selector Handling (45 minutes)**

**What We Build:**
- TypeScript parser
- Dynamic attribute detection
- Variable extraction from .ts files
- Possible values extraction

**Output:** `selectors_with_dynamic.json` (static + dynamic)

**Time Breakdown:**
- TypeScript parser: 20 min
- Dynamic detection: 15 min
- Value extraction: 10 min

**Checkpoint:** ✅ Save code + output

---

### **PHASE 3: Context Enrichment (45 minutes)**

**What We Build:**
- Keyword extraction from names
- Parent context extraction
- Element type analysis
- CSS class analysis
- Framework pattern detection

**Output:** `selectors_enriched.json` (with context)

**Time Breakdown:**
- Keyword extraction: 15 min
- Context analysis: 20 min
- Framework patterns: 10 min

**Checkpoint:** ✅ Save code + output

---

### **PHASE 4: Priority & Final Features (30 minutes)**

**What We Build:**
- Priority calculation
- Usage scenario generation (optional: LLM)
- Final validation
- Statistics report

**Output:** `selectors_enriched_all_modules.json` (complete!)

**Time Breakdown:**
- Priority calculation: 15 min
- Final touches: 10 min
- Testing: 5 min

**Checkpoint:** ✅ Save final code + output

---

## 💾 Checkpoint Strategy

### **What to Save at Each Checkpoint:**

```
After each phase:
1. Save Python code files ✅
2. Save intermediate JSON output ✅
3. Save progress log ✅
4. Save this document with checkmarks ✅

If session times out:
→ Resume from last checkpoint
→ Load intermediate JSON
→ Continue to next phase
```

### **File Structure:**

```
TA_AI_Project/
├── enriched_selector_extractor.py       ← Main code
├── modules/
│   ├── html_parser.py                   ← Phase 1
│   ├── dynamic_extractor.py             ← Phase 2
│   ├── context_enricher.py              ← Phase 3
│   └── priority_calculator.py           ← Phase 4
├── output/
│   ├── phase1_selectors_basic.json      ← After Phase 1
│   ├── phase2_selectors_dynamic.json    ← After Phase 2
│   ├── phase3_selectors_enriched.json   ← After Phase 3
│   └── phase4_selectors_final.json      ← After Phase 4
└── PROGRESS_LOG.txt                     ← Track progress
```

---

## 📊 Extraction Time Estimates

### **For Your Codebase:**

**Your Stats:**
- HTML files: 82
- TypeScript files: 426
- Estimated selectors: ~1200

**Time Estimates:**

| Phase | Operation | Time |
|-------|-----------|------|
| 1 | Parse 82 HTML files | 10 seconds |
| 1 | Extract static selectors | 5 seconds |
| 1 | Save to JSON | 1 second |
| **Phase 1 Total** | | **~20 seconds** |
| | | |
| 2 | Find corresponding .ts files | 15 seconds |
| 2 | Parse TypeScript | 60 seconds |
| 2 | Extract dynamic values | 30 seconds |
| **Phase 2 Total** | | **~2 minutes** |
| | | |
| 3 | Analyze HTML structure | 30 seconds |
| 3 | Extract context keywords | 45 seconds |
| 3 | Detect framework patterns | 20 seconds |
| **Phase 3 Total** | | **~2 minutes** |
| | | |
| 4 | Calculate priorities | 10 seconds |
| 4 | Generate statistics | 5 seconds |
| **Phase 4 Total** | | **~15 seconds** |
| | | |
| **TOTAL EXTRACTION TIME** | | **~5 minutes** |

**✅ Very fast! Not a problem for 5-hour session.**

---

## 🚀 Implementation Order

### **Session 1 (Today - 2 hours):**

**Goal:** Complete Phase 1 & 2

```
Hour 1:
  [0:00-0:15] Discuss & finalize approach
  [0:15-0:45] Build Phase 1: Basic extraction
  [0:45-0:50] Test Phase 1
  [0:50-0:55] Save checkpoint ✅

Hour 2:
  [0:00-0:45] Build Phase 2: Dynamic handling
  [0:45-0:50] Test Phase 2
  [0:50-0:55] Save checkpoint ✅
  [0:55-1:00] Document progress
```

**Output:**
- ✅ Basic selector extraction working
- ✅ Dynamic selector detection working
- ✅ Code saved, can resume

---

### **Session 2 (Next time - 1.5 hours):**

**Goal:** Complete Phase 3 & 4

```
Hour 1:
  [0:00-0:10] Resume from checkpoint
  [0:10-0:50] Build Phase 3: Context enrichment
  [0:50-0:55] Test Phase 3
  [0:55-1:00] Save checkpoint ✅

Hour 2:
  [0:00-0:30] Build Phase 4: Priority & finalization
  [0:30-0:40] Run full extraction
  [0:40-0:50] Validate results
  [0:50-1:00] Document & save ✅
```

**Output:**
- ✅ Complete enriched selector file
- ✅ Ready for testing with L1/L2/L3

---

## 📝 Progress Tracking

### **PROGRESS_LOG.txt:**

```
Enriched Selector Extractor - Implementation Progress
=====================================================

Date: 2025-11-03

[✅] PHASE 1: Basic Extraction (45 min)
  [✅] HTML parser implemented
  [✅] Static selector extraction working
  [✅] Module detection working
  [✅] JSON output: output/phase1_selectors_basic.json
  [✅] Checkpoint saved at: 2025-11-03 15:30

[ ] PHASE 2: Dynamic Handling (45 min)
  [ ] TypeScript parser
  [ ] Dynamic attribute detection
  [ ] Possible values extraction
  [ ] Checkpoint: output/phase2_selectors_dynamic.json

[ ] PHASE 3: Context Enrichment (45 min)
  [ ] Keyword extraction
  [ ] Context analysis
  [ ] Framework pattern detection
  [ ] Checkpoint: output/phase3_selectors_enriched.json

[ ] PHASE 4: Priority & Final (30 min)
  [ ] Priority calculation
  [ ] Final validation
  [ ] Complete: output/phase4_selectors_final.json

TOTAL TIME SPENT: 0:45
ESTIMATED REMAINING: 2:00
```

---

## 🔄 Resume Strategy

### **If Session Times Out:**

**What We Save:**
```
Checkpoint Files:
1. All Python code files ✅
2. Latest JSON output ✅
3. PROGRESS_LOG.txt ✅
4. IMPLEMENTATION_PLAN_WITH_CHECKPOINTS.md (this file) ✅
```

**How to Resume:**

```python
# Next session starts here:

# 1. Check progress
with open('PROGRESS_LOG.txt') as f:
    progress = f.read()
    # Shows: "Phase 2 completed, ready for Phase 3"

# 2. Load last checkpoint
import json
with open('output/phase2_selectors_dynamic.json') as f:
    selectors = json.load(f)

# 3. Continue from next phase
print(f"Resuming: {len(selectors)} selectors loaded")
print("Starting Phase 3: Context Enrichment...")

# 4. Run next phase
# (code continues from where it left off)
```

---

## 📂 Modular Code Structure

### **Main File: `enriched_selector_extractor.py`**

```python
"""
Enriched Selector Extractor
Modular design - each phase can run independently
"""

import json
from pathlib import Path
from modules.html_parser import HTMLParser
from modules.dynamic_extractor import DynamicExtractor
from modules.context_enricher import ContextEnricher
from modules.priority_calculator import PriorityCalculator

class EnrichedSelectorExtractor:
    def __init__(self, codebase_path, output_dir="output"):
        self.codebase_path = Path(codebase_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_all_phases(self):
        """Run all phases with checkpoints"""

        # Phase 1: Basic extraction
        print("\n=== PHASE 1: Basic Extraction ===")
        selectors = self.phase1_basic_extraction()
        self.save_checkpoint("phase1", selectors)

        # Phase 2: Dynamic handling
        print("\n=== PHASE 2: Dynamic Handling ===")
        selectors = self.phase2_dynamic_extraction(selectors)
        self.save_checkpoint("phase2", selectors)

        # Phase 3: Context enrichment
        print("\n=== PHASE 3: Context Enrichment ===")
        selectors = self.phase3_context_enrichment(selectors)
        self.save_checkpoint("phase3", selectors)

        # Phase 4: Priority & finalization
        print("\n=== PHASE 4: Priority & Finalization ===")
        selectors = self.phase4_finalization(selectors)
        self.save_checkpoint("phase4", selectors, final=True)

        return selectors

    def phase1_basic_extraction(self):
        """Extract static selectors from HTML"""
        parser = HTMLParser(self.codebase_path)
        selectors = parser.extract_static_selectors()
        print(f"✅ Extracted {len(selectors)} static selectors")
        return selectors

    def phase2_dynamic_extraction(self, selectors):
        """Add dynamic selector handling"""
        extractor = DynamicExtractor(self.codebase_path)
        selectors = extractor.add_dynamic_values(selectors)
        dynamic_count = sum(1 for s in selectors if s.get('isDynamic'))
        print(f"✅ Found {dynamic_count} dynamic selectors")
        return selectors

    def phase3_context_enrichment(self, selectors):
        """Add context keywords and semantic info"""
        enricher = ContextEnricher()
        selectors = enricher.enrich_context(selectors)
        print(f"✅ Enriched {len(selectors)} selectors with context")
        return selectors

    def phase4_finalization(self, selectors):
        """Calculate priorities and finalize"""
        calculator = PriorityCalculator()
        selectors = calculator.calculate_priorities(selectors)
        print(f"✅ Calculated priorities for {len(selectors)} selectors")
        return selectors

    def save_checkpoint(self, phase, selectors, final=False):
        """Save checkpoint after each phase"""
        if final:
            filename = "selectors_enriched_all_modules.json"
        else:
            filename = f"phase{phase.replace('phase', '')}_selectors.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(selectors, f, indent=2)

        print(f"💾 Checkpoint saved: {output_path}")

        # Update progress log
        self.update_progress_log(phase)

    def update_progress_log(self, phase):
        """Update progress log"""
        # ... update PROGRESS_LOG.txt
        pass

    def resume_from_checkpoint(self, phase):
        """Resume from last checkpoint"""
        checkpoint_file = self.output_dir / f"phase{phase}_selectors.json"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file) as f:
            selectors = json.load(f)

        print(f"✅ Resumed from checkpoint: {len(selectors)} selectors loaded")
        return selectors

# Main execution
if __name__ == "__main__":
    codebase_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src"

    extractor = EnrichedSelectorExtractor(codebase_path)

    # Run all phases (or resume from checkpoint)
    selectors = extractor.run_all_phases()

    print("\n✅ COMPLETE! All phases finished.")
    print(f"📊 Total selectors: {len(selectors)}")
```

---

## 🎯 Step-by-Step Implementation Order

### **TODAY - Let's Start with Phase 1:**

**Step 1: Create folder structure (2 min)**
```bash
mkdir modules
mkdir output
touch enriched_selector_extractor.py
touch modules/html_parser.py
touch PROGRESS_LOG.txt
```

**Step 2: Implement Phase 1 - HTML Parser (30 min)**
- Parse HTML files
- Extract data-* attributes
- Detect element types
- Detect modules from folder structure

**Step 3: Test Phase 1 (10 min)**
- Run on your codebase
- Verify output
- Check for errors

**Step 4: Save Checkpoint (5 min)**
- Save code
- Save output/phase1_selectors.json
- Update PROGRESS_LOG.txt

**STOP HERE if needed! We can resume next session.**

---

## ⏰ What If We Run Out of Time?

### **Scenario: Only 1 hour left in session**

**Option 1: Save current work**
```
1. Save all code written so far ✅
2. Save any test output ✅
3. Update PROGRESS_LOG.txt with status ✅
4. Note: "Completed Phase 1, ready for Phase 2"

Next session:
- Resume by loading phase1_selectors.json
- Continue with Phase 2
```

**Option 2: Quick implementation**
```
If we're close to finishing a phase:
- Complete current phase (even if quick & simple)
- Save checkpoint
- Polish in next session

Better to have working (simple) code than incomplete (complex) code
```

---

## 📊 Time Management Table

| Task | Estimated | If Time Limited |
|------|-----------|-----------------|
| **Phase 1** | 45 min | 30 min (simplify) |
| **Phase 2** | 45 min | Can skip initially |
| **Phase 3** | 45 min | Essential |
| **Phase 4** | 30 min | Can use simple rules |
| **Testing** | 30 min | 15 min (basic) |
| **TOTAL** | 3 hours | 1.5 hours (MVP) |

**MVP (Minimum Viable Product):**
- Phase 1: Basic extraction ✅
- Phase 3: Context (simplified) ✅
- Phase 4: Priority (simple rules) ✅

**Enhancement (Later):**
- Phase 2: Dynamic handling
- Better context analysis
- LLM-based usage scenarios

---

## ✅ Summary

### **Time Estimates:**
- **Development:** 2-3 hours (modular, can split)
- **Extraction:** 5 minutes (very fast!)
- **NOT a time problem** ✅

### **Session Management:**
- **Checkpoints after each phase** ✅
- **Can resume anytime** ✅
- **No work lost** ✅

### **Today's Plan:**
1. **Phase 1: Basic extraction** (45 min)
2. **Phase 2: Dynamic handling** (45 min)
3. **Save checkpoint** ✅
4. **Continue next session if needed**

### **Deliverables Today:**
- ✅ Working basic extractor
- ✅ Phase 1 & 2 complete (static + dynamic)
- ✅ Intermediate output saved
- ✅ Can resume for Phase 3 & 4

---

## 🚀 Ready to Start?

**Let's begin with Phase 1:**

1. Create the folder structure
2. Build HTML parser
3. Extract static selectors
4. Save checkpoint

**Should take ~45 minutes, produces working output.**

**Want to start now?**
