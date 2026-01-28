# File Structure Decision - Single vs Multiple Files

## 🎯 Two Approaches

### **Option 1: Single File (Monolithic)**
```
enriched_selector_extractor.py  (one big file ~1000 lines)
```

### **Option 2: Multiple Files (Modular)**
```
enriched_selector_extractor.py  (main orchestrator ~200 lines)
modules/
  ├── html_parser.py            (~200 lines)
  ├── dynamic_extractor.py      (~200 lines)
  ├── context_enricher.py       (~200 lines)
  └── priority_calculator.py    (~200 lines)
```

---

## ⚖️ Comparison

| Aspect | Single File | Multiple Files |
|--------|-------------|----------------|
| **Easy to share** | ✅ One file | ❌ Need to share folder |
| **Easy to run** | ✅ Just `python file.py` | ✅ Same |
| **Session timeout safe** | ⚠️ Lose all if not saved | ✅ Save module by module |
| **Testing phases** | ⚠️ Test all together | ✅ Test each separately |
| **Code organization** | ⚠️ Long file | ✅ Clear separation |
| **Debugging** | ⚠️ Find bugs in 1000 lines | ✅ Isolate to specific module |
| **Resume from checkpoint** | ⚠️ Harder | ✅ Easier |

---

## 💡 **RECOMMENDED: Hybrid Approach**

**Best of both worlds:**

```
enriched_selector_extractor.py  ← ONE FILE, but organized into CLASSES
                                   (~800-1000 lines, well-structured)
```

### **Why This Is Best:**

✅ **Easy to share:** Just one file
✅ **Easy to run:** `python enriched_selector_extractor.py`
✅ **Session safe:** Build class by class, save after each
✅ **Well organized:** Clear sections with comments
✅ **Easy to test:** Each class can be tested independently
✅ **No folder setup needed:** Just one file to create

---

## 📋 Single File Structure (Recommended)

```python
"""
enriched_selector_extractor.py

Complete enriched selector extractor in one file.
Organized into phases/classes for clarity.

Total: ~800 lines
  - HTMLParser: ~200 lines
  - DynamicExtractor: ~200 lines
  - ContextEnricher: ~200 lines
  - PriorityCalculator: ~100 lines
  - Main orchestrator: ~100 lines
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# ============================================================
# PHASE 1: HTML PARSER (Lines 1-200)
# ============================================================
class HTMLParser:
    """
    Extracts basic selectors from HTML files
    """
    def __init__(self, codebase_path):
        self.codebase_path = Path(codebase_path)

    def extract_static_selectors(self):
        """Extract all static data-* attributes from HTML"""
        # Implementation here
        pass

# ============================================================
# PHASE 2: DYNAMIC EXTRACTOR (Lines 201-400)
# ============================================================
class DynamicExtractor:
    """
    Detects dynamic selectors and extracts possible values from TypeScript
    """
    def __init__(self, codebase_path):
        self.codebase_path = Path(codebase_path)

    def add_dynamic_values(self, selectors):
        """Analyze TypeScript files to find dynamic values"""
        # Implementation here
        pass

# ============================================================
# PHASE 3: CONTEXT ENRICHER (Lines 401-600)
# ============================================================
class ContextEnricher:
    """
    Adds context keywords, semantic information
    """
    def enrich_context(self, selectors):
        """Add context keywords and semantic info"""
        # Implementation here
        pass

# ============================================================
# PHASE 4: PRIORITY CALCULATOR (Lines 601-700)
# ============================================================
class PriorityCalculator:
    """
    Calculates priority scores for selectors
    """
    def calculate_priorities(self, selectors):
        """Calculate priority score (0-10) for each selector"""
        # Implementation here
        pass

# ============================================================
# MAIN ORCHESTRATOR (Lines 701-800)
# ============================================================
class EnrichedSelectorExtractor:
    """
    Main class that orchestrates all phases
    """
    def __init__(self, codebase_path, output_dir="output"):
        self.codebase_path = Path(codebase_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_all_phases(self):
        """Run all phases with checkpoints"""
        # Phase 1
        selectors = self.phase1_basic_extraction()
        self.save_checkpoint("phase1", selectors)

        # Phase 2
        selectors = self.phase2_dynamic_extraction(selectors)
        self.save_checkpoint("phase2", selectors)

        # Phase 3
        selectors = self.phase3_context_enrichment(selectors)
        self.save_checkpoint("phase3", selectors)

        # Phase 4
        selectors = self.phase4_finalization(selectors)
        self.save_checkpoint("phase4", selectors, final=True)

        return selectors

    def save_checkpoint(self, phase, selectors, final=False):
        """Save after each phase"""
        pass

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    codebase_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src"
    extractor = EnrichedSelectorExtractor(codebase_path)
    selectors = extractor.run_all_phases()
    print(f"✅ Complete! {len(selectors)} selectors extracted")
```

---

## 🔄 Development Strategy with Single File

### **How We Build It (Incrementally):**

**Step 1: Create empty structure (5 min)**
```python
# Create file with class skeletons
class HTMLParser:
    pass

class DynamicExtractor:
    pass

# etc.
```
**Save file ✅**

---

**Step 2: Implement HTMLParser (30 min)**
```python
class HTMLParser:
    def __init__(self, codebase_path):
        # implementation
        pass

    def extract_static_selectors(self):
        # implementation
        pass
```
**Save file ✅** (HTMLParser complete)

---

**Step 3: Test HTMLParser (10 min)**
```python
# At bottom of file:
if __name__ == "__main__":
    parser = HTMLParser("C:/path/to/src")
    selectors = parser.extract_static_selectors()
    print(f"Found {len(selectors)} selectors")
```
**Save file + test output ✅**

---

**Step 4: Implement DynamicExtractor (30 min)**
```python
class DynamicExtractor:
    # implementation
    pass
```
**Save file ✅** (2 classes complete)

---

**Continue this pattern for each class...**

---

## 💾 Checkpoint Strategy with Single File

### **What Gets Saved:**

```
After each phase:
1. enriched_selector_extractor.py (updated with new class) ✅
2. output/phase1_selectors.json (data checkpoint) ✅
3. PROGRESS_LOG.txt (status) ✅

Example timeline:
  0:30 - Save: HTMLParser class complete ✅
  1:00 - Save: DynamicExtractor class complete ✅
  1:30 - Save: ContextEnricher class complete ✅
  2:00 - Save: PriorityCalculator class complete ✅
```

### **If Session Times Out:**

```
Next session:
1. Open enriched_selector_extractor.py
2. Check PROGRESS_LOG.txt
   "HTMLParser and DynamicExtractor complete"
3. Continue with ContextEnricher class
4. No work lost! ✅
```

---

## 🎯 Final Recommendation

### **USE SINGLE FILE with Class Organization**

**Reasons:**

1. ✅ **Easy to share:** Just send one file
2. ✅ **Easy to run:** No folder setup needed
3. ✅ **Session safe:** Save file after each class
4. ✅ **Well organized:** Clear sections with classes
5. ✅ **Easy to test:** Can test each class individually
6. ✅ **No imports issues:** Everything in one file
7. ✅ **Easy to resume:** Load file, continue next class

**File Size:**
- Total: ~800-1000 lines
- Well commented
- Clear sections
- Easy to navigate

---

## 📝 How We'll Build It

### **TODAY:**

```
Step 1 (5 min): Create file skeleton
  enriched_selector_extractor.py
  - Import statements
  - Empty class definitions
  - Main execution block
  Save ✅

Step 2 (30 min): Implement HTMLParser class
  - Parse HTML files
  - Extract static selectors
  Save ✅

Step 3 (10 min): Test HTMLParser
  - Run on your codebase
  - Verify output
  Save ✅

Step 4 (30 min): Implement DynamicExtractor class
  - Parse TypeScript
  - Extract dynamic values
  Save ✅

Step 5 (10 min): Test DynamicExtractor
  - Verify dynamic detection
  Save ✅

CHECKPOINT: Phase 1 & 2 complete ✅
Can resume next session for Phase 3 & 4
```

---

## ✅ Summary

### **File Structure:**

**ONE file:** `enriched_selector_extractor.py`

**Organized into:**
- HTMLParser class (~200 lines)
- DynamicExtractor class (~200 lines)
- ContextEnricher class (~200 lines)
- PriorityCalculator class (~100 lines)
- Main orchestrator (~100 lines)

**Total:** ~800 lines, well-structured

**Benefits:**
- ✅ Easy to share (one file)
- ✅ Easy to run (no setup)
- ✅ Easy to save (after each class)
- ✅ Easy to resume (no lost work)
- ✅ Well organized (clear classes)

---

**Ready to create this file?** 🚀
