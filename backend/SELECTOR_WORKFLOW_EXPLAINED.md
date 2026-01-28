# Complete Selector Workflow - How It Actually Works

## Overview

The system uses a **3-source selector strategy**:
1. **Source Code Selectors** (static, extracted once)
2. **Runtime Selectors** (dynamic, extracted per ticket in learning mode)
3. **Manual Selectors** (added manually when needed)

All are **merged** into ONE JSON file that tests use.

---

## Phase 1: Initial Setup (DONE ONCE)

### Step 1: Extract from Source Code
**File:** `extract_selectors.py` (original extraction)
**Input:** Angular HTML template files (`.html` files in codebase)
**Output:** `Selectors_Folder/selectors_enriched_all_modules.json` (884 selectors)

```bash
# This was run ONCE to get base selectors
python extract_selectors.py
```

**What it does:**
- Scans all HTML files in Angular source code
- Finds all `data-*` attributes
- Creates JSON with metadata (module, context, etc.)
- Result: **884 selectors from static HTML**

**Problem:** Source code HTML ≠ Runtime DOM (Angular transforms it)

---

## Phase 2: Runtime Learning (PER TICKET)

### Step 2: Extract from Running Application
**File:** `extract_runtime_selectors.py`
**Input:** Ticket ID (e.g., RBPLCD-8835)
**Output:** `Selectors_Folder/runtime_selectors_RBPLCD-8835.json`

```bash
# Run for EACH ticket you want to learn from
python extract_runtime_selectors.py RBPLCD-8835
python extract_runtime_selectors.py RBPLCD-8862
```

**What it does:**
1. Opens browser
2. Logs in
3. Executes test steps using L2/L3 (NOT L1, because L1 might not have selectors yet)
4. **At each step:**
   - Extracts ALL `data-*` attributes from LIVE DOM using JavaScript
   - When L2 succeeds, captures the EXACT selector that worked
   - Stores in JSON with metadata

**Output for RBPLCD-8835:**
- Total: 705 selectors extracted
- Learned from L2: 4 selectors
  - `data-test="sidebar-nav-item-nav_item_teststeps"` (Step 2)
  - `data-routerlinkid="905"` (Step 3)
  - `data-editicon="EditIcon"` (Step 5)
  - etc.

**Key Discovery:** ALL 4 learned selectors were NEW (not in source code!)

---

## Phase 3: Merge Selectors

### Step 3: Combine All Sources
**File:** `merge_selectors.py`
**Input:**
- `selectors_enriched_all_modules.json` (884 from source code)
- `runtime_selectors_RBPLCD-8835.json` (705 from runtime)
- `runtime_selectors_RBPLCD-8862.json` (525 from runtime)

**Output:** `Selectors_Folder/selectors_merged_runtime.json`

```bash
# Run AFTER extracting from tickets
python merge_selectors.py
```

**What it does:**
```
1. Load source code selectors (884)
2. Load ALL runtime selector files (705 + 525 = 1230)
3. Merge logic:
   - If selector exists in BOTH: Runtime OVERRIDES source
   - If selector only in source: Keep it
   - If selector only in runtime: Add it (NEW!)
4. Result: 1333 total selectors
```

**Merge Priority:**
```
Runtime > Source Code > Manual

Why? Runtime selectors are VERIFIED to work (learned from L2 success)
```

---

## Phase 4: Fix Priorities & Context

### Step 4: Normalize Merged Selectors
**File:** `fix_selector_priorities_v2.py`
**Input:** `selectors_merged_runtime.json`
**Output:** `selectors_merged_runtime_fixed.json`

```bash
# Run AFTER merge to normalize priorities
python fix_selector_priorities_v2.py
```

**What it does:**
1. **Remove generic selectors** (e.g., `data-test="undefined"`)
2. **Fix priorities** (5-20 range instead of 50-145)
3. **Add context keywords** from step text and textContent
4. **Enhance selectors** with tag+class for specificity

**Example:**
```json
Before:
{
  "attr": "data-test",
  "value": "sidebar-nav-item-nav_item_teststeps",
  "priority": 100,
  "context": []
}

After:
{
  "attr": "data-test",
  "value": "sidebar-nav-item-nav_item_teststeps",
  "priority": 15,
  "context": ["navigate", "runs", "test", "teststep"]
}
```

---

## Phase 5: Manual Additions (AS NEEDED)

### Step 5: Add Missing Selectors Manually
**When:** When a test fails because selector is missing
**Files:** `add_missing_selectors_8862.py`, `add_type_selector_manual.py`

```bash
# Add selectors that weren't captured
python add_missing_selectors_8862.py
```

**What it does:**
Adds selectors you know exist but weren't extracted:
```json
{
  "attr": "data-opencreatedialogdropdown",
  "value": "aeName.StructureLevel.name",
  "priority": 25,
  "context": ["dropdown", "select", "project"],
  "source": "manual_fix"
}
```

---

## Phase 6: Test Execution (USES MERGED FILE)

### Step 6: Run Tests
**File:** `run_test.py`
**Uses:** `Selectors_Folder/selectors_merged_runtime_fixed.json`

```bash
python run_test.py RBPLCD-8835
```

**What happens:**
1. **Load selectors ONCE** (1339 selectors into memory)
2. **For each test step:**
   - L1: Search in-memory list of 1339 selectors
   - L2: Try generic patterns (if L1 fails)
   - L3: Use CV (if L2 fails)
3. **No re-loading** - uses same in-memory selectors for all steps

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Source Code Extraction (ONCE)                     │
├─────────────────────────────────────────────────────────────┤
│ extract_selectors.py                                        │
│   Input:  Angular HTML files (*.html)                      │
│   Output: selectors_enriched_all_modules.json (884)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Runtime Extraction (PER TICKET, LEARNING MODE)    │
├─────────────────────────────────────────────────────────────┤
│ extract_runtime_selectors.py RBPLCD-8835                   │
│   Input:  Ticket ID                                         │
│   Process: Run test, extract from LIVE DOM                 │
│   Output: runtime_selectors_RBPLCD-8835.json (705)         │
│                                                             │
│ extract_runtime_selectors.py RBPLCD-8862                   │
│   Output: runtime_selectors_RBPLCD-8862.json (525)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Merge All Sources                                 │
├─────────────────────────────────────────────────────────────┤
│ merge_selectors.py                                          │
│   Input:  - selectors_enriched_all_modules.json (884)      │
│           - runtime_selectors_RBPLCD-8835.json (705)        │
│           - runtime_selectors_RBPLCD-8862.json (525)        │
│   Logic:  Runtime > Source (priority)                      │
│   Output: selectors_merged_runtime.json (1333)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Fix Priorities & Context                          │
├─────────────────────────────────────────────────────────────┤
│ fix_selector_priorities_v2.py                               │
│   Input:  selectors_merged_runtime.json (1333)              │
│   Process: Remove generic, normalize priority, add context  │
│   Output: selectors_merged_runtime_fixed.json (1331)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Manual Additions (AS NEEDED)                      │
├─────────────────────────────────────────────────────────────┤
│ add_missing_selectors_8862.py                               │
│   Adds: data-opencreatedialogdropdown, etc. (6 selectors)  │
│   Output: selectors_merged_runtime_fixed.json (1339)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: Test Execution (USES MERGED FILE)                 │
├─────────────────────────────────────────────────────────────┤
│ run_test.py RBPLCD-8835                                     │
│   Loads: selectors_merged_runtime_fixed.json (1339)         │
│   Uses:  In-memory search for ALL steps                    │
│   L1:    60-80% success (using merged selectors)            │
│   L2/L3: 20-40% fallback                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Points

### 1. Source Code Extraction (ONCE)
- ✅ Run ONCE to get base selectors
- ❌ Source HTML ≠ Runtime DOM
- Result: 884 selectors (many don't work at runtime)

### 2. Runtime Extraction (PER TICKET)
- ✅ Run for EACH ticket you test
- ✅ Captures REAL selectors from running app
- ✅ Learns from L2 successes
- Result: 700+ selectors per ticket (VERIFIED to work)

### 3. Merge (AFTER EACH EXTRACTION)
- ✅ Combines all sources
- ✅ Runtime overrides source
- Result: Growing database of working selectors

### 4. Test Execution (USES MERGED)
- ✅ Loads merged file ONCE
- ✅ No re-loading during test
- ✅ L1 success improves as database grows

---

## Current State

**Files in Selectors_Folder:**
1. `selectors_enriched_all_modules.json` (884) - Original from source code
2. `runtime_selectors_RBPLCD-8835.json` (705) - Learned from RBPLCD-8835
3. `runtime_selectors_RBPLCD-8862.json` (525) - Learned from RBPLCD-8862
4. `selectors_merged_runtime.json` (1333) - Merged (source + runtime)
5. `selectors_merged_runtime_fixed.json` (1339) - Fixed priorities + manual adds

**Active File:** `selectors_merged_runtime_fixed.json` (1339 selectors)
- Source: 884
- Runtime verified: 709
- Runtime only: 694
- Manual: 6

---

## Workflow for New Ticket

### If Test PASSES:
```bash
# Just run test - no extraction needed
python run_test.py RBPLCD-XXXX
```

### If Test FAILS at L1 (missing selector):
```bash
# Option 1: Extract from runtime
python extract_runtime_selectors.py RBPLCD-XXXX
python merge_selectors.py
python fix_selector_priorities_v2.py

# Option 2: Add manually (faster)
# Edit: add_missing_selectors_XXXX.py
python add_missing_selectors_XXXX.py

# Then retry
python run_test.py RBPLCD-XXXX
```

---

## Growth Strategy

**As you test more tickets:**
1. Extract runtime selectors from each new ticket
2. Merge into master file
3. Selector database grows
4. L1 success rate improves
5. Eventually covers most common UI elements

**Target:** After 10-20 tickets, L1 success should reach 80-90%

---

## Summary

| Question | Answer |
|----------|--------|
| **When is JSON loaded?** | ONCE at test start (vision_executor_agent.py line 63) |
| **Do we reload per step?** | NO - searches in-memory list |
| **Source code extraction?** | ONCE, done already (884 selectors) |
| **Runtime extraction?** | PER TICKET in learning mode (700+ per ticket) |
| **Merge?** | After EACH runtime extraction (combines all) |
| **Active file?** | `selectors_merged_runtime_fixed.json` (1339) |
| **How many sources?** | 3: Source (884) + Runtime (709) + Manual (6) |

**Key Insight:** Runtime extraction is ESSENTIAL because source code HTML ≠ running app DOM!
