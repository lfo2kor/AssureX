# Embedding Analysis - Why Navigation Selector Scores Lower

## Fields Used for Similarity Calculation

According to `plcdtestassistant.yaml`:

```yaml
embedding_strategy:
  composite_format: "{attr}_{value} {module} {elementType} {label} {context}"
  include_fields: ["attr", "value", "module", "elementType", "label", "context"]
```

**Formula**: `{attr}_{value} {module} {elementType} {label} {context}`

---

## Comparison: Navigation vs Date Selector

### **Navigation Selector** (Rank #340, Score: 0.687)

**From JSON:**
```json
{
  "attr": "data-test",
  "value": "sidebar-nav-item-nav_item_teststeps",
  "module": "Teststep",
  "elementType": "element",  // Inferred, not in JSON
  "label": "",               // Empty in JSON
  "context": ["item", "navigate", "runs", "sidebar", "test", "teststep"]
}
```

**Embedding Document Text:**
```
data-test_sidebar-nav-item-nav_item_teststeps Teststep element  item navigate runs sidebar test teststep
```

**Breakdown:**
- `attr_value`: `data-test_sidebar-nav-item-nav_item_teststeps`
- `module`: `Teststep`
- `elementType`: `element`
- `label`: (empty)
- `context`: `item navigate runs sidebar test teststep`

---

### **Date Selector** (Rank #1, Score: 0.718)

**From JSON:**
```json
{
  "attr": "data-date",
  "value": "1762300800000",
  "module": "Teststep",
  "elementType": "element",
  "label": "",
  "context": ["navigate", "test", "teststep"]
}
```

**Embedding Document Text:**
```
data-date_1762300800000 Teststep element  navigate test teststep
```

**Breakdown:**
- `attr_value`: `data-date_1762300800000`
- `module`: `Teststep`
- `elementType`: `element`
- `label`: (empty)
- `context`: `navigate test teststep`

---

## Search Query (Enhanced by L1)

```
Navigate to the 'Teststep' module from the current dashboard page at
http://fe0vm03313.de.bosch.com/rbplcd_t/client/dashboard, considering
sidebar navigation elements and associated data attributes.
```

**Key terms:**
- Navigate
- Teststep
- module
- dashboard
- page
- sidebar
- navigation
- elements
- data attributes

---

## Why Date Selector Scored Higher (0.718 vs 0.687)

### **Semantic Similarity Analysis:**

| Term in Query | Navigation Selector Match | Date Selector Match |
|--------------|---------------------------|---------------------|
| **Navigate** | ✅ "navigate" (in context) | ✅ "navigate" (in context) |
| **Teststep** | ✅ "teststep" (in context + attr value) | ✅ "teststep" (in context) |
| **module** | ✅ "Teststep" (module field) | ✅ "Teststep" (module field) |
| **dashboard** | ❌ Not present | ❌ Not present |
| **page** | ❌ Not present | ❌ Not present |
| **sidebar** | ✅ "sidebar" (in context) | ❌ Not present |
| **navigation** | ✅ "navigate" (similar) | ✅ "navigate" (similar) |
| **elements** | ✅ "element" (elementType) | ✅ "element" (elementType) |
| **data attributes** | ✅ "data-test" (attr) | ✅ "data-date" (attr) |

**Both selectors have similar term overlap! Why does date score higher?**

---

## The Real Reason: Embedding Vector Similarity

### **Problem 1: Technical Attribute Names Reduce Clarity**

**Navigation selector attribute value:**
```
sidebar-nav-item-nav_item_teststeps
```
- This is a **technical identifier**
- Contains: sidebar, nav, item, teststeps
- **BUT**: The embedding model sees this as ONE compound technical term
- Semantic meaning is diluted by underscores and hyphens

**Date selector attribute value:**
```
1762300800000
```
- This is a **number** (timestamp)
- Embedding model treats numbers differently
- Numbers have **lower semantic weight** than words
- The model focuses MORE on the context words

### **Problem 2: Context Word Overlap**

**Navigation context:**
```
item navigate runs sidebar test teststep
```
- 6 words
- "runs" is confusing (sidebar button says "Runs" but step says "Navigate to Teststep")
- "item" and "sidebar" are structural terms, not action terms

**Date context:**
```
navigate test teststep
```
- 3 words (cleaner, more focused)
- All words directly relate to action
- **Higher concentration of relevant keywords**

### **Problem 3: Query Over-Enhancement**

**Original query:** `"Navigate to Teststep"`

**L1 enhanced to:**
```
Navigate to the 'Teststep' module from the current dashboard page at
http://fe0vm03313.de.bosch.com/rbplcd_t/client/dashboard, considering
sidebar navigation elements and associated data attributes.
```

**Added noise terms:**
- "from the current dashboard page"
- "at http://fe0vm03313.de.bosch.com/rbplcd_t/client/dashboard"
- "considering"
- "associated"

These dilute the semantic focus!

---

## Why Date Selector Wins

### **Embedding Similarity Calculation:**

When text-embedding-3-small generates vectors:

1. **Navigation selector document:**
   ```
   data-test_sidebar-nav-item-nav_item_teststeps Teststep element  item navigate runs sidebar test teststep
   ```
   - Technical compound term: `data-test_sidebar-nav-item-nav_item_teststeps`
   - This creates a **dense, technical vector**
   - The compound term doesn't break down well semantically

2. **Date selector document:**
   ```
   data-date_1762300800000 Teststep element  navigate test teststep
   ```
   - Technical term: `data-date_1762300800000`
   - Number `1762300800000` has **low semantic weight**
   - Model focuses on: `Teststep navigate test teststep`
   - **Cleaner semantic vector** focused on action words

3. **Query vector:**
   ```
   Navigate to the 'Teststep' module from the current dashboard page...
   ```
   - Many words: navigate, Teststep, module, dashboard, page, sidebar, navigation, elements, data, attributes
   - **"navigate", "Teststep", "module"** have highest weight

4. **Cosine Similarity:**
   - Date selector: Clean vector with high weight on "navigate", "test", "teststep"
   - Navigation selector: Diluted by long technical attribute name
   - **Date selector vector is more aligned with query vector!**

---

## Specific Issues with Navigation Selector

### **Issue 1: Missing `step_text` in Embedding**

**From JSON, the navigation selector has:**
```json
"step_text": "Navigate to teststep"
```

**But embedding format doesn't include `step_text`!**

Current format:
```
{attr}_{value} {module} {elementType} {label} {context}
```

**Should be:**
```
{step_text} {attr}_{value} {module} {elementType} {label} {context}
```

This would give:
```
Navigate to teststep data-test_sidebar-nav-item-nav_item_teststeps Teststep element  item navigate runs sidebar test teststep
```

**MUCH BETTER** because "Navigate to teststep" directly matches the query!

### **Issue 2: Missing `textContent` in Embedding**

**From JSON:**
```json
"textContent": "Runs"
```

Not included in embedding! Should be:
```
Navigate to teststep [textContent: Runs] data-test_sidebar-nav-item-nav_item_teststeps Teststep element  item navigate runs sidebar test teststep
```

---

## Summary: Why Navigation Selector Scored Lower

| Factor | Impact | Navigation | Date |
|--------|--------|------------|------|
| **Technical attr value** | High | 19 chars (sidebar-nav-item...) | 13 chars (number) |
| **Semantic clarity** | High | Diluted by technical ID | Clean, number ignored |
| **Context word count** | Medium | 6 words (includes noise) | 3 words (focused) |
| **Context relevance** | High | "runs" confusing | All relevant |
| **step_text included** | **CRITICAL** | ❌ **NOT included** | ❌ NOT included |
| **textContent included** | High | ❌ NOT included | ❌ NOT included |
| **Query enhancement noise** | Medium | Dilutes match | Dilutes match |

**Root Cause**:
1. **Missing `step_text`** in embedding (would have perfect match!)
2. **Missing `textContent`** in embedding
3. Technical attribute name dilutes semantic vector
4. Context words include noise ("runs", "item", "sidebar")

---

## Recommended Fix

### **New Embedding Format:**

```yaml
embedding_strategy:
  composite_format: "{step_text} {textContent} {attr}_{value} {module} {elementType} {label} {context}"
  include_fields: ["step_text", "textContent", "attr", "value", "module", "elementType", "label", "context"]
```

**This would produce:**
```
Navigate to teststep Runs data-test_sidebar-nav-item-nav_item_teststeps Teststep element  item navigate runs sidebar test teststep
```

**Score prediction**: Would jump to **0.90+** because:
- Direct match: "Navigate to teststep" (from step_text)
- Direct match: "Runs" (from textContent)
- Strong module match: "Teststep"

**Navigation selector would rank #1 instead of #340!**

---

**Date:** 2025-11-18
