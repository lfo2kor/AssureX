# YAML-Driven Embedding Configuration Guide

**Date:** 2025-11-18
**Purpose:** Configure embedding fields through YAML for project-specific customization

---

## Overview

The embedding strategy is now **fully configurable** through `plcdtestassistant.yaml`. This allows you to:
- Choose which JSON fields to include in embeddings
- Define custom composite formats
- Separate embedding fields from metadata fields
- Customize per project without code changes

---

## File Locations

### **1. Configuration File**
**File:** `plcdtestassistant.yaml`
**Section:** `selectors.embedding_strategy`
**Purpose:** Define which fields to embed and how to format them

### **2. Setup Script**
**File:** `setup_vectordb.py`
**Function:** `create_composite_text(selector, config)`
**Purpose:** Reads YAML config and builds embedding text

### **3. Execution**
**Command:** `python setup_vectordb.py`
**Purpose:** Creates ChromaDB collection with YAML-configured embeddings

---

## YAML Configuration Structure

```yaml
selectors:
  source_file: "Selectors_Folder/selectors_merged_runtime_fixed.json"

  embedding_strategy:
    # Composite format for embedding text
    composite_format: "{textContent} {ariaLabel} | {role} {tagName} | in {module} | {context} | page: {page_context}"

    # Fields to include in embedding (for semantic search)
    include_fields:
      - "textContent"      # What user sees on screen
      - "ariaLabel"        # Accessibility label
      - "role"             # button, link, input
      - "tagName"          # HTML tag
      - "module"           # Module/page context
      - "context"          # Keywords
      - "pageUrl"          # URL (converted to page_context)

    # Fields stored as metadata ONLY (not in embedding)
    metadata_fields:
      - "attr"             # data-test, data-id, etc.
      - "value"            # Attribute value
      - "full_selector"    # [data-test='...']
      - "id"               # Unique ID
      - "priority"         # Ranking
      - "isDynamic"        # Boolean
      - "elementType"      # Classification
```

---

## How It Works

### **Step 1: YAML Defines Fields**

**include_fields** → Used for embedding (semantic matching)
**metadata_fields** → Stored but NOT embedded (returned after match)

### **Step 2: setup_vectordb.py Reads Config**

```python
def create_composite_text(selector, config):
    # Read from YAML
    embedding_strategy = config['selectors']['embedding_strategy']
    composite_format = embedding_strategy['composite_format']
    include_fields = embedding_strategy['include_fields']

    # Build field values
    field_values = {}
    for field in include_fields:
        if field == "pageUrl":
            # Special: convert URL to page_context
            field_values['page_context'] = extract_page_context(selector['pageUrl'])
        elif field == "context":
            # Special: handle list or string
            field_values['context'] = ' '.join(selector['context'])
        else:
            # Regular field
            field_values[field] = selector.get(field, '')

    # Format using composite_format
    composite = composite_format.format(**field_values)

    return composite
```

### **Step 3: Result**

**For navigation selector:**
```json
{
  "textContent": "Runs",
  "role": "link",
  "tagName": "mat-list-item",
  "module": "Teststep",
  "context": ["sidebar", "navigate", "runs"],
  "pageUrl": "http://.../client/dashboard",
  "attr": "data-test",
  "value": "sidebar-nav-item-nav_item_teststeps"
}
```

**Embedding text (from include_fields):**
```
Runs | link mat-list-item | in Teststep | sidebar navigate runs | page: dashboard
```

**Metadata (from metadata_fields):**
```json
{
  "attr": "data-test",
  "value": "sidebar-nav-item-nav_item_teststeps",
  "full_selector": "[data-test='sidebar-nav-item-nav_item_teststeps']",
  "id": "nav_item_teststeps",
  "priority": 15,
  "isDynamic": false
}
```

---

## Available Placeholders

### **From JSON Fields:**
- `{textContent}` - Visible text on element
- `{ariaLabel}` - Accessibility label
- `{role}` - Element role (button, link, input)
- `{tagName}` - HTML tag
- `{module}` - Module name
- `{context}` - Keywords (list converted to string)
- `{label}` - Field label
- `{elementType}` - Element classification
- `{attr}` - Data attribute name
- `{value}` - Data attribute value

### **Computed Fields:**
- `{page_context}` - Extracted from pageUrl (e.g., "dashboard")

---

## Example Configurations

### **Configuration 1: Current (Human-Readable)**

```yaml
composite_format: "{textContent} {ariaLabel} | {role} {tagName} | in {module} | {context} | page: {page_context}"
include_fields: ["textContent", "ariaLabel", "role", "tagName", "module", "context", "pageUrl"]
```

**Result:**
```
Runs | link mat-list-item | in Teststep | sidebar navigate runs | page: dashboard
```

**Best for:** Matching on visible element properties

---

### **Configuration 2: Minimal (Context-Focused)**

```yaml
composite_format: "{module} | {context} | {page_context}"
include_fields: ["module", "context", "pageUrl"]
```

**Result:**
```
Teststep | sidebar navigate runs | dashboard
```

**Best for:** Simple keyword matching

---

### **Configuration 3: Detailed (Maximum Info)**

```yaml
composite_format: "{textContent} ({ariaLabel}) | {role} {tagName} element | module: {module} | context: {context} | page: {page_context} | label: {label}"
include_fields: ["textContent", "ariaLabel", "role", "tagName", "module", "context", "pageUrl", "label"]
```

**Result:**
```
Runs () | link mat-list-item element | module: Teststep | context: sidebar navigate runs | page: dashboard | label:
```

**Best for:** Maximum semantic information

---

### **Configuration 4: Action-Oriented**

```yaml
composite_format: "Action on {role} with text '{textContent}' in {module} ({page_context})"
include_fields: ["role", "textContent", "module", "pageUrl"]
```

**Result:**
```
Action on link with text 'Runs' in Teststep (dashboard)
```

**Best for:** Natural language matching

---

## Project-Specific Customization

### **Scenario 1: Your Project Has No textContent**

If your selectors don't have `textContent`, use:

```yaml
composite_format: "{role} {tagName} | in {module} | {context} | {page_context}"
include_fields: ["role", "tagName", "module", "context", "pageUrl"]
metadata_fields: ["attr", "value", "full_selector", "id"]
```

---

### **Scenario 2: You Want to Include Attribute Names**

If you want attribute names in embedding:

```yaml
composite_format: "{attr}={value} | {textContent} | {module} | {context}"
include_fields: ["attr", "value", "textContent", "module", "context"]
metadata_fields: ["full_selector", "id", "priority"]
```

**Warning:** This makes embeddings more technical and may reduce semantic matching quality.

---

### **Scenario 3: Multiple Page Types**

If you have selectors from many different pages:

```yaml
composite_format: "{textContent} | {role} | {module} at {page_context} | {context}"
include_fields: ["textContent", "role", "module", "pageUrl", "context"]
```

This emphasizes page context for better filtering.

---

## Rebuilding ChromaDB

### **After Changing YAML:**

```bash
# 1. Backup current database
cp -r data/chromadb data/chromadb_backup_20251118

# 2. Rebuild with new config
python setup_vectordb.py

# 3. Verify
python check_chromadb.py
```

### **What Happens:**

1. `setup_vectordb.py` reads your YAML configuration
2. For each selector in JSON:
   - Builds embedding text using `composite_format` and `include_fields`
   - Stores metadata using `metadata_fields`
3. Generates embeddings via Azure OpenAI
4. Stores to ChromaDB

**Time:** ~2-3 minutes for 1,340 selectors

---

## Testing Your Configuration

### **Test Script:**

```python
# test_embedding_config.py
from config_loader import load_config
from setup_vectordb import create_composite_text

# Load config
config = load_config()

# Sample selector
selector = {
    "textContent": "Runs",
    "role": "link",
    "tagName": "mat-list-item",
    "module": "Teststep",
    "context": ["sidebar", "navigate", "runs"],
    "pageUrl": "http://fe0vm03313.de.bosch.com/rbplcd_t/client/dashboard",
    "attr": "data-test",
    "value": "sidebar-nav-item-nav_item_teststeps"
}

# Generate embedding text
embedding_text = create_composite_text(selector, config)

print("Embedding text:")
print(embedding_text)
```

**Run:**
```bash
python test_embedding_config.py
```

**Expected output:**
```
Embedding text:
Runs | link mat-list-item | in Teststep | sidebar navigate runs | page: dashboard
```

---

## Benefits

### **1. No Code Changes**
- Modify YAML only
- No Python editing required
- Easy to test different formats

### **2. Project-Specific**
- Each project can have custom config
- Different composite formats per use case
- Flexible field selection

### **3. Separation of Concerns**
- Embedding fields = semantic matching
- Metadata fields = technical details
- Clear separation improves quality

### **4. Easy Rollback**
- Keep backup of ChromaDB
- Change YAML and rebuild
- Compare results

---

## Best Practices

### **1. Include Human-Readable Fields**
✅ textContent, ariaLabel, role
❌ attr, value (too technical)

### **2. Use Page Context**
✅ Convert pageUrl to page_context
❌ Store full URLs in embedding

### **3. Separate Technical Details**
✅ Put attr/value in metadata_fields
✅ Return them after matching

### **4. Test Before Production**
- Test with 10-20 selectors first
- Verify embedding text quality
- Check similarity scores

### **5. Document Your Changes**
```yaml
# Custom format for Project XYZ
# Date: 2025-11-18
# Reason: Emphasize textContent over context keywords
composite_format: "{textContent} | {role} in {module}"
```

---

## Troubleshooting

### **Problem: Field Not Found**
**Error:** `KeyError: 'textContent'`

**Solution:** Field doesn't exist in JSON. Either:
1. Remove from `include_fields`
2. Add default value handling in `create_composite_text`

---

### **Problem: Empty Embeddings**
**Issue:** Embedding text is blank or has many empty sections

**Solution:** Check which fields are populated:
```python
# Add to setup_vectordb.py
print(f"Populated fields: {[k for k,v in field_values.items() if v]}")
```

Only include fields that are >50% populated.

---

### **Problem: Poor Matching**
**Issue:** Correct selector ranks low

**Solution:**
1. Check embedding text format
2. Ensure query uses same keywords
3. Consider adding more semantic fields
4. Test different composite_format

---

## Summary

**Key Files:**
- `plcdtestassistant.yaml` - Configuration
- `setup_vectordb.py` - Implementation
- `check_chromadb.py` - Verification

**Key Concepts:**
- `include_fields` → Embedded (semantic)
- `metadata_fields` → Stored only (technical)
- `composite_format` → How to combine fields

**Workflow:**
1. Edit YAML configuration
2. Run `python setup_vectordb.py`
3. Test with `python plcd_taseq.py RBPLCD-8835`
4. Adjust and repeat

---

**Next Step:** Ready to rebuild ChromaDB with the new configuration?
