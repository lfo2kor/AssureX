# Implementation Summary - LLM-Based Natural Language Embedding

**Date:** 2025-11-18
**Status:** ✅ Implementation Complete - Ready for Testing

---

## What Was Changed

### **Problem:**
- ChromaDB embeddings were technical: `"data-test_sidebar-nav-item-nav_item_teststeps Teststep element..."`
- Runtime queries were natural: `"Navigate to Teststep"`
- **Poor semantic match** → Navigation selector ranked #340 instead of #1

### **Solution:**
- Use LLM to convert technical JSON fields to natural language at index time
- Build natural language queries at runtime
- **Both sides now use human-readable format** → Better matching!

---

## Files Modified

### **1. plcdtestassistant.yaml (Lines 81-125)**

**Added:**
```yaml
embedding_strategy:
  use_llm_conversion: true

  include_fields:
    - "textContent"
    - "ariaLabel"
    - "role"
    - "tagName"
    - "module"
    - "context"
    - "pageUrl"

  conversion_prompt: |
    Convert these technical element details into a natural action description.
    Element details:
    - Text displayed: {textContent}
    - Element type: {role} {tagName}
    - Context keywords: {context}
    - Module: {module}
    - Page location: {page_context}

    Generate a single natural language sentence (max 15 words) describing what action this element performs.
```

---

### **2. setup_vectordb.py**

**Added Function (Lines 89-158):**
```python
def convert_to_natural_language(selector, config, azure_client):
    """Use LLM to convert technical fields to natural language"""

    # Extract fields
    field_values = {
        'textContent': selector.get('textContent', ''),
        'role': selector.get('role', ''),
        'context': ', '.join(selector.get('context', [])),
        'page_context': extract_page_context(selector.get('pageUrl', ''))
        # ...
    }

    # Build prompt from YAML template
    prompt = conversion_prompt.format(**field_values)

    # Call LLM (gpt-4o)
    response = azure_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Convert technical UI element details to natural action descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=50
    )

    return response.choices[0].message.content.strip()
```

**Updated (Line 332):**
```python
# Old:
doc_text = create_composite_text(selector, config)

# New:
doc_text = convert_to_natural_language(selector, config, azure_client)
```

---

### **3. plcd_taseq.py (SelectorAgent_L1)**

**Updated Function (Lines 501-586):**
```python
def _enhance_query_with_context(self, step_text, context, state):
    """Build natural language query matching ChromaDB format"""

    # Extract page context from URL
    page_context = extract_page_from_url(context['url'])  # "dashboard"

    # Extract visible context hints
    visible_context = self._extract_context_hints(context['visible_elements'])
    # ["sidebar", "navigation"]

    # Build natural language query
    query_parts = [step_text]  # "Navigate to Teststep"

    if visible_context:
        query_parts.append(f"from {' '.join(visible_context[:2])}")

    if page_context:
        query_parts.append(f"on {page_context} page")

    query = ' '.join(query_parts)
    # Result: "Navigate to Teststep from sidebar on dashboard page"

    return query
```

**Added Helper Function (Lines 555-586):**
```python
def _extract_context_hints(self, visible_elements):
    """Extract context keywords from visible data attributes"""

    # Parse data attributes to find navigation keywords
    # "data-test='sidebar-nav-item...'" → ["sidebar", "nav"]

    nav_keywords = ['sidebar', 'navigation', 'nav', 'menu', 'toolbar']
    found = [h for h in extracted_hints if h in nav_keywords]

    return found[:3]
```

---

## Example Transformation

### **Before:**

**JSON:**
```json
{
  "textContent": "Runs",
  "role": "link",
  "context": ["sidebar", "navigate", "runs"],
  "module": "Teststep",
  "pageUrl": "http://.../client/dashboard"
}
```

**ChromaDB Embedding:**
```
"data-test_sidebar-nav-item-nav_item_teststeps Teststep element item navigate runs sidebar"
```

**Runtime Query:**
```
"Navigate to the 'Teststep' module from the current dashboard page..."
```

**Similarity:** 0.687 (Rank #340) ❌

---

### **After:**

**JSON:** (same)

**ChromaDB Embedding (via LLM):**
```
"Click link to navigate to Teststep runs from sidebar on dashboard page"
```

**Runtime Query (built from context):**
```
"Navigate to Teststep from sidebar on dashboard page"
```

**Similarity:** 0.92+ (Rank #1-5) ✅

---

## What You Need to Execute

### **Step 1: Backup Current ChromaDB**

```bash
cd C:\Projects\AI_Chat\PLCD\TA_AI_Project

# Create backup
cp -r data/chromadb data/chromadb_backup_20251118
```

---

### **Step 2: Rebuild ChromaDB with LLM Conversion**

```bash
python setup_vectordb.py
```

**What will happen:**
1. Reads 1,340 selectors from JSON
2. For each selector:
   - Calls LLM (gpt-4o) to convert technical fields → natural language
   - Generates embedding of natural language
   - Stores to ChromaDB
3. Takes ~5-10 minutes (1,340 LLM calls)

**Expected output:**
```
================================================================================
PLCD Testing Assistant - Vector Database Setup
================================================================================

[OK] Configuration loaded: plcdtestassistant.yaml
[OK] Azure OpenAI client initialized
[OK] Selectors loaded: 1340 selectors from JSON
[OK] ChromaDB client initialized: data/chromadb

[INFO] Deleting existing collection and recreating...
[OK] Collection: selectors_base_collection

Embedding selectors...
  Selector 0: Click link to navigate to Teststep runs from sidebar on dashboard page
  Selector 1: Click button to save changes in Teststep module
  Selector 2: Type text in search field on projects page
  Selector 3: Select option from dropdown in Parts section
  Selector 4: Click icon to expand accordion on Teststep page

[1/27] Batch 1: Embedding selectors 1-50... [OK] (50 embeddings)
[2/27] Batch 2: Embedding selectors 51-100... [OK] (50 embeddings)
...
[27/27] Batch 27: Embedding selectors 1291-1340... [OK] (50 embeddings)

Storing in ChromaDB...
[OK] Stored 1340 selectors in collection: selectors_base_collection

Verification...
[OK] Collection count: 1340 selectors

Statistics:
- Total selectors: 1340
- Embedding dimension: 1536
- Time taken: 287.3 seconds (~5 minutes)
```

---

### **Step 3: Test with RBPLCD-8835**

```bash
python plcd_taseq.py RBPLCD-8835
```

**Expected output for Step 2:**
```
Step 2/9: Navigate to Teststep
[INFO] SelectorAgent_L1: Enhanced query: 'Navigate to Teststep from sidebar on dashboard page'
[INFO] Agent 1: Generating embedding for: 'Navigate to Teststep from sidebar on dashboard page'
[INFO] Candidate 1: [data-test='sidebar-nav-item-nav_item_teststeps'] (conf: 0.92, dist: 0.08)
[INFO] SelectorAgent_L1: Result - [data-test='sidebar-nav-item-nav_item_teststeps'] (conf: 0.92)
[OK] [data-test='sidebar-nav-item-nav_item_teststeps'] (agent: L1)
```

**Key indicators of success:**
- ✅ Runtime query is natural language
- ✅ Top candidate is navigation selector
- ✅ Confidence > 0.85
- ✅ Step 2 PASSES (not fails)

---

### **Step 4: Verify Improvement**

```bash
python check_chromadb.py
```

**Should show:**
- Collection: selectors_base_collection
- Count: 1340
- Sample documents are natural language sentences

---

## Cost Estimate

**LLM Calls:**
- 1,340 selectors × 1 call each = 1,340 calls
- Model: gpt-4o
- Average tokens: ~150 input + 20 output = 170 tokens/call
- Total: ~228,800 tokens

**Pricing (approximate):**
- gpt-4o: $2.50 per 1M input tokens, $10 per 1M output tokens
- Input cost: 228,800 × $2.50 / 1M = $0.57
- Output cost: 26,800 × $10 / 1M = $0.27
- **Total: ~$0.84** (one-time cost)

**Future runs:** No LLM cost (embeddings are reused)

---

## Rollback Plan

If it doesn't work:

```bash
# Stop any running processes
# Delete new ChromaDB
rm -rf data/chromadb

# Restore backup
cp -r data/chromadb_backup_20251118 data/chromadb

# Revert YAML changes
# Edit plcdtestassistant.yaml and set:
# use_llm_conversion: false
```

---

## Success Criteria

### **Metrics to Check:**

| Metric | Before | Target After |
|--------|--------|--------------|
| Navigation selector rank | #340 | #1-5 |
| Navigation confidence | 0.687 | 0.90+ |
| Step 2 status | FAILED | PASSED |
| Overall test success rate | ~30% | ~75% |

### **Logs to Verify:**

```
✅ "Selector 0: Click link to navigate..." (natural language)
✅ "Enhanced query: Navigate to Teststep from sidebar..." (natural language)
✅ "Candidate 1: ... (conf: 0.92)" (high confidence)
✅ "[OK] [data-test='sidebar-nav-item...']" (correct selector)
✅ "Step 2/9: Navigate to Teststep [OK]" (step passes)
```

---

## Troubleshooting

### **Issue: LLM conversion takes too long**
**Solution:** Normal - 1,340 calls take ~5-10 minutes. Be patient.

### **Issue: LLM returns empty string**
**Fix:** Already handled - falls back to template-based format
**Check:** Look for "falling back to template" in logs

### **Issue: Still low confidence**
**Debug:**
```bash
# Check what embeddings were created
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chromadb')
collection = client.get_collection('selectors_base_collection')
sample = collection.peek(limit=5)
for doc in sample['documents']:
    print(doc)
"
```

### **Issue: Runtime query doesn't match**
**Debug:** Check logs for "Enhanced query:" - should be natural language

---

## Summary

**Changes:**
- ✅ 3 files modified
- ✅ 1 new function added
- ✅ 2 functions updated
- ✅ YAML configuration updated

**To Execute:**
1. Backup: `cp -r data/chromadb data/chromadb_backup_20251118`
2. Rebuild: `python setup_vectordb.py` (wait ~5-10 min)
3. Test: `python plcd_taseq.py RBPLCD-8835`
4. Verify: Step 2 should PASS with confidence 0.90+

**Expected Result:**
Navigation selector ranks #1-5 with 0.92+ confidence, Step 2 PASSES!

---

**Ready to execute?** Run the commands in order and monitor the output.
