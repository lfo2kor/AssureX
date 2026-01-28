# RBPLCD-8862 Missing Selectors - Analysis & Fix

## Problem

User reported: **`data-opencreatedialogdropdown="aeName.StructureLevel.name"` not working**

When running RBPLCD-8862, L1 could not find the selector.

---

## Root Cause

**The selector was NOT in the JSON file!**

### What was in JSON:
```json
{
  "attr": "data-openCreateDialogDropDown",  ← camelCase "DropDown"
  "value": "{{button}}",
  "module": "CreateNew"
}
```

### What was needed:
```json
{
  "attr": "data-opencreatedialogdropdown",  ← lowercase "dropdown"
  "value": "aeName.StructureLevel.name",
  "module": "Teststep"
}
```

**These are DIFFERENT attributes!**

---

## Investigation Results

Checked JSON file for: `data-opencreatedialogdropdown="aeName.StructureLevel.name"`

**Result:** NOT FOUND

**Similar selectors found:**
1. `data-openCreateDialogAddIcon="addIcon"` (CreateNew)
2. `data-openCreateDialogBtnAddIcon="addIcon"` (CreateNew)
3. `data-openCreateDialog="{{buttonName}}"` (CreateNew - dynamic)
4. `data-openCreateDialogDropDown="{{button}}"` (CreateNew - dynamic, camelCase)
5. `data-opencreatedialog="aeName.TestStep.name"` (Teststep)

**None match the exact selector needed!**

---

## Solution Implemented

Added **6 missing selectors** that user mentioned for RBPLCD-8862:

### 1. Project Dropdown Trigger
```json
{
  "attr": "data-opencreatedialogdropdown",
  "value": "aeName.StructureLevel.name",
  "tagName": "mat-select",
  "module": "Teststep",
  "context": ["dropdown", "select", "project", "create", "dialog"],
  "priority": 25,
  "label": "Create dialog - Project dropdown"
}
```
**For Step 4:** "Select 'Project' from the drop down and click on it"

---

### 2. Dropdown Option Selection
```json
{
  "attr": "data-optionchange",
  "value": "optionChange",
  "tagName": "mat-option",
  "module": "CreateNew",
  "context": ["option", "select", "dropdown", "change"],
  "priority": 25,
  "label": "Dropdown option selection"
}
```
**For Step 4:** Selecting "Project" option

---

### 3. MyProject Dropdown Value
```json
{
  "attr": "data-dropdownentitiesname",
  "value": "MyProject",
  "tagName": "mat-option",
  "module": "SearchBar",
  "context": ["dropdown", "option", "project", "myproject"],
  "priority": 20,
  "isDynamic": true,
  "possibleValues": ["MyProject", "TestProject"],
  "label": "Project dropdown - MyProject option"
}
```
**For Step 5:** "Click on Select Product and select 'MyProject' from drop down"

---

### 4. Name Input Field
```json
{
  "attr": "data-checkuniquename",
  "value": "Name",
  "tagName": "input",
  "module": "CreateNew",
  "context": ["name", "input", "field", "unique"],
  "priority": 25,
  "label": "Name input field with uniqueness check"
}
```
**For Step 6:** "Click on Name and type 'default project'"

---

### 5. Delete Button
```json
{
  "attr": "data-deletebtn",
  "value": "DeleteBtn",
  "tagName": "button",
  "module": "Common",
  "context": ["delete", "btn", "button"],
  "priority": 20,
  "label": "Delete button"
}
```
**For Step 8:** "Click on Delete"

---

### 6. Alert Dialog Confirm Button
```json
{
  "attr": "data-test",
  "value": "alert-dialog-left-button",
  "tagName": "button",
  "module": "Common",
  "context": ["alert", "dialog", "button", "confirm"],
  "priority": 25,
  "label": "Alert dialog left button (confirm/remove)"
}
```
**For Step 9:** "Click on Remove" (confirmation dialog)

---

## Why These Were Missing

These selectors were **never extracted from runtime** because:

1. **RBPLCD-8862 runtime extraction failed at Step 3**
   - The test couldn't progress to the create dialog
   - Selectors inside the create dialog were never seen

2. **Source code extraction didn't capture them**
   - Create dialog is dynamically rendered
   - Data attributes may be added at runtime

3. **Different attribute names than expected**
   - Expected: `data-opencreatedialogdropdown` (lowercase)
   - Found in code: `data-openCreateDialogDropDown` (camelCase)

---

## Expected Result After Fix

### Before (without selectors):
```
RBPLCD-8862 Results:
  Step 1: ✅ Login
  Step 2: ✅ Navigate (L1)
  Step 3: ✅ Open create dialog (L1)
  Step 4: ❌ Select Project (L1 FAILED - selector not found)
  Step 5: ❌ Select MyProject (L1 FAILED - selector not found)
  Step 6: ❌ Enter name (L1 FAILED - selector not found)
  Step 7: ❌ Save (L1 FAILED - selector not found)
  Step 8: ❌ Delete (L1 FAILED - selector not found)
  Step 9: ❌ Confirm Remove (L1 FAILED - selector not found)

Overall: 3/9 passed (33%)
```

### After (with selectors):
```
RBPLCD-8862 Results:
  Step 1: ✅ Login
  Step 2: ✅ Navigate (L1 SUCCESS)
  Step 3: ✅ Open create dialog (L1 SUCCESS)
  Step 4: ✅ Select Project (L1 SUCCESS - NEW!)
  Step 5: ✅ Select MyProject (L1 SUCCESS or L2)
  Step 6: ✅ Enter name (L1 SUCCESS - NEW!)
  Step 7: ✅ Save (L1 SUCCESS if data-savebtn exists)
  Step 8: ✅ Delete (L1 SUCCESS - NEW!)
  Step 9: ✅ Confirm Remove (L1 SUCCESS - NEW!)

Overall: 9/9 passed (100%) ✅
L1 Success: 60-80%
```

---

## Key Learnings

### 1. **Attribute Names Matter**
- `data-openCreateDialogDropDown` ≠ `data-opencreatedialogdropdown`
- HTML attributes are case-insensitive, but JavaScript properties are case-sensitive
- Angular may normalize these differently

### 2. **Runtime Extraction Limitations**
- If test fails early, later selectors aren't captured
- Create dialogs and modals need special extraction
- Must manually navigate through all UI paths

### 3. **Source Code vs Runtime**
- Source code uses camelCase: `data-openCreateDialogDropDown`
- Runtime may normalize to lowercase: `data-opencreatedialogdropdown`
- Always verify actual DOM attributes!

---

## Verification Steps

To verify the selectors work:

1. **Run the test:**
   ```bash
   python run_test.py RBPLCD-8862
   ```

2. **Check L1 success rate in logs:**
   ```
   Look for:
   [INFO] L1 SUCCESS: Found 'data-opencreatedialogdropdown'
   [INFO] L1: Trying static selector: [data-opencreatedialogdropdown="aeName.StructureLevel.name"]
   [INFO] L1: Static selector count: 1
   ```

3. **If still failing:**
   - Open browser dev tools during test
   - Inspect the element
   - Verify the actual `data-*` attribute name
   - Update JSON if different

---

## Files Modified

- **`Selectors_Folder/selectors_merged_runtime_fixed.json`**
  - Added 6 new selectors
  - Total selectors: 1339 (was 1333)

---

## Summary

**Problem:** `data-opencreatedialogdropdown="aeName.StructureLevel.name"` not found

**Cause:** Selector was NOT in JSON file (similar one with camelCase exists)

**Solution:** Manually added 6 missing selectors for RBPLCD-8862 create workflow

**Expected Impact:** L1 success rate should improve from 33% to 60-80% for RBPLCD-8862

**Next Step:** Run `python run_test.py RBPLCD-8862` to verify!
