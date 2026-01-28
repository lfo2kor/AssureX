# PLCD Testing Assistant - Feedback Tool

## Overview

The Feedback Tool allows testers to correct wrong selectors discovered during test execution and add contextual learning to improve future test runs.

## Why Use the Feedback Tool?

When tests fail or use incorrect selectors, the system may "learn" and store these wrong selectors in the runtime collection. The Feedback Tool allows you to:

1. **Correct wrong selectors** - Replace incorrect selectors with the right ones
2. **Add contextual reasoning** - Explain WHY a selector is correct to help AI learn better
3. **Improve future accuracy** - Corrected selectors are used automatically in future runs
4. **Maintain audit trail** - All corrections are exported to JSON for review

## How It Works

```
┌─────────────────┐
│  Run Test       │  Step 4 uses wrong selector: mat-expansion-panel-header
│  (plcd_ta.py)   │  → Accidentally passes but wrong element clicked
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HTML Report    │  Reports/RBPLCD-8835_20251118_103458_report.html
│  Generated      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feedback Tool  │  python feedback_tool.py Reports/RBPLCD-8835_...html
│  (Tester Input) │  → Tester corrects Step 4 selector
└────────┬────────┘         → Adds reason: "Specific Parts accordion"
         │
         ▼
┌─────────────────┐
│  Runtime        │  [data-expensionpanelheader="aeName.UnitUnderTest.names"]
│  Collection     │  + reason stored with high confidence (0.95)
│  Updated        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Next Run       │  Step 4 now uses corrected selector automatically!
│  (plcd_ta.py)   │  Runtime collection returns corrected selector
└─────────────────┘
```

## Usage

### Step 1: Run a Test
```bash
python plcd_ta.py RBPLCD-8835
```

This generates a report like: `Reports\RBPLCD-8835_20251118_103458_report.html`

### Step 2: Review the Report
Open the HTML report in a browser and identify steps with wrong selectors.

### Step 3: Run Feedback Tool
```bash
python feedback_tool.py Reports\RBPLCD-8835_20251118_103458_report.html
```

### Step 4: Provide Corrections

The tool will:
1. Show current runtime collection statistics
2. Parse and display all steps from the report
3. Ask which steps to correct

**Example Session:**

```
================================================================================
PLCD Testing Assistant - Feedback Tool
================================================================================

Ticket: RBPLCD-8835
Module: Teststep
Total Steps: 6

--------------------------------------------------------------------------------
Step Summary:
--------------------------------------------------------------------------------
✓ Step 1: Login
   Selector: [aria-label='close navigation']
   Agent: L2 | Confidence: 0.42 | Status: PASSED

✓ Step 2: navigate to Teststep
   Selector: [data-test='sidebar-nav-item-nav_item_teststeps']
   Agent: Runtime | Confidence: 0.95 | Status: PASSED

✓ Step 3: click on Teststep named as default_Measurement01
   Selector: [data-attribute='default_Measurement01']
   Agent: Runtime | Confidence: 0.99 | Status: PASSED

✓ Step 4: open parts accordion
   Selector: mat-expansion-panel-header[role='button']
   Agent: Runtime | Confidence: 0.89 | Status: PASSED  ← WRONG!

✓ Step 5: click on edit button of parts default_testobject_01
   Selector: [data-detailview='detailView']
   Agent: L2 | Confidence: 0.67 | Status: PASSED  ← WRONG!

✗ Step 6: Click on Type and select "Type 5" from drop down
   Selector: [data-attribute='Type']
   Agent: L1 | Confidence: 0.55 | Status: FAILED  ← WRONG!

--------------------------------------------------------------------------------
Enter step numbers to correct (comma-separated, e.g., 4,5,6) or 'q' to quit: 4,5,6

================================================================================
Correcting Step 4/6
================================================================================
Step Text: open parts accordion
Current Selector: mat-expansion-panel-header[role='button']
Current Agent: Runtime | Confidence: 0.89
Status: PASSED
--------------------------------------------------------------------------------
Enter correct selector: [data-expensionpanelheader="aeName.UnitUnderTest.names"]

Enter reason/context (optional - helps AI learn better):
Examples:
  - 'Specific Parts accordion, not generic expansion panel'
  - 'Edit button icon, not the detail view container'
  - 'Type dropdown trigger, not the label'
Reason: Specific Parts accordion, not generic expansion panel

--------------------------------------------------------------------------------
Correction Summary:
  Step: open parts accordion
  Old Selector: mat-expansion-panel-header[role='button']
  New Selector: [data-expensionpanelheader="aeName.UnitUnderTest.names"]
  Reason: Specific Parts accordion, not generic expansion panel

Save this correction? (y/n): y
[OK] Correction saved.

[... repeat for Steps 5 and 6 ...]

================================================================================
Saving corrections to runtime collection...
================================================================================

[OK] Successfully saved 3 corrections
[OK] Feedback report exported to: Feedback\RBPLCD-8835_20251118_135500_feedback.json

================================================================================
Runtime Collection Statistics
================================================================================
Total Selectors: 8

By Module:
  Teststep: 8

By Agent:
  Runtime: 5
  UserCorrected: 3  ← New corrections!

User Corrected: 3

================================================================================
Feedback collection complete!
================================================================================

Next time you run the same ticket, the corrected selectors will be used automatically.
Run the test again to verify the corrections:
  python plcd_ta.py RBPLCD-8835
```

### Step 5: Re-run Test to Verify
```bash
python plcd_ta.py RBPLCD-8835
```

The corrected selectors will now be used automatically!

## Key Features

### 1. **Contextual Learning**
When you provide a reason, it's embedded along with the step text:
- **Without reason**: Embedding of `"open parts accordion [data-expensionpanelheader='...']"`
- **With reason**: Embedding of `"open parts accordion Specific Parts accordion, not generic expansion panel"`

The reason enriches the semantic space, making future matches more accurate.

### 2. **Automatic Cleanup**
The tool automatically:
- Finds and deletes old wrong entries for the corrected steps
- Replaces them with corrected versions
- Preserves other learned selectors that were correct

### 3. **Audit Trail**
Every correction session generates a JSON file in the `Feedback` folder:

```json
{
  "ticket_id": "RBPLCD-8835",
  "module": "Teststep",
  "original_report": "Reports\\RBPLCD-8835_20251118_103458_report.html",
  "feedback_date": "2025-11-18T13:55:00.123456",
  "corrections_count": 3,
  "corrections": [
    {
      "step_number": 4,
      "step_text": "open parts accordion",
      "old_selector": "mat-expansion-panel-header[role='button']",
      "correct_selector": "[data-expensionpanelheader=\"aeName.UnitUnderTest.names\"]",
      "reason": "Specific Parts accordion, not generic expansion panel",
      "module": "Teststep",
      "ticket_id": "RBPLCD-8835",
      "original_agent": "Runtime",
      "original_confidence": 0.89
    }
  ]
}
```

### 4. **High Confidence Scoring**
User-corrected selectors are stored with:
- **Confidence: 0.95** (high confidence)
- **Metadata: `user_corrected: true`** for tracking
- **Agent: "UserCorrected"** for identification

This ensures they're prioritized in future runs.

## Best Practices

### When to Provide Corrections

1. **After a test fails** - Correct the failed steps
2. **When selector is wrong but test passes** - Critical! The wrong selector will be learned otherwise
3. **When confidence is low** - Even if step passes, low confidence suggests wrong selector

### Writing Good Reasons

**Good Reasons:**
- ✅ "Specific Parts accordion, not generic expansion panel"
- ✅ "Edit icon button inside the row, not the container"
- ✅ "Type dropdown trigger mat-select, not the display label"
- ✅ "Search button in header toolbar, not pagination search"

**Poor Reasons:**
- ❌ "This is the right one" (not informative)
- ❌ "Use this selector" (doesn't explain why)
- ❌ "" (empty - better than nothing but less helpful)

**Why it matters:** The AI embeds your reason with the step text, so descriptive reasons help it learn the distinction between similar elements.

## Troubleshooting

### "Report file not found"
- Ensure you're providing the full path to the HTML report
- Use tab completion or copy-paste the path

### "No selectors found in ChromaDB"
- The runtime collection might be empty (first run)
- This is normal - corrections will populate it

### Corrections not taking effect
1. Verify corrections were saved: Check the feedback JSON file
2. Check runtime collection stats: Run `python feedback_tool.py <report>` and quit to see stats
3. Clear cache: Delete `database/chroma_db/runtime_learned_collection` to force re-learning

## Technical Details

### How Corrections Are Stored

Each correction creates a new entry in the runtime collection:

```python
{
    'id': 'RBPLCD-8835_step_4_corrected_1234567890.123',
    'metadata': {
        'module': 'Teststep',
        'selector': '[data-expensionpanelheader="aeName.UnitUnderTest.names"]',
        'step_text': 'open parts accordion',
        'confidence': 0.95,
        'user_corrected': True,
        'correction_reason': 'Specific Parts accordion, not generic expansion panel',
        'original_selector': 'mat-expansion-panel-header[role="button"]',
        ...
    },
    'embedding': [0.123, -0.456, ...],  # Generated from step_text + reason
    'document': 'open parts accordion Specific Parts accordion, not generic expansion panel'
}
```

### Retrieval in Future Runs

When `plcd_ta.py` executes Step 4:
1. Generates embedding for `"open parts accordion"`
2. Queries runtime collection
3. Finds corrected entry with high similarity (confidence ~0.95)
4. Uses corrected selector `[data-expensionpanelheader="..."]`
5. Step passes with correct element!

## Summary

The Feedback Tool is essential for:
- **Fixing wrong selectors** that pollute the runtime collection
- **Improving AI accuracy** through contextual learning
- **Building a knowledge base** of correct selectors over time

Use it regularly after test runs to maintain high selector quality!
