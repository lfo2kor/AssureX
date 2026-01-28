# 🔧 Runtime Selector Extraction - How To Do It

## 3 Approaches to Extract Selectors from Live Application

---

## **Approach 1: Automated Test-Guided Extraction** ⭐ **RECOMMENDED**

### Concept
Follow the exact same test steps but in "extraction mode" - at each step, capture ALL selectors from the page.

### How It Works

```
┌─────────────────────────────────────────────────┐
│ 1. Read Jira Ticket Test Steps                 │
│    "navigate to teststep"                       │
│    "click on teststep named as X"               │
│    "open parts accordion"                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 2. Execute Each Step with Playwright           │
│    - Use L2 (Generic patterns) to perform step │
│    - Step succeeds → We're on the right page   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 3. Extract Selectors from Current Page DOM     │
│    page.evaluate(() => {                        │
│      // Get all elements with data-* attributes│
│      // Record their properties                 │
│      // Identify which are clickable/visible    │
│    })                                           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 4. Save to runtime_selectors.json              │
│    With context: step number, module, action   │
└─────────────────────────────────────────────────┘
```

### Advantages
✅ **Automated** - No manual work
✅ **Context-aware** - Knows which selectors belong to which step
✅ **Accurate** - Gets exact selectors from running app
✅ **Repeatable** - Can re-run anytime app updates

### Implementation Steps

**Step 1: Create Extraction Script**

```python
class RuntimeSelectorExtractor:
    def __init__(self, page, config):
        self.page = page
        self.config = config
        self.extracted_selectors = []

    def extract_from_current_page(self, step_context):
        """
        Extract all data-* selectors from current page.

        Args:
            step_context: {
                'step_num': 4,
                'step_text': 'open parts accordion',
                'module': 'DetailView',
                'action': 'expand'
            }
        """
        # JavaScript to extract selectors
        selectors = self.page.evaluate("""
            () => {
                const selectors = [];

                // Find all elements with data-* attributes
                const elements = document.querySelectorAll('[data-*]');

                elements.forEach(el => {
                    // Get all data-* attributes
                    const dataAttrs = Array.from(el.attributes)
                        .filter(attr => attr.name.startsWith('data-'));

                    if (dataAttrs.length === 0) return;

                    // Check element properties
                    const rect = el.getBoundingClientRect();
                    const isVisible = rect.width > 0 && rect.height > 0;
                    const isClickable = (
                        el.matches('button, a, input, select, [role="button"], [role="link"]') ||
                        el.onclick !== null ||
                        window.getComputedStyle(el).cursor === 'pointer'
                    );

                    // Extract each data-* attribute
                    dataAttrs.forEach(attr => {
                        selectors.push({
                            attr: attr.name,
                            value: attr.value,
                            tagName: el.tagName.toLowerCase(),
                            className: el.className,
                            textContent: el.textContent.trim().substring(0, 50),
                            isVisible: isVisible,
                            isClickable: isClickable,
                            role: el.getAttribute('role'),
                            ariaLabel: el.getAttribute('aria-label')
                        });
                    });
                });

                return selectors;
            }
        """)

        # Enrich with context
        for selector in selectors:
            selector.update({
                'step_num': step_context['step_num'],
                'step_text': step_context['step_text'],
                'module': step_context['module'],
                'action': step_context['action'],
                'extractedDate': datetime.now().isoformat(),
                'extractionMode': 'runtime'
            })

        self.extracted_selectors.extend(selectors)
        return selectors
```

**Step 2: Integrate with Test Runner**

```python
def run_test_in_extraction_mode(ticket_id):
    """
    Run test in extraction mode - captures selectors at each step.
    """
    # Parse Jira ticket
    jira_data = parse_jira_ticket(ticket_id)

    # Initialize browser
    browser = playwright.chromium.launch()
    page = browser.new_page()

    # Initialize extractor
    extractor = RuntimeSelectorExtractor(page, config)

    # Login
    auto_login(page, config)

    # Execute each step
    for step in jira_data['steps']:
        logger.info(f"Extracting selectors for Step {step['num']}: {step['text']}")

        # Execute step using L2 (Generic patterns) - we know L2 works
        step_executor.execute_step_with_L2_only(step)

        # Wait for page to stabilize
        page.wait_for_load_state('networkidle')

        # Extract selectors from current page
        step_context = {
            'step_num': step['num'],
            'step_text': step['text'],
            'module': detect_module(step['text']),
            'action': detect_action(step['text'])
        }
        selectors = extractor.extract_from_current_page(step_context)

        logger.info(f"  Extracted {len(selectors)} selectors")

    # Save to JSON
    output_file = f"Selectors_Folder/runtime_selectors_{ticket_id}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'ticket_id': ticket_id,
                'extractionDate': datetime.now().isoformat(),
                'extractionMode': 'runtime',
                'testUrl': config['test_url']
            },
            'selectors': extractor.extracted_selectors
        }, f, indent=2)

    logger.info(f"Runtime selectors saved to: {output_file}")
    browser.close()
```

**Step 3: Use Extracted Selectors**

```python
# Merge runtime selectors with source code selectors
def merge_selectors(source_selectors, runtime_selectors):
    """
    Merge runtime selectors with source code selectors.
    Runtime selectors take priority (more accurate).
    """
    merged = {}

    # Add source code selectors first
    for sel in source_selectors:
        key = f"{sel['attr']}={sel['value']}"
        merged[key] = sel

    # Override with runtime selectors (higher priority)
    for sel in runtime_selectors:
        key = f"{sel['attr']}={sel['value']}"
        if key in merged:
            # Runtime version wins
            merged[key].update({
                'runtimeVerified': True,
                'isClickable': sel['isClickable'],
                'isVisible': sel['isVisible']
            })
        else:
            # New selector found only at runtime
            merged[key] = sel

    return list(merged.values())
```

---

## **Approach 2: Interactive Selector Picker** 🖱️

### Concept
User manually goes through the app while a script highlights all data-* elements. User clicks to select the right one.

### How It Works

```python
def interactive_selector_picker():
    """
    Opens app with highlighting overlay.
    User clicks elements they want to capture.
    """
    # Inject highlighting CSS/JS
    page.add_script_tag(content="""
        document.addEventListener('mouseover', (e) => {
            const el = e.target;
            const dataAttrs = Array.from(el.attributes)
                .filter(a => a.name.startsWith('data-'));

            if (dataAttrs.length > 0) {
                // Highlight element
                el.style.outline = '3px solid red';

                // Show tooltip with selector
                const tooltip = document.createElement('div');
                tooltip.textContent = dataAttrs.map(a =>
                    `[${a.name}="${a.value}"]`
                ).join(', ');
                tooltip.style.cssText = 'position:absolute; background:black; color:white; padding:5px;';
                document.body.appendChild(tooltip);
            }
        });

        document.addEventListener('click', (e) => {
            if (e.shiftKey) {  // Shift+Click to capture
                const dataAttrs = Array.from(e.target.attributes)
                    .filter(a => a.name.startsWith('data-'));

                // Send to Python
                window.capturedSelector = dataAttrs;
            }
        });
    """)
```

### Advantages
✅ **Accurate** - User confirms correct element
✅ **Visual** - Can see what's being selected
✅ **Flexible** - Works for any app flow

### Disadvantages
❌ **Manual** - Requires user interaction
❌ **Slow** - One element at a time

---

## **Approach 3: Hybrid Learning Mode** 🧠

### Concept
Run tests normally. When L2 succeeds, capture that selector and save it for next time.

### How It Works

```python
class LearningModeExecutor:
    def execute_step(self, step):
        # Try L1 first
        success, selector = try_L1(step)

        if success:
            return (True, selector, "L1")

        # Try L2
        success, selector = try_L2(step)

        if success:
            # LEARN THIS SELECTOR!
            self.learn_selector(step, selector)
            return (True, selector, "L2")

        # Try L3 (CV)
        return try_L3(step)

    def learn_selector(self, step, successful_selector):
        """
        Save successful L2 selector for future use.
        """
        # Parse the L2 selector to extract data-* attribute
        # Example: "button:has-text('Save')" → find data-* on this button

        element = self.page.locator(successful_selector).first

        # Get all data-* attributes from this element
        data_attrs = element.evaluate("""
            el => Array.from(el.attributes)
                .filter(a => a.name.startsWith('data-'))
                .map(a => ({name: a.name, value: a.value}))
        """)

        if data_attrs:
            # Save to learned selectors
            learned_selector = {
                'attr': data_attrs[0]['name'],
                'value': data_attrs[0]['value'],
                'module': self.current_module,
                'step_text': step['text'],
                'learned_from': successful_selector,
                'learnedDate': datetime.now().isoformat(),
                'confidence': 0.8
            }

            self.selector_loader.add_learned_selector(learned_selector)
```

### Advantages
✅ **Zero manual work** - Fully automated
✅ **Self-improving** - Gets better over time
✅ **No extra test runs** - Learns during normal tests

### Disadvantages
❌ **Slow initial phase** - First run uses L2
❌ **Needs multiple runs** - Builds up knowledge gradually

---

## 🎯 **Recommended: Hybrid Approach**

Combine **Approach 1** (automated extraction) + **Approach 3** (learning mode):

```
Phase 1: One-time extraction
├─ Run extraction script on key test tickets
├─ Captures 80-90% of selectors
└─ Saves to runtime_selectors.json

Phase 2: Ongoing learning
├─ Run tests normally with learning mode enabled
├─ When new selectors discovered, add to JSON
└─ System improves continuously
```

---

## 📊 Comparison Table

| Approach | Accuracy | Speed | Manual Work | Maintenance |
|----------|----------|-------|-------------|-------------|
| **1. Automated Extraction** | 95% | Fast (5 min/test) | None | Low |
| **2. Interactive Picker** | 100% | Slow (30 min/test) | High | Low |
| **3. Hybrid Learning** | 85% (grows) | Fast | None | None |
| **1 + 3 Hybrid** | 95%+ | Fast | None | None |

---

## 🚀 Implementation Plan

### **Week 1: Automated Extraction**
1. Create `runtime_selector_extractor.py`
2. Run on 5-10 key test tickets
3. Generate `runtime_selectors.json`
4. Merge with existing `selectors_enriched_all_modules.json`

### **Week 2: Validation**
1. Test with merged selectors
2. Measure L1 success rate improvement
3. Expected: 40% → 85%+

### **Week 3: Learning Mode**
1. Implement `LearningModeExecutor`
2. Enable for all tests
3. System self-improves over time

---

## 💾 Example Output

### Runtime Selector JSON
```json
{
  "metadata": {
    "ticket_id": "RBPLCD-8835",
    "extractionDate": "2025-11-04T11:30:00",
    "extractionMode": "runtime",
    "testUrl": "http://fe0vm03313.de.bosch.com/rbplcd_t"
  },
  "selectors": [
    {
      "attr": "data-saveBtn",
      "value": "SaveBtn",
      "tagName": "button",
      "className": "mat-mdc-button",
      "textContent": "Save",
      "isVisible": true,
      "isClickable": true,
      "role": null,
      "ariaLabel": null,
      "step_num": 7,
      "step_text": "click on save",
      "module": "DetailView",
      "action": "save",
      "extractedDate": "2025-11-04T11:30:15",
      "extractionMode": "runtime",
      "runtimeVerified": true
    },
    {
      "attr": "data-attribute",
      "value": "Type",
      "tagName": "input",
      "className": "mat-mdc-autocomplete-trigger",
      "textContent": "",
      "isVisible": true,
      "isClickable": true,
      "role": "combobox",
      "ariaLabel": "Type",
      "step_num": 6,
      "step_text": "Click on Type from mandatory field",
      "module": "DetailView",
      "action": "dropdown_select",
      "extractedDate": "2025-11-04T11:30:12",
      "extractionMode": "runtime",
      "runtimeVerified": true
    }
  ]
}
```

---

## ✅ Next Steps

Would you like me to:
1. **Implement the automated extraction script** (Approach 1)?
2. **Implement learning mode** (Approach 3)?
3. **Implement both** (1 + 3 hybrid)?

I recommend **option 3** - implement both for maximum benefit!
