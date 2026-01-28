# LLM-Based Intelligent Level Selection Strategy

## 🎯 The Brilliant Idea

**Current Approach (Blind Fallback):**
```
Step → Try L1 → Fail → Try L2 → Fail → Try L3 → Maybe succeed
        0.1s     1s           3s
```

**New Approach (LLM-Guided Intelligence):**
```
Step → LLM analyzes step + context → Predicts best level → Execute directly
       0.2s                          0.1s-3s (only what's needed)
```

**Benefits:**
- ✅ Skip levels that won't work
- ✅ Go directly to best level
- ✅ Generate targeted selectors
- ✅ Learn from previous steps
- ✅ Understand step complexity

---

## 🧠 LLM Intelligence Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM INTELLIGENCE LAYER                   │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  Input:                                                     │
│    • JIRA step text                                         │
│    • Sequential context (previous steps, state)            │
│    • Available selectors (selectors.json)                  │
│    • Test history (what worked before)                     │
│                                                             │
│  LLM Analysis:                                              │
│    • Step complexity assessment                            │
│    • Action type classification                            │
│    • Required modules identification                       │
│    • Level prediction (L1/L2/L3)                           │
│    • Selector strategy generation                          │
│                                                             │
│  Output:                                                    │
│    • Best level to try: L1/L2/L3                           │
│    • Targeted selector strategy                            │
│    • Fallback plan                                         │
│    • Confidence score                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│              INTELLIGENT EXECUTION ENGINE                   │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  If LLM predicts L1 (high confidence):                     │
│    → Execute L1 with targeted strategy                     │
│    → If fails, try LLM's fallback plan                     │
│                                                             │
│  If LLM predicts L2 (medium complexity):                   │
│    → Skip L1, go directly to L2                            │
│    → Use LLM-generated pattern hints                       │
│                                                             │
│  If LLM predicts L3 (complex/ambiguous):                   │
│    → Skip L1/L2, go directly to L3                         │
│    → Use LLM-generated CV prompts                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 LLM Analysis Components

### 1. Step Complexity Assessment

**LLM analyzes step and classifies:**

```python
LLM_PROMPT = """
Analyze this test step and classify its complexity:

Step: "{step_text}"
Previous steps: {previous_steps_summary}
Current state: {state_info}

Classify as:
1. SIMPLE: Single action, clear selector (e.g., "Click Save button")
2. MEDIUM: Requires pattern matching (e.g., "Select Type from dropdown")
3. COMPLEX: Multiple actions or ambiguous (e.g., "Create task under X and copy to Y")
4. ROW_SCOPED: Action on specific row (e.g., "Click edit of part named X")
5. CROSS_MODULE: Action spans modules (e.g., "Expand Parts accordion" from Teststep)

Output JSON:
{
  "complexity": "SIMPLE|MEDIUM|COMPLEX|ROW_SCOPED|CROSS_MODULE",
  "action_type": "click|navigate|input|select|verify|expand",
  "target_element": "button|dropdown|input|accordion|row",
  "requires_row_scoping": true/false,
  "requires_cross_module": true/false,
  "confidence": 0.0-1.0
}
"""
```

**Example Analysis:**

**Step: "Open parts accordion"**
```json
{
  "complexity": "CROSS_MODULE",
  "action_type": "expand",
  "target_element": "accordion",
  "requires_row_scoping": false,
  "requires_cross_module": true,
  "confidence": 0.95,
  "reasoning": "User is in Teststep module but needs to access Parts accordion (different module)"
}
```

---

### 2. Level Prediction with Reasoning

**LLM predicts best level:**

```python
LLM_PROMPT = """
Based on step analysis, predict the best level to execute:

Step: "{step_text}"
Complexity: {complexity}
Sequential context: {context}
Available selectors: {selector_count} in selectors.json

Level capabilities:
- L1: Fast (0.1s), custom selectors from JSON, requires exact match
- L2: Medium (1s), generic HTML patterns, handles row scoping
- L3: Slow (3s), CV-guided, handles ambiguity and complex cases

Predict:
{
  "recommended_level": "L1|L2|L3",
  "confidence": 0.0-1.0,
  "reasoning": "why this level",
  "skip_levels": ["L1", "L2"],  // levels to skip
  "fallback_plan": {
    "if_fails": "L2",
    "reason": "..."
  }
}
"""
```

**Example Prediction:**

**Step: "Open parts accordion" (from Teststep context)**
```json
{
  "recommended_level": "L1",
  "confidence": 0.85,
  "reasoning": "Sequential context should enable cross-module selector matching. Parts accordion likely in selectors.json with module='parts'. With sequential context, visible_modules will include 'parts'.",
  "skip_levels": [],
  "fallback_plan": {
    "if_fails": "L2",
    "reason": "L2 has accordion patterns that can match by text content"
  }
}
```

**Step: "Click on edit button of part default_testobject_01"**
```json
{
  "recommended_level": "L2",
  "confidence": 0.90,
  "reasoning": "Requires row scoping ('of part default_testobject_01'). L1 doesn't handle row scoping well. L2 has specialized row scoping patterns.",
  "skip_levels": ["L1"],
  "fallback_plan": {
    "if_fails": "L3",
    "reason": "If L2 row scoping fails, CV can visually identify the row"
  }
}
```

**Step: "Create a new task under default_StructureLevel_1"**
```json
{
  "recommended_level": "L3",
  "confidence": 0.95,
  "reasoning": "Complex multi-action step: (1) Open create dialog, (2) Select parent from dropdown, (3) Fill form. Requires visual guidance and multiple interactions. Neither L1 nor L2 can handle this complexity.",
  "skip_levels": ["L1", "L2"],
  "fallback_plan": {
    "if_fails": "MANUAL",
    "reason": "This step may require human intervention"
  }
}
```

---

### 3. Targeted Selector Strategy

**LLM generates specific selector strategy for the predicted level:**

**For L1:**
```python
LLM_PROMPT = """
Generate L1 selector search strategy:

Step: "{step_text}"
State: {sequential_state}
Available selectors: {selectors_summary}

Generate:
{
  "search_keywords": ["keyword1", "keyword2", "keyword3"],
  "search_modules": ["module1", "module2"],
  "required_context": ["context1", "context2"],
  "priority_filter": ">=8",
  "exact_selector_hint": "data-parts-accordion"  // if known
}
"""
```

**Example:**
```json
{
  "search_keywords": ["parts", "accordion", "panel", "expand"],
  "search_modules": ["parts", "teststep"],
  "required_context": ["accordion", "expansion-panel"],
  "priority_filter": ">=7",
  "exact_selector_hint": null
}
```

**For L2:**
```python
LLM_PROMPT = """
Generate L2 pattern strategy:

Step: "{step_text}"
Action type: {action_type}
Row identifier: {row_identifier}

Generate:
{
  "pattern_priority": [
    "pattern_template_1",
    "pattern_template_2"
  ],
  "extracted_text": "Save",
  "row_scoping_strategy": "exact_text_match"
}
"""
```

**For L3:**
```python
LLM_PROMPT = """
Generate L3 CV prompt enhancement:

Step: "{step_text}"
Context: {full_context}

Generate:
{
  "cv_focus_areas": ["top-right corner", "action bar"],
  "expected_element_type": "button with icon",
  "spatial_hints": "near the row containing 'default_testobject_01'",
  "visual_characteristics": "edit icon (pencil)"
}
"""
```

---

## 🔄 Complete Flow Example

### Example: RBPLCD-8835 Step 4 - "Open parts accordion"

#### Phase 1: LLM Analysis

**Input to LLM:**
```json
{
  "step_text": "open parts accordion",
  "step_number": 4,
  "jira_module": "teststep",
  "previous_steps": [
    {"num": 1, "text": "Login", "status": "PASSED"},
    {"num": 2, "text": "navigate to teststep", "status": "PASSED"},
    {"num": 3, "text": "click on teststep named as default_Measurement01", "status": "PASSED"}
  ],
  "sequential_state": {
    "current_module": "teststep",
    "navigation_path": ["login", "teststep"],
    "visible_modules": ["teststep"],
    "edit_mode": false,
    "viewing_details": true
  },
  "available_selectors_summary": {
    "total": 888,
    "modules": ["teststep", "parts", "create-new", "entity-attribute", "..."],
    "accordion_selectors": 3
  }
}
```

**LLM Response:**
```json
{
  "complexity_analysis": {
    "complexity": "CROSS_MODULE",
    "action_type": "expand",
    "target_element": "accordion",
    "requires_row_scoping": false,
    "requires_cross_module": true,
    "confidence": 0.95
  },

  "level_prediction": {
    "recommended_level": "L1",
    "confidence": 0.85,
    "reasoning": "This is a cross-module selector scenario. User is viewing teststep details, but Parts accordion belongs to the 'parts' module. With sequential context enabled, the system will detect the 'expand' action and add 'parts' module to visible_modules. L1 should find the accordion selector in selectors.json.",
    "skip_levels": [],
    "time_estimate": "0.1-0.2s",
    "fallback_plan": {
      "if_fails": "L2",
      "reason": "L2 has generic accordion patterns: .mat-expansion-panel-header:has-text('Parts')"
    }
  },

  "l1_strategy": {
    "search_keywords": ["parts", "accordion", "panel", "expansion"],
    "search_modules": ["parts", "teststep"],
    "required_context": ["accordion", "expansion-panel", "section"],
    "priority_filter": ">=7",
    "exact_selector_hint": "data-partspanel or data-parts-accordion",
    "scoring_weights": {
      "module_match": 20,
      "context_match": 10,
      "keyword_match": 5
    }
  },

  "expected_outcome": {
    "success_probability": 0.85,
    "failure_reasons": [
      "Selector not in JSON",
      "Module dependency not configured",
      "Selector attribute name different than expected"
    ]
  }
}
```

#### Phase 2: Intelligent Execution

```python
def execute_step_with_llm_guidance(step, state):
    # Get LLM analysis
    llm_analysis = llm.analyze_step(step, state)

    recommended_level = llm_analysis['level_prediction']['recommended_level']
    confidence = llm_analysis['level_prediction']['confidence']

    logger.info(f"LLM recommends: {recommended_level} (confidence: {confidence})")
    logger.info(f"Reasoning: {llm_analysis['level_prediction']['reasoning']}")

    # Execute based on LLM recommendation
    if recommended_level == "L1" and confidence >= 0.7:
        # Try L1 with LLM-generated strategy
        result = execute_l1_with_strategy(step, llm_analysis['l1_strategy'])

        if result.success:
            logger.info("✅ L1 succeeded as predicted by LLM")
            return result
        else:
            # Try LLM's fallback plan
            fallback = llm_analysis['level_prediction']['fallback_plan']
            logger.info(f"L1 failed, trying {fallback['if_fails']} as suggested by LLM")
            return execute_level(fallback['if_fails'], step)

    elif recommended_level == "L2":
        logger.info("Skipping L1, going directly to L2 as recommended")
        return execute_l2_with_strategy(step, llm_analysis['l2_strategy'])

    elif recommended_level == "L3":
        logger.info("Skipping L1/L2, going directly to L3 as recommended")
        return execute_l3_with_strategy(step, llm_analysis['l3_strategy'])
```

---

## 📊 LLM Decision Matrix

**LLM uses this decision logic:**

| Step Characteristics | LLM Recommendation | Reasoning |
|---------------------|-------------------|-----------|
| **Simple action + selector exists** | L1 (conf: 0.9) | "Click Save" - clear button, likely in JSON |
| **Cross-module + sequential context** | L1 (conf: 0.8) | "Open Parts" from Teststep - context handles it |
| **Row scoping required** | L2 (conf: 0.9) | "Click edit of part X" - L2 has row patterns |
| **Generic pattern** | L2 (conf: 0.8) | "Select Type dropdown" - standard UI pattern |
| **Multi-action complex** | L3 (conf: 0.95) | "Create task under X" - needs CV guidance |
| **Ambiguous text** | L3 (conf: 0.9) | "Click ... +" - visual identification needed |
| **Verification step** | L2 (conf: 0.85) | "Message displayed" - text search patterns |

---

## 💡 Advanced LLM Features

### Feature 1: Learning from History

```python
LLM_PROMPT = """
Analyze step execution history to improve predictions:

Current step: "{step_text}"
Similar past steps:
1. "open parts accordion" → L1 succeeded (RBPLCD-8835)
2. "expand attributes section" → L1 succeeded (RBPLCD-9001)
3. "open measurements panel" → L1 failed, L2 succeeded (RBPLCD-8920)

Pattern identified: "open/expand" steps for sections succeed in L1 85% of time

Prediction for current step: L1 (confidence: 0.87, based on history)
"""
```

**Implementation:**
```python
class LLMWithMemory:
    def __init__(self):
        self.execution_history = []

    def analyze_with_history(self, step):
        # Find similar past steps
        similar_steps = self.find_similar_steps(step)

        # Include in LLM prompt
        analysis = llm.analyze(
            step=step,
            history=similar_steps,
            success_patterns=self.extract_patterns()
        )

        # Update history
        self.execution_history.append({
            'step': step,
            'prediction': analysis,
            'actual_result': None  # filled after execution
        })

        return analysis
```

---

### Feature 2: Dynamic Selector Generation

**LLM can generate selectors on-the-fly:**

```python
LLM_PROMPT = """
Generate a selector for this step:

Step: "Click on edit button of part default_testobject_01"
Context: Viewing teststep details, Parts accordion expanded
UI Framework: Angular Material

Generate selector that:
1. Finds the row containing "default_testobject_01"
2. Locates the edit button within that row

Output:
{
  "selector": "tr:has-text('default_testobject_01') >> [data-editicon]",
  "type": "playwright_css",
  "confidence": 0.85,
  "explanation": "Finds table row with text, then edit icon within that row"
}
"""
```

**This enables:**
- Generate selectors for steps not in JSON
- Adapt to UI changes
- Handle edge cases

---

### Feature 3: Intelligent Fallback Planning

**LLM creates multi-level fallback strategies:**

```python
{
  "primary_strategy": {
    "level": "L1",
    "selector": "data-parts-accordion",
    "expected_success": 0.85
  },
  "fallback_strategies": [
    {
      "level": "L1",
      "selector": "data-partspanel",
      "condition": "if primary not found",
      "expected_success": 0.70
    },
    {
      "level": "L2",
      "pattern": ".mat-expansion-panel-header:has-text('Parts')",
      "condition": "if L1 selectors not found",
      "expected_success": 0.90
    },
    {
      "level": "L3",
      "cv_prompt": "Find the accordion header with text 'Parts'",
      "condition": "if L2 pattern fails",
      "expected_success": 0.95
    }
  ]
}
```

---

## 🎯 Implementation Plan

### Week 1: LLM Analysis Layer

**Add LLM intelligence before execution:**

```python
# In step_executor.py

def execute_step(self, step):
    # NEW: Get LLM guidance
    llm_guidance = self.llm_analyzer.analyze_step(
        step_text=step['text'],
        step_number=step['num'],
        sequential_state=self.context_tracker.state,
        history=self.execution_history
    )

    logger.info(f"LLM Analysis: {llm_guidance['level_prediction']}")

    # Execute based on LLM recommendation
    recommended_level = llm_guidance['recommended_level']

    if recommended_level == "L1":
        success = self._try_level1_with_llm_strategy(
            step['text'],
            llm_guidance['l1_strategy']
        )
        if success:
            return success

        # Try LLM's fallback
        if llm_guidance['fallback_plan']['if_fails'] == "L2":
            return self._try_level2_with_llm_strategy(...)

    elif recommended_level == "L2":
        logger.info("LLM suggests skipping L1, going to L2")
        return self._try_level2_with_llm_strategy(...)

    elif recommended_level == "L3":
        logger.info("LLM suggests skipping L1/L2, going to L3")
        return self._try_level3_with_llm_strategy(...)
```

---

### Week 2: LLM Learning & Optimization

**Track success/failure, improve predictions:**

```python
class LLMAnalyzer:
    def learn_from_execution(self, step, prediction, actual_result):
        """Update LLM's understanding based on actual results"""

        # Store result
        self.history.append({
            'step': step,
            'predicted_level': prediction['recommended_level'],
            'predicted_confidence': prediction['confidence'],
            'actual_level_succeeded': actual_result['level_used'],
            'success': actual_result['status'] == 'PASSED'
        })

        # If prediction was wrong, analyze why
        if prediction['recommended_level'] != actual_result['level_used']:
            self.analyze_misprediction(step, prediction, actual_result)

        # Update patterns
        self.update_success_patterns()

    def update_success_patterns(self):
        """Extract patterns from execution history"""

        patterns = {
            'cross_module_steps': {'success_rate': 0.85, 'best_level': 'L1'},
            'row_scoped_steps': {'success_rate': 0.90, 'best_level': 'L2'},
            'complex_steps': {'success_rate': 0.75, 'best_level': 'L3'}
        }

        # Use patterns in future predictions
        self.learned_patterns = patterns
```

---

## 📈 Expected Benefits

### Time Savings

**Current (Blind Fallback):**
```
Step → L1 (fail, 0.1s) → L2 (fail, 1s) → L3 (success, 3s) = 4.1s total
```

**With LLM (Smart Skip):**
```
Step → LLM analysis (0.2s) → L3 directly (success, 3s) = 3.2s total
Savings: 0.9s (22% faster)
```

**For complex steps that need L3:**
- Current: Always waste 1.1s trying L1/L2 first
- LLM: Skip directly to L3
- **Savings: 1.1s per complex step**

---

### Success Rate Improvement

**Better level selection:**
- L1: Used for 85% of simple steps (vs 60% now)
- L2: Used for row-scoped steps (90% success vs 70%)
- L3: Only for truly complex steps (95% success vs 75%)

**Overall improvement:**
- Current success: ~92%
- With LLM: ~96%
- **+4% improvement**

---

### Cost Reduction

**Fewer L3 calls:**
- Current: 20% of steps use L3 (expensive CV)
- With LLM: 10% of steps use L3 (smart skip from L1/L2)
- **50% reduction in CV costs**

**LLM costs:**
- Analysis per step: ~$0.001 (much cheaper than CV)
- Saves by reducing L3: ~$0.02 per test
- **Net savings: ~$0.015 per test**

---

## 🚀 Quick Start Implementation

### Minimal Version (This Week)

```python
# Add to step_executor.py

from openai import AzureOpenAI

class LLMStepAnalyzer:
    def __init__(self):
        self.client = AzureOpenAI(...)

    def analyze_step(self, step_text, state):
        prompt = f"""
        Analyze this test step and recommend the best execution level:

        Step: "{step_text}"
        Current state: {state}

        Output JSON:
        {{
          "recommended_level": "L1|L2|L3",
          "confidence": 0.0-1.0,
          "reasoning": "brief explanation"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

# Use in executor
llm_analyzer = LLMStepAnalyzer()
guidance = llm_analyzer.analyze_step(step_text, state)

if guidance['recommended_level'] == 'L2' and guidance['confidence'] > 0.8:
    logger.info("LLM suggests skipping L1")
    return self._try_level2(...)
```

---

## 📊 Test Results Prediction

**RBPLCD-8835 with LLM:**

| Step | Text | LLM Predicts | Current | With LLM |
|------|------|--------------|---------|----------|
| 1 | Login | L1 (0.95) | L1 ✅ | L1 ✅ |
| 2 | Navigate | L1 (0.90) | L1 ✅ | L1 ✅ |
| 3 | Click row | L2 (0.85) | L2 ⚠️ | L2 ✅ (skip L1) |
| 4 | **Expand accordion** | L1 (0.85) | L2 ❌ | L1 ✅ (faster!) |
| 5 | **Edit button** | L2 (0.90) | L2 ⚠️ | L2 ✅ (skip L1) |
| 6 | **Type dropdown** | L1 (0.80) | L2/L3 ❌ | L1 ✅ |
| 7 | Save | L1 (0.85) | L1/L2 ⚠️ | L1 ✅ |
| 8 | Verify | L2 (0.90) | L2 ✅ | L2 ✅ (skip L1) |

**Results:**
- Faster: Skips L1 for row-scoped steps (saves 0.1s × 2 = 0.2s)
- More accurate: Predicts L1 will work for cross-module steps
- Overall: 2-3 seconds faster per test

---

## 🎯 Recommendation

### **Combine Both Approaches!**

**Week 1: Sequential Context + LLM Analysis**
1. Add sequential context (state tracking)
2. Add LLM step analyzer (level prediction)
3. Use both together

**Expected Results:**
- L1: 25% → **75%** (sequential + LLM targeting)
- Execution time: 40s → **15s** (3x faster)
- Success rate: 92% → **96%**

**Why this is powerful:**
- Sequential context enables L1 cross-module matching
- LLM decides when to use L1 vs skip to L2/L3
- Best of both worlds!

---

Ready to implement the LLM layer on top of sequential context? 🚀
