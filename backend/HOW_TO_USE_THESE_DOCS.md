# How to Use These Specification Documents

## Overview

You now have **two comprehensive specification documents** that fully describe your AI-powered vision-based test automation PoC:

1. **`spec.md`** - Complete Technical Specification
2. **`constitution.md`** - Implementation Guidelines and Patterns

## What's in These Documents

### 📄 `spec.md` - The "WHAT"

This document describes **WHAT** your system does:

**Key Sections:**
- **Section 1-2:** Executive summary and technology stack
- **Section 3:** Complete system architecture with all 3 agents
- **Section 4:** State schema (shared memory between agents)
- **Section 5:** Input requirements (config file, Jira tickets, user input)
- **Section 6-7:** GPT-4o Vision integration and retry logic
- **Section 8:** File organization
- **Section 9:** Output specifications (HTML reports, videos, scripts)
- **Section 11:** Current implementation status (what's done vs not done)
- **Section 12:** Extension opportunities - **8 NEW AGENT IDEAS**

**Current Agents Documented:**
1. **Jira Parser Agent** - Parses Jira tickets to extract test steps
2. **Vision Executor Agent** - Uses GPT-4o vision to execute tests
3. **Report Generator Agent** - Generates HTML reports and Playwright scripts

**New Agent Suggestions (Section 12.1):**
1. Pre-Execution Validator Agent
2. Post-Execution Analyzer Agent
3. Test Data Generator Agent
4. Screenshot Comparator Agent
5. Natural Language Reporter Agent
6. Notification Agent
7. Ticket Validator Agent
8. Cross-Test Learning Agent

---

### 📋 `constitution.md` - The "HOW"

This document describes **HOW** to implement the system:

**Key Sections:**
- **Sections 1-14:** Core principles (code organization, agent design, error handling, etc.)
- **Implementation Guidelines:** Step-by-step checklist for adding new agents
- **Must NOT Do:** Anti-patterns to avoid (with examples)
- **Must Do:** Required practices (with examples)
- **Success Criteria:** Code quality checklist

**Use this document when:**
- Adding new agents
- Modifying existing agents
- Making architectural decisions
- Ensuring code quality

---

## How to Use These Documents in Claude.ai

### Option 1: Add New Agents (Recommended)

**Steps:**

1. **Upload Both Documents to Claude.ai**
   - Upload `spec.md`
   - Upload `constitution.md`

2. **Ask Claude to Add a New Agent**

   Example prompts:

   ```
   "Based on spec.md and constitution.md, implement a Post-Execution Analyzer Agent
   that compares results against acceptance criteria using GPT-4o. Follow all the
   patterns in constitution.md."
   ```

   ```
   "Add a Notification Agent that sends Slack messages when tests complete.
   Follow the agent template in constitution.md section 'When Adding a New Agent'."
   ```

   ```
   "Implement a Validation Agent that validates execution results. Use the example
   in constitution.md as a template."
   ```

3. **Claude will:**
   - Create the new agent file following the patterns
   - Add proper error handling
   - Add logging
   - Update the workflow
   - Provide integration instructions

---

### Option 2: Modify Existing System

**Example prompts:**

```
"Based on spec.md, add conditional routing to retry failed executions.
See constitution.md section on 'Modifying Workflow' for examples."
```

```
"Enhance the Jira Parser Agent to handle multiple ticket formats.
Follow the error handling patterns in constitution.md."
```

---

### Option 3: Understand the Current System

**Example prompts:**

```
"Explain how the Vision Executor Agent works based on spec.md section 3."
```

```
"What are the current limitations of the PoC according to spec.md section 16?"
```

```
"Show me the state schema and explain how data flows between agents."
```

---

## Quick Reference Guide

### Current System Capabilities (from spec.md Section 11)

✅ **Implemented:**
- Multi-agent architecture with LangGraph
- GPT-4o vision-based execution
- Playwright browser automation
- HTML report generation
- Video recording
- 3-level retry logic
- Error handling

❌ **Not Implemented (Future):**
- Multiple simultaneous tests
- Cloud deployment
- Multi-browser support
- CI/CD integration
- Test scheduling

---

### Agent Template (from constitution.md)

When adding a new agent, follow this structure:

```python
"""
Agent Name

Description: What this agent does
"""

import logging
from typing import Dict
from models.state import TestAutomationState


def agent_name(state: TestAutomationState) -> TestAutomationState:
    """
    Brief description of agent functionality.

    Args:
        state: Current workflow state containing required inputs

    Returns:
        Updated state with new data added

    Raises:
        ValueError: If required input is missing
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("=" * 70)
    logger.info("AGENT NAME - Starting")
    logger.info("=" * 70)

    try:
        # 1. Validate inputs
        if 'required_field' not in state:
            raise ValueError("Missing required_field in state")

        # 2. Read from state
        config = state['config']
        required_data = state['required_field']

        # 3. Process
        result = process_data(required_data, config, logger)

        # 4. Update state
        state['agent_result'] = result

        logger.info("AGENT NAME - Complete")
        logger.info("=" * 70)

        return state

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        state['errors'].append({
            'agent': 'agent_name',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        raise
```

---

## Suggested Next Steps

### 1. Understand Current System
- Read `spec.md` sections 1-3 (architecture overview)
- Review current agents in section 3

### 2. Choose Enhancements
- Review section 12.1 in `spec.md` for agent ideas
- Pick 1-2 agents to implement first

### 3. Use Claude.ai to Implement
- Upload both documents
- Use prompts like the examples above
- Claude will follow all the patterns automatically

### 4. Iterative Development
- Add one agent at a time
- Test each agent before adding the next
- Update documentation as you go

---

## File Locations

```
C:\Projects\AI_Chat\PLCD\TA_AI_Project\
├── spec.md                    ← Upload this to Claude.ai
├── constitution.md            ← Upload this to Claude.ai
├── HOW_TO_USE_THESE_DOCS.md  ← You are here
│
├── run_test.py               ← Main entry point
├── agents/
│   ├── jira_parser_agent.py
│   ├── vision_executor_agent.py
│   └── report_generator_agent.py
├── workflows/
│   └── test_workflow.py
└── ... (other files)
```

---

## Example: Adding a Notification Agent

**Step 1:** Upload `spec.md` and `constitution.md` to Claude.ai

**Step 2:** Prompt Claude:

```
Based on spec.md and constitution.md, create a Notification Agent that:
1. Reads execution results from state
2. Sends a Slack message with test summary
3. Includes pass/fail status, execution time, and link to HTML report
4. Uses Slack webhook URL from config
5. Follows all patterns in constitution.md

Create the agent file and show me how to integrate it into the workflow.
```

**Step 3:** Claude will generate:
- `agents/notification_agent.py` with proper error handling
- Instructions to add to `workflows/test_workflow.py`
- Config updates for `plcdtest_config.yaml`

---

## Tips for Success

### ✅ Do This:
- Always upload BOTH documents to Claude.ai
- Reference specific sections when asking questions
- Ask Claude to follow constitution.md patterns explicitly
- Start with simple agents, then add complexity

### ❌ Avoid This:
- Uploading only one document (need both for full context)
- Asking for features marked "Out of Scope" in spec.md section 11
- Ignoring the patterns in constitution.md
- Adding multiple complex agents at once

---

## Common Prompts for Claude.ai

### Understanding
```
"Explain the Vision Executor Agent architecture from spec.md section 3.2"
"What is the state schema and how does it work?"
"Show me all the inputs required to run a test"
```

### Implementation
```
"Add a [AGENT_NAME] agent following the template in constitution.md"
"Enhance the retry logic to support 5 attempts instead of 3"
"Add parallel execution for independent test steps"
```

### Debugging
```
"The Jira Parser is failing. Show me how to add better error handling based on constitution.md section 5"
"How should I handle API rate limits according to the constitution?"
```

### Extension
```
"Implement the Post-Execution Analyzer Agent from spec.md section 12.1"
"Add conditional routing to the workflow based on test complexity"
"Create a Test Data Generator Agent with dynamic data support"
```

---

## Summary

You now have:

1. ✅ **Complete specification** of your PoC (`spec.md`)
2. ✅ **Implementation guidelines** (`constitution.md`)
3. ✅ **8 agent ideas** for extensions
4. ✅ **Agent template** to follow
5. ✅ **Clear patterns** for code quality

**Next step:** Upload both documents to claude.ai and start adding agents!

---

**Questions?** Both documents are comprehensive and self-contained. Everything you need to extend the system is documented.

**Good luck with your enhancements!** 🚀
