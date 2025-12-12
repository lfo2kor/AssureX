"""
FeedbackParserAgent - Natural Language Feedback Parser

Parses tester's natural language feedback and extracts structured insights.
Uses GPT-4o to understand feedback like:
"Step 5 is a false positive, should use [data-editnodebtn='default_UUT_01']"

Author: AI Testing Assistant
"""

import json
import re
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)


class FeedbackParserAgent:
    """
    Agent that parses natural language feedback from testers into structured JSON.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FeedbackParserAgent.

        Args:
            config: Configuration dictionary with Azure OpenAI settings
        """
        self.config = config
        azure_config = config.get('azure_openai', {})

        self.client = AzureOpenAI(
            api_key=azure_config.get('api_key'),
            api_version=azure_config.get('api_version', '2024-02-15-preview'),
            azure_endpoint=azure_config.get('endpoint')
        )

        self.model = azure_config.get('deployment_gpt4o', 'gpt-4o')
        logger.info(f"feedback_parser initialized (model: {self.model})")

    def parse_feedback(
        self,
        feedback_text: str,
        test_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse natural language feedback into structured format.

        Args:
            feedback_text: Natural language feedback from tester
            test_context: Context from test execution (steps, results, etc.)

        Returns:
            Structured feedback dictionary with step corrections
        """
        if not feedback_text or feedback_text.strip() == '':
            return {}

        logger.info("Parsing natural language feedback...")

        # Build context for the LLM
        test_steps = test_context.get('steps', [])
        test_results = test_context.get('results', {})

        steps_summary = self._build_steps_summary(test_steps, test_results)

        # Create the prompt for GPT-4o
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(feedback_text, steps_summary)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            parsed_feedback = json.loads(response.choices[0].message.content)
            logger.info(f"✓ Parsed feedback for {len(parsed_feedback.get('steps', {}))} steps")

            return parsed_feedback

        except Exception as e:
            logger.error(f"Failed to parse feedback: {e}")
            # Fallback to manual extraction if LLM fails
            return self._fallback_parse(feedback_text, test_steps)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for feedback parsing."""
        return """You are an expert test feedback analyzer. Extract structured feedback from tester's natural language input.

Your job is to identify:
1. Which step number the feedback refers to
2. Type of feedback: false_positive, failed, timing_issue, true_positive
3. Correct selector (if provided)
4. Expected behavior (for negative tests)
5. Any additional context or reasoning

Output ONLY valid JSON in this exact format:
{
  "steps": {
    "step_5": {
      "feedback_type": "false_positive",
      "correct_selector": "[data-editnodebtn='default_UUT_01']",
      "issue_description": "clicked wrong button",
      "reasoning": "should click edit button not the node itself"
    },
    "step_6": {
      "feedback_type": "failed",
      "correct_selector": "[data-checkuniquename='attribut1']",
      "issue_description": "selector not found"
    }
  },
  "test_type": "positive",
  "expected_outcome": "test should pass",
  "summary": "Steps 5 and 6 need selector corrections"
}

Valid feedback_type values:
- "false_positive": Test passed but used wrong selector/clicked wrong element
- "failed": Test failed, providing correct selector
- "timing_issue": Correct selector but needs wait/retry
- "true_positive": Test result is correct, no changes needed
- "application_bug": Test correctly identified a bug in the application

If tester doesn't mention a step, don't include it in output.
If no specific feedback, return: {"steps": {}, "summary": "No specific feedback provided"}
"""

    def _build_user_prompt(self, feedback_text: str, steps_summary: str) -> str:
        """Build the user prompt with feedback and context."""
        return f"""Test Execution Context:
{steps_summary}

Tester's Feedback:
{feedback_text}

Parse the feedback above and extract structured information as JSON."""

    def _build_steps_summary(
        self,
        test_steps: List[Dict[str, Any]],
        test_results: Dict[str, Any]
    ) -> str:
        """Build a summary of test steps for context."""
        summary = []

        for i, step in enumerate(test_steps, 1):
            step_text = step.get('text', step.get('step', ''))
            status = "UNKNOWN"

            # Try to get status from results
            step_results = test_results.get('step_results', [])
            if i <= len(step_results):
                status = step_results[i-1].get('status', 'UNKNOWN')

            summary.append(f"Step {i}: [{status}] {step_text}")

        return "\n".join(summary)

    def _fallback_parse(
        self,
        feedback_text: str,
        test_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fallback parser using regex if LLM fails.
        Simple extraction of step numbers and selectors.
        """
        logger.warning("Using fallback regex-based parser")

        result = {"steps": {}, "summary": "Parsed with fallback method"}

        # Pattern: "step 5" or "Step 5:" or "step-5"
        step_pattern = r'step[\s\-:]*(\d+)'
        # Pattern: [data-something='value'] or [data-something="value"]
        selector_pattern = r'\[([^\]]+)\]'

        lines = feedback_text.lower().split('\n')

        for line in lines:
            # Find step numbers
            step_matches = re.findall(step_pattern, line, re.IGNORECASE)
            # Find selectors
            selector_matches = re.findall(selector_pattern, line)

            if step_matches and selector_matches:
                step_num = int(step_matches[0])
                selector = f"[{selector_matches[0]}]"

                # Determine feedback type from keywords
                feedback_type = "failed"
                if "false positive" in line or "wrong" in line:
                    feedback_type = "false_positive"
                elif "timing" in line or "wait" in line:
                    feedback_type = "timing_issue"

                result["steps"][f"step_{step_num}"] = {
                    "feedback_type": feedback_type,
                    "correct_selector": selector,
                    "issue_description": line.strip()
                }

        return result

    def validate_parsed_feedback(
        self,
        parsed_feedback: Dict[str, Any],
        test_context: Dict[str, Any]
    ) -> bool:
        """
        Validate that parsed feedback makes sense.

        Args:
            parsed_feedback: Parsed feedback dictionary
            test_context: Original test context

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(parsed_feedback, dict):
            return False

        if 'steps' not in parsed_feedback:
            return False

        steps = parsed_feedback.get('steps', {})
        total_steps = len(test_context.get('steps', []))

        # Check that step numbers are valid
        for step_key in steps.keys():
            try:
                step_num = int(step_key.replace('step_', ''))
                if step_num < 1 or step_num > total_steps:
                    logger.warning(f"Invalid step number: {step_num}")
                    return False
            except ValueError:
                logger.warning(f"Invalid step key format: {step_key}")
                return False

        return True


if __name__ == "__main__":
    # Test the parser
    import yaml

    with open('plcdtest_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    parser = FeedbackParserAgent(config)

    test_feedback = """
    Step 5 is a false positive. It used [data-editnode='default_testobject_01']
    but should use [data-editnodebtn='default_UUT_01'].
    Step 6 failed because the correct selector is [data-checkuniquename='attribut1'].
    """

    test_context = {
        'steps': [
            {'text': 'Step 1'},
            {'text': 'Step 2'},
            {'text': 'Step 3'},
            {'text': 'Step 4'},
            {'text': 'Click edit button for test object default_UUT_01'},
            {'text': 'Click on attribut1 field'}
        ],
        'results': {
            'step_results': [
                {'status': 'PASSED'},
                {'status': 'PASSED'},
                {'status': 'PASSED'},
                {'status': 'PASSED'},
                {'status': 'PASSED'},
                {'status': 'FAILED'}
            ]
        }
    }

    result = parser.parse_feedback(test_feedback, test_context)
    print(json.dumps(result, indent=2))
