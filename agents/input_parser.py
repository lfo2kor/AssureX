"""
InputParser Agent - Analyzes test failures using tools and LLM reasoning.
Extracts structured information and identifies missing data.
"""

import json
from typing import Dict, Any, List
import yaml
from pathlib import Path
from .agent_tools import AgentTools
from .llm_client import get_azure_openai_client


class InputParserAgent:
    """
    Agent that analyzes test failures by combining multiple evidence sources.
    Uses tools to gather facts, then LLM to reason about root cause.
    """

    def __init__(self, config: Dict[str, Any], tools: AgentTools):
        """
        Initialize InputParser agent.

        Args:
            config: Configuration from feedback_config.yaml
            tools: Shared agent tools instance
        """
        self.config = config
        self.tools = tools
        self.llm_config = config.get('llm', {})

        # Load system prompt
        prompts_path = Path(__file__).parent.parent / 'prompts' / 'agent_prompts.yaml'
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.system_prompt = self._build_system_prompt(prompts['input_parser'])

        # Initialize Azure OpenAI client
        self.client, self.model = get_azure_openai_client()

    def _build_system_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """Build complete system prompt from config."""
        prompt_parts = [
            f"Role: {prompt_config['role']}",
            "",
            "Task:",
            prompt_config['task'],
            "",
            "Available Tools:",
        ]

        for tool in prompt_config['tools_available']:
            prompt_parts.append(f"- {tool['name']}: {tool['purpose']}")
            prompt_parts.append(f"  When to use: {tool['when_to_use']}")

        prompt_parts.extend([
            "",
            "Reasoning Process:",
            prompt_config['reasoning_template']
        ])

        return "\n".join(prompt_parts)

    def analyze_failure(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test failure using tools and LLM reasoning.

        Args:
            failure_context: Dictionary containing:
                - ticket_id: Ticket identifier
                - failed_step: Step number that failed
                - step_name: Description of the step
                - error_message: Error from test execution
                - screenshot_path: Path to failure screenshot
                - html_snapshot_path: Path to HTML snapshot
                - test_file_path: Path to generated test file
                - session_data: Sequential execution context

        Returns:
            Dictionary with structured analysis results
        """
        print(f"\n InputParser analyzing failure at step {failure_context.get('failed_step')}...")

        # Step 1: Gather evidence using tools
        evidence = self._gather_evidence(failure_context)

        # Step 2: LLM reasoning with evidence
        analysis = self._analyze_with_llm(failure_context, evidence)

        # Step 3: Score confidence and identify gaps
        result = self._finalize_analysis(analysis, evidence)

        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Missing fields: {len(result['missing_fields'])}")

        return result

    def _gather_evidence(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use tools to gather evidence about the failure."""
        evidence = {}

        # Tool 1: Read screenshot
        if failure_context.get('screenshot_path'):
            print("    Analyzing screenshot...")
            evidence['screenshot'] = self.tools.read_screenshot(
                failure_context['screenshot_path']
            )

        # Tool 2: Parse HTML snapshot
        if failure_context.get('html_snapshot_path'):
            print("    Parsing HTML snapshot...")
            # Extract failed selector from error message if present
            target_selector = self._extract_selector_from_error(
                failure_context.get('error_message', '')
            )
            evidence['html'] = self.tools.parse_html_snapshot(
                failure_context['html_snapshot_path'],
                target_selector=target_selector
            )

        # Tool 3: Read test code
        if failure_context.get('test_file_path'):
            print("    Reading test code...")
            evidence['test_code'] = self.tools.read_test_code(
                failure_context['test_file_path'],
                line_number=failure_context.get('failed_line_number')
            )

        # Tool 4: Search vector DB for similar failures
        print("    Searching for similar past failures...")
        search_query = self._build_search_query(failure_context)
        evidence['vector_search'] = self.tools.search_vector_db(
            query=search_query,
            top_k=self.config.get('vector_db', {}).get('search_params', {}).get('top_k', 5)
        )

        # Tool 5: Analyze sequential context
        if failure_context.get('session_data'):
            print("    Analyzing sequential context...")
            evidence['sequential_context'] = self.tools.analyze_sequential_context(
                failure_context['session_data'],
                failure_context.get('failed_step')
            )

        return evidence

    def _extract_selector_from_error(self, error_message: str) -> str:
        """Extract selector from error message if present."""
        # Common patterns in Playwright errors
        patterns = [
            'Selector "',
            'selector \'',
            'waiting for selector "',
            'locator("'
        ]

        for pattern in patterns:
            if pattern in error_message.lower():
                start = error_message.lower().find(pattern) + len(pattern)
                end = error_message.find('"', start)
                if end == -1:
                    end = error_message.find("'", start)
                if end > start:
                    return error_message[start:end]

        return None

    def _build_search_query(self, failure_context: Dict[str, Any]) -> str:
        """Build semantic search query from failure context."""
        parts = []

        # Add step name
        if failure_context.get('step_name'):
            parts.append(failure_context['step_name'])

        # Add error type
        error = failure_context.get('error_message', '')
        if 'timeout' in error.lower():
            parts.append('timeout')
        if 'not found' in error.lower() or 'not visible' in error.lower():
            parts.append('element not found')
        if 'click' in error.lower():
            parts.append('click interaction')

        return ' '.join(parts) if parts else 'test failure'

    def _analyze_with_llm(self, failure_context: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to reason about failure with gathered evidence."""

        # Build analysis prompt
        user_prompt = self._build_analysis_prompt(failure_context, evidence)

        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_config.get('temperature', 0.1),
                max_tokens=self.llm_config.get('max_tokens', 2000),
                response_format={"type": "json_object"}
            )

            # Parse response
            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            print(f"     LLM analysis error: {e}")
            # Return minimal analysis on error
            return {
                'root_cause': 'unknown',
                'confidence': 0.1,
                'error': str(e)
            }

    def _build_analysis_prompt(self, failure_context: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Build comprehensive prompt with failure context and evidence."""

        prompt_parts = [
            "# Test Failure Analysis",
            "",
            "## Failure Context",
            f"Ticket: {failure_context.get('ticket_id')}",
            f"Failed Step: {failure_context.get('failed_step')} - {failure_context.get('step_name')}",
            f"Error: {failure_context.get('error_message')}",
            ""
        ]

        # Add evidence from tools
        if evidence.get('screenshot', {}).get('success'):
            prompt_parts.extend([
                "## Screenshot Evidence",
                f"Screenshot captured: {evidence['screenshot'].get('description')}",
                "Note: Screenshot shows visual state at failure moment",
                ""
            ])

        if evidence.get('html', {}).get('success'):
            html_ev = evidence['html']
            prompt_parts.extend([
                "## HTML Snapshot Evidence",
                f"Total elements in DOM: {html_ev.get('total_elements')}",
                f"Interactive elements found: {len(html_ev.get('interactive_elements', []))}",
                ""
            ])

            # Show interactive elements
            if html_ev.get('interactive_elements'):
                prompt_parts.append("Key interactive elements:")
                for elem in html_ev['interactive_elements'][:10]:
                    elem_desc = f"  - <{elem['tag']}>"
                    if elem.get('id'):
                        elem_desc += f" id='{elem['id']}'"
                    if elem.get('data-testid'):
                        elem_desc += f" data-testid='{elem['data-testid']}'"
                    if elem.get('text'):
                        elem_desc += f" text='{elem['text'][:30]}'"
                    prompt_parts.append(elem_desc)
                prompt_parts.append("")

            # Show target selector analysis if present
            if html_ev.get('target_selector_analysis'):
                tsa = html_ev['target_selector_analysis']
                prompt_parts.extend([
                    f"Target selector '{tsa['selector']}' analysis:",
                    f"  Found: {tsa['found']}",
                ])
                if tsa.get('suggestions'):
                    prompt_parts.append("  Suggestions:")
                    for sugg in tsa['suggestions']:
                        prompt_parts.append(f"    - {sugg['type']}: {sugg['values'][:3]}")
                prompt_parts.append("")

        if evidence.get('test_code', {}).get('success'):
            code_ev = evidence['test_code']
            if code_ev.get('focused_line'):
                prompt_parts.extend([
                    "## Test Code Context",
                    f"Failed line {code_ev['focused_line']['line_number']}: {code_ev['focused_line']['line_content']}",
                    "",
                    "Context:",
                    "```python",
                    code_ev['focused_line']['context'],
                    "```",
                    ""
                ])

        if evidence.get('vector_search', {}).get('success'):
            vs_ev = evidence['vector_search']
            if vs_ev.get('results'):
                prompt_parts.extend([
                    f"## Similar Past Failures ({vs_ev['results_count']} found)",
                    ""
                ])
                for i, result in enumerate(vs_ev['results'][:3], 1):
                    prompt_parts.extend([
                        f"### Similar Case {i} (similarity: {result.get('similarity', 0):.2f})",
                        f"{result.get('content', '')[:200]}...",
                        ""
                    ])

        if evidence.get('sequential_context', {}).get('success'):
            seq_ev = evidence['sequential_context']
            patterns = seq_ev.get('patterns', {})
            prompt_parts.extend([
                "## Sequential Context",
                f"Consecutive passes before failure: {patterns.get('consecutive_passes_before_failure', 0)}",
                f"Any previous failures: {patterns.get('any_previous_failures', False)}",
                f"Last action before failure: {patterns.get('last_action_before_failure')}",
                ""
            ])

        # Add analysis instructions
        prompt_parts.extend([
            "## Your Task",
            "",
            "Based on the evidence above, provide a JSON analysis with:",
            "",
            "{",
            '  "root_cause": "Specific technical reason for failure (e.g., selector_changed, timing_issue, element_hidden)",',
            '  "root_cause_explanation": "Detailed explanation of why it failed",',
            '  "element_details": {',
            '    "element_type": "Type of UI element (button, dropdown, input, etc.)",',
            '    "old_selector": "Selector that failed (if applicable)",',
            '    "suggested_new_selector": "Potential new selector from HTML evidence (if found)",',
            '    "selector_type": "Type of selector (id, class, data-testid, text, xpath)"',
            '  },',
            '  "evidence_used": ["List of which tools provided key information"],',
            '  "confidence": 0.0-1.0,',
            '  "confidence_reasoning": "Why this confidence level",',
            '  "missing_fields": ["List of information still needed from tester"],',
            '  "cascading_failure": true/false,',
            '  "similar_to_past_cases": true/false',
            "}",
            "",
            "Use the evidence to be as specific as possible. If evidence is insufficient, list what's missing.",
            "Confidence guidelines:",
            "- 0.9-1.0: All tools confirm, exact match in vector DB",
            "- 0.7-0.9: Strong evidence from multiple tools",
            "- 0.5-0.7: Some evidence, need clarification",
            "- 0.3-0.5: Multiple possibilities, need tester choice",
            "- 0.0-0.3: Insufficient evidence"
        ])

        return "\n".join(prompt_parts)

    def _finalize_analysis(self, llm_analysis: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize analysis with validation and metadata."""

        result = {
            'agent': 'input_parser',
            'success': True,
            'timestamp': Path(__file__).stat().st_mtime,  # Quick timestamp

            # Core analysis from LLM
            'root_cause': llm_analysis.get('root_cause', 'unknown'),
            'root_cause_explanation': llm_analysis.get('root_cause_explanation', ''),
            'element_details': llm_analysis.get('element_details', {}),

            # Confidence and gaps
            'confidence': llm_analysis.get('confidence', 0.5),
            'confidence_reasoning': llm_analysis.get('confidence_reasoning', ''),
            'missing_fields': llm_analysis.get('missing_fields', []),

            # Context
            'evidence_used': llm_analysis.get('evidence_used', []),
            'cascading_failure': llm_analysis.get('cascading_failure', False),
            'similar_to_past_cases': llm_analysis.get('similar_to_past_cases', False),

            # Raw evidence for next agents
            'evidence': evidence,

            # Raw LLM output
            'raw_llm_output': llm_analysis
        }

        # Determine next action based on confidence and missing fields
        confidence_threshold = self.config.get('confidence_thresholds', {}).get(
            'ask_questions_threshold', 0.5
        )

        if result['confidence'] >= confidence_threshold and not result['missing_fields']:
            result['next_action'] = 'proceed_to_solution_search'
        else:
            result['next_action'] = 'ask_questions'

        return result


def create_input_parser_agent(config: Dict[str, Any], tools: AgentTools) -> InputParserAgent:
    """Factory function to create InputParser agent."""
    return InputParserAgent(config, tools)
