"""
InsightEnhancer Agent - Creates complete, actionable insights.
Combines all information into learning that teaches the system.
"""

import json
from typing import Dict, Any
from datetime import datetime
from .llm_client import get_azure_openai_client
import yaml
from pathlib import Path


class InsightEnhancerAgent:
    """
    Agent that synthesizes complete insights from all collected information.
    Creates insights that are complete, actionable, and generalizable.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InsightEnhancer agent.

        Args:
            config: Configuration from feedback_config.yaml
        """
        self.config = config
        self.llm_config = config.get('llm', {})

        # Load system prompt
        prompts_path = Path(__file__).parent.parent / 'prompts' / 'agent_prompts.yaml'
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.system_prompt = self._build_system_prompt(prompts['insight_enhancer'])

        # Initialize OpenAI client
        self.client, self.model = get_azure_openai_client()

    def _build_system_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """Build complete system prompt from config."""
        prompt_parts = [
            f"Role: {prompt_config['role']}",
            "",
            "Task:",
            prompt_config['task'],
            "",
            "Your output must be:",
            "1. Complete (all required fields filled)",
            "2. Actionable (includes exact code fix)",
            "3. Generalizable (extracts reusable pattern)",
            "4. Contextual (explains why in this specific case)",
            "5. Preventive (guidance for avoiding similar issues)",
            "",
            "Pattern Extraction Guidelines:",
            prompt_config['pattern_extraction_guide'],
        ]

        return "\n".join(prompt_parts)

    def enhance_insight(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        solution_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create complete insight from all collected information.

        Args:
            failure_context: Original failure context
            parser_analysis: Analysis from InputParser
            tester_answers: Answers from QuestionGenerator
            solution_analysis: Analysis from SolutionSearcher

        Returns:
            Complete insight dictionary
        """
        print(f"\n InsightEnhancer creating complete learning...")

        # Generate enhanced insight with LLM
        insight = self._enhance_with_llm(
            failure_context,
            parser_analysis,
            tester_answers,
            solution_analysis
        )

        # Add metadata
        insight['metadata'] = {
            'ticket_id': failure_context.get('ticket_id'),
            'timestamp': datetime.now().isoformat(),
            'session_id': failure_context.get('session_id'),
            'failed_step': failure_context.get('failed_step'),
            'step_name': failure_context.get('step_name')
        }

        # Calculate learning metadata
        insight['learning_metadata'] = self._calculate_learning_metadata(
            parser_analysis,
            tester_answers,
            solution_analysis
        )

        print(f"   Learning quality: {insight['learning_metadata']['reusability']}")
        print(f"   Confidence: {insight['learning_metadata']['confidence']:.2f}")

        return {
            'agent': 'insight_enhancer',
            'insight': insight
        }

    def _enhance_with_llm(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        solution_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to create enhanced insight."""

        user_prompt = self._build_enhancement_prompt(
            failure_context,
            parser_analysis,
            tester_answers,
            solution_analysis
        )

        try:
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

            insight = json.loads(response.choices[0].message.content)
            return insight

        except Exception as e:
            print(f"     LLM enhancement error: {e}")
            # Return minimal insight on error
            return self._create_fallback_insight(
                failure_context,
                parser_analysis,
                tester_answers,
                solution_analysis
            )

    def _build_enhancement_prompt(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        solution_analysis: Dict[str, Any]
    ) -> str:
        """Build comprehensive enhancement prompt."""

        prompt_parts = [
            "# Create Complete Insight",
            "",
            "## All Information Collected",
            "",
            "### Original Failure",
            f"Ticket: {failure_context.get('ticket_id')}",
            f"Step: {failure_context.get('failed_step')} - {failure_context.get('step_name')}",
            f"Error: {failure_context.get('error_message')}",
            "",
            "### InputParser Analysis",
            f"Root Cause: {parser_analysis.get('root_cause')}",
            f"Explanation: {parser_analysis.get('root_cause_explanation')}",
            f"Element Type: {parser_analysis.get('element_details', {}).get('element_type')}",
            f"Old Selector: {parser_analysis.get('element_details', {}).get('old_selector')}",
            f"Cascading Failure: {parser_analysis.get('cascading_failure')}",
            "",
            "### Tester Input"
        ]

        if tester_answers:
            for key, value in tester_answers.items():
                if value != 'skipped':
                    prompt_parts.append(f"{key}: {value}")
        else:
            prompt_parts.append("(No tester input needed - high confidence auto-analysis)")

        prompt_parts.append("")

        # Add solution recommendations
        if solution_analysis.get('recommended_solution'):
            rec_sol = solution_analysis['recommended_solution']
            prompt_parts.extend([
                "### Recommended Solution",
                f"Approach: {rec_sol.get('approach')}",
                f"Code: {rec_sol.get('code')}",
                f"Based on: {rec_sol.get('based_on')}",
                f"Confidence: {rec_sol.get('confidence')}",
                ""
            ])

        # Add patterns found
        if solution_analysis.get('extracted_patterns'):
            prompt_parts.extend([
                "### Patterns from Similar Cases",
                ""
            ])
            for pattern in solution_analysis['extracted_patterns']:
                prompt_parts.extend([
                    f"Pattern: {pattern.get('pattern_name')}",
                    f"  Occurrences: {pattern.get('occurrences')}",
                    f"  Description: {pattern.get('pattern_description')}",
                    ""
                ])

        # Insight structure instructions
        prompt_parts.extend([
            "## Your Task",
            "",
            "Create a complete, actionable insight in JSON format:",
            "",
            "{",
            '  "failure_analysis": {',
            '    "symptom": "What the tester observed",',
            '    "root_cause": "Specific technical reason (from parser + tester)",',
            '    "root_cause_detailed": "Full explanation combining all sources",',
            '    "evidence": ["tool1 showed X", "tester confirmed Y", "similar case had Z"],',
            '    "sequential_context": "Relevant info from previous steps",',
            '    "cascading_failure": true/false',
            '  },',
            '  "solution": {',
            '    "description": "Plain English what to do",',
            '    "old_code": "Exact code that failed",',
            '    "new_code": "Exact fixed code (use tester input + recommendations)",',
            '    "code_explanation": "Why this fix works",',
            '    "requirements": ["Any prerequisites for this fix"]',
            '  },',
            '  "generalized_pattern": {',
            '    "pattern_name": "Descriptive name for vector DB",',
            '    "applies_to": "When to use this pattern (be specific)",',
            '    "code_template": "Generalized code with {placeholders}",',
            '    "examples": ["Example scenario 1", "Example scenario 2"],',
            '    "success_indicators": "How to know if pattern applies"',
            '  },',
            '  "prevention": {',
            '    "guideline": "How to avoid this in future tests",',
            '    "detection": "How to spot this issue early",',
            '    "best_practice": "Recommended approach going forward"',
            '  },',
            '  "relationships": {',
            '    "similar_tickets": ["Ticket IDs from similar cases"],',
            '    "related_patterns": ["Pattern names if building on existing"],',
            '    "depends_on_steps": [previous step numbers if relevant]',
            '  }',
            '}',
            "",
            "CRITICAL:",
            "- Use EXACT code from tester input or recommended solution",
            "- Extract generalizable pattern (not just ticket-specific fix)",
            "- Explain WHY fix works, not just WHAT changed",
            "- Prevention should help avoid this whole class of failures"
        ])

        return "\n".join(prompt_parts)

    def _create_fallback_insight(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        solution_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create basic insight when LLM fails."""

        # Extract code from solution or tester
        new_code = None
        if solution_analysis.get('recommended_solution', {}).get('code'):
            new_code = solution_analysis['recommended_solution']['code']
        elif tester_answers:
            for key, value in tester_answers.items():
                if 'code' in key.lower() or 'selector' in key.lower():
                    new_code = value
                    break

        return {
            'failure_analysis': {
                'symptom': failure_context.get('error_message'),
                'root_cause': parser_analysis.get('root_cause', 'unknown'),
                'root_cause_detailed': parser_analysis.get('root_cause_explanation', 'Details unavailable'),
                'evidence': parser_analysis.get('evidence_used', []),
                'sequential_context': '',
                'cascading_failure': parser_analysis.get('cascading_failure', False)
            },
            'solution': {
                'description': 'Fix based on tester input',
                'old_code': parser_analysis.get('element_details', {}).get('old_selector', ''),
                'new_code': new_code or 'See tester answers',
                'code_explanation': 'Manual fix provided by tester',
                'requirements': []
            },
            'generalized_pattern': {
                'pattern_name': f"{parser_analysis.get('root_cause', 'unknown')}_fix",
                'applies_to': 'Similar failures',
                'code_template': new_code or '',
                'examples': [],
                'success_indicators': ''
            },
            'prevention': {
                'guideline': 'Review similar patterns',
                'detection': 'Check error message',
                'best_practice': 'Use stable selectors'
            },
            'relationships': {
                'similar_tickets': [],
                'related_patterns': [],
                'depends_on_steps': []
            }
        }

    def _calculate_learning_metadata(
        self,
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        solution_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metadata about learning quality."""

        # Tester effort
        if not tester_answers or all(v == 'skipped' for v in tester_answers.values()):
            tester_effort = 'low'  # Auto-resolved
        elif len(tester_answers) <= 2:
            tester_effort = 'medium'
        else:
            tester_effort = 'high'

        # Reusability based on patterns found
        pattern_count = len(solution_analysis.get('extracted_patterns', []))
        similar_count = len(solution_analysis.get('similar_cases', []))

        if pattern_count >= 2 or similar_count >= 3:
            reusability = 'high'
        elif pattern_count >= 1 or similar_count >= 1:
            reusability = 'medium'
        else:
            reusability = 'low'

        # Overall confidence (average of parser and solution confidence)
        parser_conf = parser_analysis.get('confidence', 0.5)
        solution_conf = solution_analysis.get('confidence', 0.5)
        overall_conf = (parser_conf + solution_conf) / 2

        # Priority for embedding
        if overall_conf >= 0.8 and reusability == 'high':
            priority = 'high'
        elif overall_conf >= 0.6:
            priority = 'medium'
        else:
            priority = 'low'

        return {
            'confidence': overall_conf,
            'tester_effort': tester_effort,
            'reusability': reusability,
            'priority_for_embedding': priority,
            'auto_resolved': tester_effort == 'low'
        }


def create_insight_enhancer_agent(config: Dict[str, Any]) -> InsightEnhancerAgent:
    """Factory function to create InsightEnhancer agent."""
    return InsightEnhancerAgent(config)
