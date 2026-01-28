"""
QualityValidator Agent - Validates insights for completeness and quality.
Ensures insights are ready for saving and future reuse.
"""

import json
import ast
from typing import Dict, Any, List, Tuple
from .llm_client import get_azure_openai_client
import yaml
from pathlib import Path


class QualityValidatorAgent:
    """
    Agent that validates insights before they're saved.
    Checks completeness, accuracy, and actionability.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize QualityValidator agent.

        Args:
            config: Configuration from feedback_config.yaml
        """
        self.config = config
        self.llm_config = config.get('llm', {})

        # Load system prompt and validation criteria
        prompts_path = Path(__file__).parent.parent / 'prompts' / 'agent_prompts.yaml'
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.system_prompt = self._build_system_prompt(prompts['quality_validator'])
            self.validation_criteria = prompts['quality_validator']['validation_criteria']

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
            "Decision Matrix:",
            "- All checks pass + confidence >= 0.8: AUTO-APPROVE",
            "- Minor issues + confidence 0.6-0.8: APPROVE with flags",
            "- Significant gaps + confidence 0.4-0.6: REQUEST more info",
            "- Major issues + confidence < 0.4: REJECT, restart feedback loop"
        ]

        return "\n".join(prompt_parts)

    def validate_insight(self, enhanced_insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate insight for quality and completeness.

        Args:
            enhanced_insight: Insight from InsightEnhancer

        Returns:
            Validation results with decision
        """
        print(f"\n QualityValidator checking insight quality...")

        # Run all validation checks
        validation_results = {
            'completeness': self._check_completeness(enhanced_insight),
            'code_quality': self._check_code_quality(enhanced_insight),
            'specificity': self._check_specificity(enhanced_insight),
            'evidence_support': self._check_evidence_support(enhanced_insight),
            'generalizability': self._check_generalizability(enhanced_insight)
        }

        # Calculate overall confidence
        confidence_score = self._calculate_confidence(validation_results, enhanced_insight)

        # Make decision
        decision, flags = self._make_decision(validation_results, confidence_score)

        # Use LLM for additional quality assessment
        llm_assessment = self._assess_with_llm(enhanced_insight, validation_results)

        print(f"   Decision: {decision}")
        print(f"   Confidence: {confidence_score:.2f}")
        if flags:
            print(f"   Flags: {len(flags)}")

        return {
            'agent': 'quality_validator',
            'validated': decision in ['approve', 'approve_with_flags'],
            'decision': decision,
            'confidence_score': confidence_score,
            'validation_results': validation_results,
            'flags': flags,
            'llm_assessment': llm_assessment,
            'action_required': self._get_action_required(decision),
            'improvement_suggestions': llm_assessment.get('suggestions', [])
        }

    def _check_completeness(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all required fields are present and non-empty."""
        required_fields = self.validation_criteria['completeness']['required_fields']

        missing_fields = []
        for field_path in required_fields:
            if not self._get_nested_value(insight['insight'], field_path):
                missing_fields.append(field_path)

        passed = len(missing_fields) == 0

        return {
            'passed': passed,
            'missing_fields': missing_fields,
            'message': 'All required fields present' if passed else f'Missing {len(missing_fields)} fields'
        }

    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        value = obj
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _check_code_quality(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Check if code is syntactically valid and executable."""
        issues = []

        solution = insight['insight'].get('solution', {})
        new_code = solution.get('new_code', '')

        if not new_code or new_code in ['', 'See tester answers', 'N/A']:
            issues.append('No executable code provided')
            return {
                'passed': False,
                'issues': issues,
                'message': 'No valid code to validate'
            }

        # Check for placeholder values
        placeholders = ['XXX', 'TODO', '...', 'FIXME', 'placeholder']
        for placeholder in placeholders:
            if placeholder in new_code:
                issues.append(f'Contains placeholder: {placeholder}')

        # Try to validate Python syntax (if it looks like Python code)
        if 'page.' in new_code or 'await ' in new_code or 'def ' in new_code:
            syntax_valid, syntax_error = self._validate_python_syntax(new_code)
            if not syntax_valid:
                issues.append(f'Syntax error: {syntax_error}')

        # Check for common Playwright patterns
        if 'page.' in new_code:
            valid_patterns = ['click', 'fill', 'get_by', 'locator', 'wait_for', 'text_content']
            has_valid_pattern = any(pattern in new_code for pattern in valid_patterns)
            if not has_valid_pattern:
                issues.append('Code doesn\'t use recognized Playwright patterns')

        passed = len(issues) == 0

        return {
            'passed': passed,
            'issues': issues,
            'message': 'Code quality checks passed' if passed else f'{len(issues)} issues found'
        }

    def _validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """Validate Python code syntax."""
        try:
            # Try to parse as expression first
            ast.parse(code, mode='eval')
            return True, None
        except SyntaxError:
            try:
                # Try as statement/module
                ast.parse(code, mode='exec')
                return True, None
            except SyntaxError as e:
                return False, str(e)

    def _check_specificity(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Check if root cause is specific, not vague."""
        vague_examples = self.validation_criteria['specificity']['bad_examples']

        failure_analysis = insight['insight'].get('failure_analysis', {})
        root_cause = failure_analysis.get('root_cause_detailed', '')

        # Check if too short (likely vague)
        if len(root_cause) < 20:
            return {
                'passed': False,
                'vague_fields': ['root_cause_detailed'],
                'message': 'Root cause explanation too brief'
            }

        # Check for vague phrases
        vague_phrases = [
            'selector issue',
            'element not found',
            'timing problem',
            'test failed',
            'error occurred'
        ]

        is_vague = any(phrase in root_cause.lower() for phrase in vague_phrases)

        if is_vague:
            return {
                'passed': False,
                'vague_fields': ['root_cause_detailed'],
                'message': 'Root cause uses vague terminology'
            }

        return {
            'passed': True,
            'vague_fields': [],
            'message': 'Root cause is specific and detailed'
        }

    def _check_evidence_support(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Check if claims are supported by evidence."""
        failure_analysis = insight['insight'].get('failure_analysis', {})
        evidence = failure_analysis.get('evidence', [])

        if not evidence or len(evidence) == 0:
            return {
                'passed': False,
                'unsupported_claims': ['root_cause'],
                'message': 'No evidence provided to support claims'
            }

        # Check if evidence is substantial (not just generic statements)
        substantial_evidence = [
            ev for ev in evidence
            if len(ev) > 20 and not ev.startswith('See ')
        ]

        if len(substantial_evidence) < 1:
            return {
                'passed': False,
                'unsupported_claims': ['evidence too generic'],
                'message': 'Evidence lacks substance'
            }

        return {
            'passed': True,
            'unsupported_claims': [],
            'message': f'{len(evidence)} evidence items provided'
        }

    def _check_generalizability(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Check if pattern is generalizable, not ticket-specific."""
        pattern = insight['insight'].get('generalized_pattern', {})
        pattern_name = pattern.get('pattern_name', '')
        applies_to = pattern.get('applies_to', '')

        issues = []

        # Check if pattern name contains ticket ID (too specific)
        metadata = insight.get('metadata', {})
        ticket_id = metadata.get('ticket_id', '')

        if ticket_id and ticket_id in pattern_name:
            issues.append(f'Pattern name contains ticket ID: {ticket_id}')

        # Check if "applies_to" is generic enough
        if len(applies_to) < 15:
            issues.append('Pattern applicability description too brief')

        # Check if code template has placeholders for generalization
        code_template = pattern.get('code_template', '')
        has_placeholder = '{' in code_template and '}' in code_template

        if code_template and not has_placeholder and len(code_template) > 10:
            issues.append('Code template lacks placeholders for generalization')

        passed = len(issues) == 0

        return {
            'passed': passed,
            'issues': issues,
            'message': 'Pattern is generalizable' if passed else f'{len(issues)} generalizability issues'
        }

    def _calculate_confidence(
        self,
        validation_results: Dict[str, Any],
        insight: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""
        # Count passed checks
        passed_checks = sum(1 for result in validation_results.values() if result['passed'])
        total_checks = len(validation_results)

        validation_score = passed_checks / total_checks if total_checks > 0 else 0

        # Get learning metadata confidence
        learning_conf = insight.get('learning_metadata', {}).get('confidence', 0.5)

        # Weighted average (validation 60%, learning confidence 40%)
        overall = (validation_score * 0.6) + (learning_conf * 0.4)

        return round(overall, 2)

    def _make_decision(
        self,
        validation_results: Dict[str, Any],
        confidence_score: float
    ) -> Tuple[str, List[str]]:
        """Make approval decision based on validation results."""
        flags = []

        # Critical failures (must pass)
        if not validation_results['completeness']['passed']:
            return 'reject', ['Missing required fields']

        if not validation_results['code_quality']['passed']:
            # Check if issues are critical
            issues = validation_results['code_quality']['issues']
            critical = any('syntax error' in issue.lower() for issue in issues)
            if critical:
                return 'reject', ['Critical code quality issues']
            else:
                flags.extend(issues)

        # Decision based on confidence thresholds
        thresholds = self.config.get('confidence_thresholds', {})

        if confidence_score >= thresholds.get('validation_approve', 0.8):
            if not flags:
                return 'approve', []
            else:
                return 'approve_with_flags', flags

        elif confidence_score >= thresholds.get('validation_flag', 0.6):
            # Gather all issues as flags
            for check_name, result in validation_results.items():
                if not result['passed']:
                    flags.append(f"{check_name}: {result.get('message', 'failed')}")
            return 'approve_with_flags', flags

        elif confidence_score >= thresholds.get('validation_reject', 0.4):
            return 'request_more_info', ['Low confidence - need additional information']

        else:
            return 'reject', ['Confidence too low - restart analysis']

    def _assess_with_llm(
        self,
        insight: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM for additional quality assessment."""

        user_prompt = self._build_assessment_prompt(insight, validation_results)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_config.get('temperature', 0.1),
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            assessment = json.loads(response.choices[0].message.content)
            return assessment

        except Exception as e:
            print(f"     LLM assessment error: {e}")
            return {
                'quality_score': 0.5,
                'suggestions': [],
                'strengths': [],
                'weaknesses': []
            }

    def _build_assessment_prompt(
        self,
        insight: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM assessment."""

        prompt_parts = [
            "# Assess Insight Quality",
            "",
            "## Insight to Validate",
            json.dumps(insight['insight'], indent=2),
            "",
            "## Automated Validation Results",
            ""
        ]

        for check_name, result in validation_results.items():
            status = " PASS" if result['passed'] else " FAIL"
            prompt_parts.append(f"{check_name}: {status} - {result.get('message')}")

        prompt_parts.extend([
            "",
            "## Your Task",
            "",
            "Provide additional quality assessment in JSON:",
            "{",
            '  "quality_score": 0.0-1.0,',
            '  "strengths": ["What this insight does well"],',
            '  "weaknesses": ["What could be improved"],',
            '  "suggestions": ["Specific improvements to make"],',
            '  "value_for_learning": "high|medium|low"',
            '}',
            "",
            "Focus on:",
            "- Is the root cause explanation clear and actionable?",
            "- Will the code fix actually solve the problem?",
            "- Is the pattern truly reusable for future cases?",
            "- Are prevention guidelines practical?"
        ])

        return "\n".join(prompt_parts)

    def _get_action_required(self, decision: str) -> Dict[str, str]:
        """Get action descriptions for each decision."""
        actions = {
            'approve': 'Save to insights/raw/ and proceed',
            'approve_with_flags': 'Save but mark for human review',
            'request_more_info': 'Return to QuestionGenerator with specific asks',
            'reject': 'Restart feedback loop with better prompts'
        }

        return {
            'decision': decision,
            'action': actions.get(decision, 'Unknown action'),
            'next_step': 'save_insight' if decision in ['approve', 'approve_with_flags'] else 'restart_loop'
        }


def create_quality_validator_agent(config: Dict[str, Any]) -> QualityValidatorAgent:
    """Factory function to create QualityValidator agent."""
    return QualityValidatorAgent(config)
