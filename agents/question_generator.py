"""
QuestionGenerator Agent - Creates intelligent, context-aware questions for testers.
Generates questions dynamically based on missing information, not hardcoded.
"""

import json
from typing import Dict, Any, List
from .llm_client import get_azure_openai_client
import yaml
from pathlib import Path


class QuestionGeneratorAgent:
    """
    Agent that generates targeted questions to collect missing information from testers.
    Questions are context-aware and informed by vector DB patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize QuestionGenerator agent.

        Args:
            config: Configuration from feedback_config.yaml
        """
        self.config = config
        self.llm_config = config.get('llm', {})

        # Load system prompt
        prompts_path = Path(__file__).parent.parent / 'prompts' / 'agent_prompts.yaml'
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.system_prompt = self._build_system_prompt(prompts['question_generator'])

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
            "Question Design Principles:",
            "- Use multiple choice when there are 2-5 common patterns",
            "- Offer 'Other' option with text input for edge cases",
            "- Show what you already found to give context",
            "- Explain WHY you're asking (what gap it fills)",
            "- Suggest answers based on vector DB similar cases",
            "",
            "Question Types Available:"
        ]

        for qtype, details in prompt_config.get('question_types', {}).items():
            prompt_parts.append(f"\n{qtype}:")
            prompt_parts.append(f"  When: {details['when']}")
            prompt_parts.append(f"  Template: {details['template']}")

        return "\n".join(prompt_parts)

    def generate_questions(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate questions based on missing information from InputParser.
        NEW: Creates single-step selector feedback with L1/L2/L3 attempts.

        Args:
            failure_context: Original failure context (includes l1_attempts, l2_attempts, l3_attempts)
            parser_analysis: Analysis from InputParser agent

        Returns:
            Dictionary with generated questions
        """
        print(f"\n QuestionGenerator creating questions for tester...")

        # NEW: Check if this is a selector-fix scenario (simplified feedback)
        if self._is_selector_fix_scenario(failure_context, parser_analysis):
            return self._generate_selector_feedback_question(failure_context, parser_analysis)

        # Legacy multi-question flow
        missing_fields = parser_analysis.get('missing_fields', [])

        if not missing_fields:
            print("     No missing fields - skipping questions")
            return {
                'agent': 'question_generator',
                'questions_needed': False,
                'questions': []
            }

        # Generate questions using LLM
        questions = self._generate_with_llm(
            failure_context,
            parser_analysis,
            missing_fields
        )

        print(f"   Generated {len(questions)} questions")

        return {
            'agent': 'question_generator',
            'questions_needed': True,
            'questions': questions,
            'missing_fields': missing_fields
        }

    def _is_selector_fix_scenario(self, failure_context: Dict[str, Any], parser_analysis: Dict[str, Any]) -> bool:
        """Check if this is a simple selector fix (L1/L2/L3 all tried selectors)."""
        # Check if L1/L2/L3 attempts are present
        has_attempts = (
            failure_context.get('l1_attempts') or
            failure_context.get('l2_attempts') or
            failure_context.get('l3_attempts')
        )

        # Check if root cause is selector-related
        root_cause = parser_analysis.get('root_cause', '')
        is_selector_issue = any(keyword in root_cause.lower() for keyword in ['selector', 'element', 'locator', 'not found'])

        return has_attempts and is_selector_issue

    def _generate_selector_feedback_question(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate single-step selector feedback question.

        Args:
            failure_context: Failure context with L1/L2/L3 attempts
            parser_analysis: Parser analysis with suggestions

        Returns:
            Question dictionary for single-step feedback
        """
        print("   Using single-step selector feedback")

        # Extract L1/L2/L3 attempts
        l1_attempts = failure_context.get('l1_attempts', [])
        l2_attempts = failure_context.get('l2_attempts', [])
        l3_attempts = failure_context.get('l3_attempts', [])

        total_attempts = len(l1_attempts) + len(l2_attempts) + len(l3_attempts)

        # Get next suggestions from vector DB (next 5 after L1 tried first K)
        # For now, use suggestions from parser_analysis if available
        evidence = parser_analysis.get('evidence', {})
        vector_results = evidence.get('vector_search', {}).get('results', [])

        # L1 typically tries top 5, so show next 5 (positions 5-10)
        l1_count = len(l1_attempts)
        next_suggestions = vector_results[l1_count:l1_count+5] if len(vector_results) > l1_count else []

        # Build question options from next suggestions
        options = []
        for i, result in enumerate(next_suggestions, 1):
            # Extract selector from result content (simplified - actual implementation may vary)
            selector = result.get('metadata', {}).get('correct_selector', result.get('content', '')[:100])
            similarity = result.get('similarity', 0)

            options.append({
                'label': f"Suggestion {i}",
                'value': selector,
                'similarity': similarity,
                'from_vector_db': True
            })

        # Create single selector feedback question
        question = {
            'field': 'correct_selector',
            'question': 'What is the correct selector?',
            'question_type': 'selector_feedback',  # New type
            'step_number': failure_context.get('failed_step', '?'),
            'total_attempts': total_attempts,
            'l1_attempts': l1_attempts,
            'l2_attempts': l2_attempts,
            'l3_attempts': l3_attempts,
            'options': options,
            'allow_custom_input': True
        }

        return {
            'agent': 'question_generator',
            'questions_needed': True,
            'questions': [question],
            'feedback_type': 'single_step_selector'
        }

    def _generate_with_llm(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        missing_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Use LLM to generate context-aware questions."""

        user_prompt = self._build_question_prompt(
            failure_context,
            parser_analysis,
            missing_fields
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

            result = json.loads(response.choices[0].message.content)
            return result.get('questions', [])

        except Exception as e:
            print(f"     LLM question generation error: {e}")
            # Fallback to basic questions
            return self._generate_fallback_questions(missing_fields)

    def _build_question_prompt(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        missing_fields: List[str]
    ) -> str:
        """Build prompt for question generation."""

        prompt_parts = [
            "# Generate Tester Questions",
            "",
            "## Context",
            f"Ticket: {failure_context.get('ticket_id')}",
            f"Failed Step: {failure_context.get('failed_step')} - {failure_context.get('step_name')}",
            f"Error: {failure_context.get('error_message')}",
            "",
            "## What We Know (from InputParser)",
            f"Root Cause: {parser_analysis.get('root_cause')}",
            f"Explanation: {parser_analysis.get('root_cause_explanation')}",
            f"Confidence: {parser_analysis.get('confidence'):.2f}",
            ""
        ]

        # Add evidence found
        evidence_used = parser_analysis.get('evidence_used', [])
        if evidence_used:
            prompt_parts.extend([
                "Evidence gathered:",
                *[f"  - {ev}" for ev in evidence_used],
                ""
            ])

        # Add element details if available
        element_details = parser_analysis.get('element_details', {})
        if element_details:
            prompt_parts.extend([
                "Element information:",
                f"  Type: {element_details.get('element_type', 'unknown')}",
                f"  Old selector: {element_details.get('old_selector', 'N/A')}",
            ])
            if element_details.get('suggested_new_selector'):
                prompt_parts.append(f"  Suggested new: {element_details.get('suggested_new_selector')}")
            prompt_parts.append("")

        # Add similar cases if found
        evidence = parser_analysis.get('evidence', {})
        if evidence.get('vector_search', {}).get('results'):
            prompt_parts.extend([
                "Similar past cases found:",
            ])
            for i, result in enumerate(evidence['vector_search']['results'][:2], 1):
                prompt_parts.append(f"  {i}. Similarity {result.get('similarity', 0):.2f}: {result.get('content', '')[:100]}...")
            prompt_parts.append("")

        # Specify missing information
        prompt_parts.extend([
            "## Missing Information",
            f"Fields we need: {', '.join(missing_fields)}",
            "",
            "## Your Task",
            "",
            "Generate questions to collect the missing information. Return JSON:",
            "",
            "{",
            '  "questions": [',
            '    {',
            '      "field": "field_name_this_question_fills",',
            '      "question": "Clear question text with context",',
            '      "question_type": "multiple_choice|text_input|code_snippet",',
            '      "context_shown": "What evidence you reference in the question",',
            '      "options": [  // For multiple_choice only',
            '        {',
            '          "label": "Short option label",',
            '          "value": "value_to_store",',
            '          "description": "Why this option (from similar cases if applicable)",',
            '          "from_vector_db": true/false',
            '        }',
            '      ],',
            '      "default_value": "suggested default based on evidence",',
            '      "allow_custom_input": true/false',
            '    }',
            '  ]',
            '}',
            "",
            "Guidelines:",
            "- Show what you already found (reference evidence)",
            "- For multiple choice, rank options by likelihood (most common first)",
            "- If vector DB had similar solutions, offer those as options",
            "- Keep questions simple and actionable",
            f"- Maximum {self.config.get('tester_interaction', {}).get('max_questions_per_failure', 5)} questions",
            "- Combine related missing fields into one question if possible"
        ])

        return "\n".join(prompt_parts)

    def _generate_fallback_questions(self, missing_fields: List[str]) -> List[Dict[str, Any]]:
        """Generate basic fallback questions if LLM fails."""
        questions = []

        for field in missing_fields[:3]:  # Limit to 3
            if 'selector' in field.lower():
                questions.append({
                    'field': field,
                    'question': f'What is the correct selector for this element?',
                    'question_type': 'text_input',
                    'context_shown': 'None',
                    'allow_custom_input': True
                })
            elif 'timing' in field.lower() or 'wait' in field.lower():
                questions.append({
                    'field': field,
                    'question': 'Is this a timing issue?',
                    'question_type': 'multiple_choice',
                    'options': [
                        {'label': 'Yes, needs wait', 'value': 'wait_needed'},
                        {'label': 'No, different issue', 'value': 'not_timing'}
                    ],
                    'allow_custom_input': True
                })
            else:
                questions.append({
                    'field': field,
                    'question': f'Please provide information about: {field}',
                    'question_type': 'text_input',
                    'allow_custom_input': True
                })

        return questions

    def collect_answers(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Present questions to tester and collect answers.
        NEW: Single-step feedback showing L1/L2/L3 attempts + Next 5 suggestions.

        Args:
            questions: List of question dictionaries

        Returns:
            Dictionary mapping fields to answers
        """
        if not questions:
            return {}

        # Check if this is selector feedback (new single-step approach)
        if len(questions) == 1 and questions[0].get('field') == 'correct_selector':
            return self._collect_selector_feedback(questions[0])

        # Legacy multi-question flow (backward compatible)
        print("\n" + "="*60)
        print(" I need your help to understand this failure better")
        print("="*60)

        answers = {}

        for i, q in enumerate(questions, 1):
            print(f"\n Question {i}/{len(questions)}")

            # Show context if available
            if q.get('context_shown') and q['context_shown'] != 'None':
                print(f"   Context: {q['context_shown']}")

            print(f"\n   {q['question']}")

            if q['question_type'] == 'multiple_choice':
                answers[q['field']] = self._ask_multiple_choice(q)
            elif q['question_type'] == 'code_snippet':
                answers[q['field']] = self._ask_code_snippet(q)
            else:
                answers[q['field']] = self._ask_text_input(q)

        print("\n" + "="*60)
        print(" Thank you! Processing your answers...")
        print("="*60 + "\n")

        return answers

    def _collect_selector_feedback(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        New single-step selector feedback form showing L1/L2/L3 attempts.

        Args:
            question: Question dictionary with attempts and suggestions

        Returns:
            Answer dictionary
        """
        print("\n" + "="*66)
        print(f" STEP {question.get('step_number', '?')} FAILED - Already tried {question.get('total_attempts', 0)} selectors")
        print("="*66)

        # Show L1 attempts
        l1_attempts = question.get('l1_attempts', [])
        if l1_attempts:
            print(f"L1 (ChromaDB): Tried {len(l1_attempts)} ✗")
            for i, attempt in enumerate(l1_attempts[:3], 1):  # Show first 3
                selector = attempt.get('selector', 'unknown')
                similarity = attempt.get('similarity', 0)
                print(f"  ✗ [{i}] {selector[:60]}  (sim: {similarity:.2f})" if similarity else f"  ✗ [{i}] {selector[:60]}")
            if len(l1_attempts) > 3:
                print(f"  ... and {len(l1_attempts) - 3} more")

        # Show L2 attempts
        l2_attempts = question.get('l2_attempts', [])
        if l2_attempts:
            print(f"\nL2 (Live DOM): Tried {len(l2_attempts)} ✗")
            for i, attempt in enumerate(l2_attempts[:3], 1):
                selector = attempt.get('selector', 'unknown')
                print(f"  ✗ [{i}] {selector[:60]}")
            if len(l2_attempts) > 3:
                print(f"  ... and {len(l2_attempts) - 3} more")

        # Show L3 attempts
        l3_attempts = question.get('l3_attempts', [])
        if l3_attempts:
            print(f"\nL3 (Vision): Tried {len(l3_attempts)} ✗")
            for attempt in l3_attempts:
                print(f"  ✗ {attempt.get('selector', 'Vision click')}")

        # Show next suggestions
        print(f"\n" + "-"*66)
        suggestions = question.get('options', [])
        if suggestions:
            available_count = len(suggestions)
            print(f"NEXT {available_count} AI SUGGESTIONS (from ChromaDB):")
            for i, sugg in enumerate(suggestions, 1):
                selector = sugg.get('value', sugg.get('label', 'unknown'))
                similarity = sugg.get('similarity', 0)
                print(f"  [{i}] {selector[:58]}  (sim: {similarity:.2f})" if similarity else f"  [{i}] {selector[:58]}")
        else:
            print("No more AI suggestions available.")

        print("-"*66)

        # Collect input
        print("Enter correct selector:")
        if suggestions:
            prompt = f"  [Type custom] OR [Enter 1-{len(suggestions)} to pick]: "
        else:
            prompt = "  [Type custom selector]: "

        while True:
            try:
                user_input = input(prompt).strip()

                if not user_input:
                    print("Please provide a selector")
                    continue

                # Check if picking from suggestions
                if user_input.isdigit() and suggestions:
                    choice = int(user_input)
                    if 1 <= choice <= len(suggestions):
                        selected = suggestions[choice - 1]
                        selector = selected.get('value', selected.get('label'))
                        print(f"→ Selected: {selector} ✓")
                        return {
                            'correct_selector': selector,
                            'tester_source': f'picked_suggestion_{choice}',
                            'suggestion_index': choice - 1
                        }
                    else:
                        print(f"Please enter a number between 1 and {len(suggestions)}")
                        continue

                # Custom selector
                print(f"→ Using custom selector: {user_input} ✓")
                return {
                    'correct_selector': user_input,
                    'tester_source': 'custom_input'
                }

            except KeyboardInterrupt:
                print("\n Skipping feedback...")
                return {'correct_selector': None, 'tester_source': 'skipped'}
            except Exception as e:
                print(f"Error: {e}. Please try again.")

    def _ask_multiple_choice(self, question: Dict[str, Any]) -> str:
        """Ask a multiple choice question."""
        options = question.get('options', [])

        for i, opt in enumerate(options, 1):
            desc = f" - {opt['description']}" if opt.get('description') else ""
            from_db = " [from similar case]" if opt.get('from_vector_db') else ""
            print(f"   [{i}] {opt['label']}{from_db}{desc}")

        if question.get('allow_custom_input'):
            print(f"   [{len(options) + 1}] Other (I'll type my answer)")

        while True:
            try:
                choice = input(f"\n   Your answer (1-{len(options) + (1 if question.get('allow_custom_input') else 0)}): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(options):
                    selected = options[choice_num - 1]
                    print(f"    Selected: {selected['label']}")
                    return selected['value']
                elif question.get('allow_custom_input') and choice_num == len(options) + 1:
                    custom = input("   Enter your answer: ").strip()
                    return custom
                else:
                    print(f"   Please enter a number between 1 and {len(options) + (1 if question.get('allow_custom_input') else 0)}")
            except ValueError:
                print("   Please enter a valid number")
            except KeyboardInterrupt:
                print("\n   Skipping question...")
                return "skipped"

    def _ask_text_input(self, question: Dict[str, Any]) -> str:
        """Ask a text input question."""
        if question.get('default_value'):
            print(f"   Suggested: {question['default_value']}")

        while True:
            try:
                answer = input("   Your answer: ").strip()
                if answer:
                    return answer
                elif question.get('default_value'):
                    print(f"   Using suggested value: {question['default_value']}")
                    return question['default_value']
                else:
                    print("   Please provide an answer")
            except KeyboardInterrupt:
                print("\n   Skipping question...")
                return "skipped"

    def _ask_code_snippet(self, question: Dict[str, Any]) -> str:
        """Ask for a code snippet."""
        print("   Enter code (press Ctrl+D or Ctrl+Z when done):")
        print("   " + "-"*50)

        lines = []
        try:
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\n   Skipping question...")
            return "skipped"

        code = '\n'.join(lines)
        print("   " + "-"*50)
        print(f"    Received {len(lines)} lines of code")

        return code


def create_question_generator_agent(config: Dict[str, Any]) -> QuestionGeneratorAgent:
    """Factory function to create QuestionGenerator agent."""
    return QuestionGeneratorAgent(config)
