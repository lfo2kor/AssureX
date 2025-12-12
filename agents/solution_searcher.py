"""
SolutionSearcher Agent - Searches for and analyzes similar past failures.
Extracts patterns and determines solution applicability.
"""

import json
from typing import Dict, Any, List
from .llm_client import get_azure_openai_client
import yaml
from pathlib import Path
from .agent_tools import AgentTools


class SolutionSearcherAgent:
    """
    Agent that searches vector DB for similar failures and synthesizes solutions.
    Doesn't just return top result - analyzes WHY it's similar and if solution applies.
    """

    def __init__(self, config: Dict[str, Any], tools: AgentTools):
        """
        Initialize SolutionSearcher agent.

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
            self.system_prompt = self._build_system_prompt(prompts['solution_searcher'])

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
            "Search Strategy:",
            prompt_config['search_strategy'],
            "",
            "Similarity Analysis Dimensions:",
            "- Symptom match: Same error message/behavior?",
            "- Element match: Same type of UI element?",
            "- Context match: Same application area/flow?",
            "- Action match: Same interaction type?",
            "- Solution applicability: Will their fix work here?",
        ]

        return "\n".join(prompt_parts)

    def search_and_analyze(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for similar solutions and analyze applicability.

        Args:
            failure_context: Original failure context
            parser_analysis: Analysis from InputParser
            tester_answers: Optional answers from QuestionGenerator

        Returns:
            Dictionary with solution recommendations
        """
        print(f"\n SolutionSearcher analyzing similar past failures...")

        # Step 1: Build comprehensive search queries
        search_queries = self._build_search_queries(
            failure_context,
            parser_analysis,
            tester_answers
        )

        # Step 2: Search vector DB with multiple queries
        search_results = self._search_vector_db(search_queries)

        print(f"   Found {len(search_results)} relevant similar cases")

        # Step 3: Analyze results with LLM
        if search_results:
            analysis = self._analyze_with_llm(
                failure_context,
                parser_analysis,
                tester_answers,
                search_results
            )
        else:
            analysis = {
                'similar_cases': [],
                'recommended_solution': None,
                'confidence': 0.0
            }

        return {
            'agent': 'solution_searcher',
            'search_queries_used': search_queries,
            'similar_cases_found': len(search_results),
            'similar_cases': analysis.get('similar_cases', []),
            'extracted_patterns': analysis.get('extracted_patterns', []),
            'recommended_solution': analysis.get('recommended_solution'),
            'confidence': analysis.get('confidence', 0.0),
            'raw_search_results': search_results
        }

    def _build_search_queries(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any]
    ) -> List[str]:
        """Build multiple search queries for comprehensive search."""
        queries = []

        # Query 1: Based on root cause and element type
        root_cause = parser_analysis.get('root_cause', '')
        element_type = parser_analysis.get('element_details', {}).get('element_type', '')

        if root_cause and element_type:
            queries.append(f"{root_cause} {element_type} {failure_context.get('step_name', '')}")

        # Query 2: Based on error message
        error = failure_context.get('error_message', '')
        if error:
            # Extract key parts of error
            if 'timeout' in error.lower():
                queries.append(f"timeout {element_type}")
            if 'not found' in error.lower():
                queries.append(f"selector not found {element_type}")

        # Query 3: Based on tester answers (if available)
        if tester_answers:
            # If tester provided selector type
            for key, value in tester_answers.items():
                if 'selector' in key.lower() and value != 'skipped':
                    queries.append(f"{value} {element_type} selector")

        # Query 4: Broad query as fallback
        queries.append(f"{element_type} interaction failure")

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:3]  # Limit to top 3 queries

    def _search_vector_db(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple vector DB searches and combine results."""
        all_results = []
        seen_ids = set()

        for query in queries:
            result = self.tools.search_vector_db(
                query=query,
                top_k=3  # Get top 3 for each query
            )

            if result.get('success') and result.get('results'):
                for item in result['results']:
                    # Use content hash as ID to avoid duplicates
                    item_id = hash(item.get('content', '')[:100])

                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        item['query'] = query  # Track which query found this
                        all_results.append(item)

        # Sort by similarity score
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # Return top results
        max_results = self.config.get('vector_db', {}).get('search_params', {}).get('top_k', 5)
        return all_results[:max_results]

    def _analyze_with_llm(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to analyze search results and recommend solutions."""

        user_prompt = self._build_analysis_prompt(
            failure_context,
            parser_analysis,
            tester_answers,
            search_results
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

            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            print(f"     LLM analysis error: {e}")
            return {
                'similar_cases': [],
                'recommended_solution': None,
                'confidence': 0.0,
                'error': str(e)
            }

    def _build_analysis_prompt(
        self,
        failure_context: Dict[str, Any],
        parser_analysis: Dict[str, Any],
        tester_answers: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Build comprehensive analysis prompt."""

        prompt_parts = [
            "# Solution Analysis from Similar Cases",
            "",
            "## Current Failure",
            f"Ticket: {failure_context.get('ticket_id')}",
            f"Step: {failure_context.get('failed_step')} - {failure_context.get('step_name')}",
            f"Error: {failure_context.get('error_message')}",
            "",
            "## Analysis So Far",
            f"Root Cause: {parser_analysis.get('root_cause')}",
            f"Element Type: {parser_analysis.get('element_details', {}).get('element_type')}",
        ]

        if parser_analysis.get('element_details', {}).get('old_selector'):
            prompt_parts.append(f"Old Selector: {parser_analysis['element_details']['old_selector']}")

        if parser_analysis.get('element_details', {}).get('suggested_new_selector'):
            prompt_parts.append(f"Suggested New: {parser_analysis['element_details']['suggested_new_selector']}")

        prompt_parts.append("")

        # Add tester answers if available
        if tester_answers:
            prompt_parts.extend([
                "## Tester Input",
                ""
            ])
            for key, value in tester_answers.items():
                if value != 'skipped':
                    prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("")

        # Add search results
        prompt_parts.extend([
            f"## Similar Past Cases ({len(search_results)} found)",
            ""
        ])

        for i, result in enumerate(search_results, 1):
            prompt_parts.extend([
                f"### Case {i} (Similarity: {result.get('similarity', 0):.2f})",
                f"Found by query: {result.get('query')}",
                f"Content: {result.get('content', '')[:300]}...",
                ""
            ])
            if result.get('metadata'):
                prompt_parts.append(f"Metadata: {result['metadata']}")
                prompt_parts.append("")

        # Analysis instructions
        prompt_parts.extend([
            "## Your Task",
            "",
            "Analyze the similar cases and provide a JSON response:",
            "",
            "{",
            '  "similar_cases": [',
            '    {',
            '      "case_number": 1-5,',
            '      "similarity_score": 0.0-1.0,',
            '      "symptom_match": "exact|similar|different",',
            '      "element_match": "same|similar|different",',
            '      "context_match": "same|similar|different",',
            '      "solution_extracted": "description of their fix",',
            '      "solution_code": "code snippet if available",',
            '      "applicability_to_current": "high|medium|low",',
            '      "applicability_reasoning": "why this does/doesn\'t apply"',
            '    }',
            '  ],',
            '  "extracted_patterns": [',
            '    {',
            '      "pattern_name": "descriptive name",',
            '      "occurrences": "count in similar cases",',
            '      "pattern_description": "what the pattern is",',
            '      "code_template": "generalized code",',
            '      "applies_to_current": true/false',
            '    }',
            '  ],',
            '  "recommended_solution": {',
            '    "approach": "description of recommended fix",',
            '    "code": "specific code for current case",',
            '    "based_on": "which similar case(s) or pattern",',
            '    "confidence": 0.0-1.0,',
            '    "reasoning": "why this is recommended"',
            '  }',
            '}',
            "",
            "Analysis Guidelines:",
            "- If 3+ cases use same solution  Extract pattern",
            "- If cases conflict  Recommend based on highest similarity",
            "- If no cases apply  Return null for recommended_solution",
            "- Consider tester input when evaluating applicability",
            "- Adapt solutions to current context, don't just copy"
        ])

        return "\n".join(prompt_parts)


def create_solution_searcher_agent(config: Dict[str, Any], tools: AgentTools) -> SolutionSearcherAgent:
    """Factory function to create SolutionSearcher agent."""
    return SolutionSearcherAgent(config, tools)
