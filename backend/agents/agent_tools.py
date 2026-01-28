"""
Shared tools for intelligent feedback agents.
These tools provide evidence-gathering capabilities for LLM-based reasoning.
"""

import os
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from PIL import Image
import chromadb
from chromadb.utils import embedding_functions


class AgentTools:
    """Shared tools that agents use to gather evidence and analyze failures."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent tools with configuration.

        Args:
            config: Configuration dictionary from feedback_config.yaml
        """
        self.config = config
        self.vector_db_path = config.get('vector_db', {}).get('path', 'vectordb/plcd_context.db')
        self.project_root = Path(__file__).parent.parent

        # Initialize vector DB client
        self._init_vector_db()

    def _init_vector_db(self):
        """Initialize ChromaDB client for vector search."""
        try:
            # Use same setup as existing vectordb
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name=self.config.get('vector_db', {}).get('embedding_model', 'text-embedding-3-small')
            )

            client = chromadb.PersistentClient(path=str(self.project_root / "vectordb"))
            self.collection = client.get_or_create_collection(
                name="plcd_context",
                embedding_function=openai_ef
            )
        except Exception as e:
            print(f"Warning: Could not initialize vector DB: {e}")
            self.collection = None

    def read_screenshot(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Analyze screenshot to extract visual information.

        Args:
            screenshot_path: Path to screenshot image

        Returns:
            Dictionary with screenshot analysis results
        """
        try:
            full_path = self.project_root / screenshot_path

            if not full_path.exists():
                return {
                    'success': False,
                    'error': f'Screenshot not found: {screenshot_path}'
                }

            # Open and get basic image info
            with Image.open(full_path) as img:
                analysis = {
                    'success': True,
                    'path': screenshot_path,
                    'dimensions': {
                        'width': img.width,
                        'height': img.height
                    },
                    'format': img.format,
                    'mode': img.mode,
                    'description': f'Screenshot captured at resolution {img.width}x{img.height}'
                }

                # Note: For advanced analysis (OCR, element detection), integrate with vision models
                # For now, we rely on LLM's multimodal capabilities when screenshot is shown

                return analysis

        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading screenshot: {str(e)}'
            }

    def parse_html_snapshot(self, html_path: str, target_selector: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse HTML snapshot to extract DOM information.

        Args:
            html_path: Path to HTML snapshot file
            target_selector: Optional specific selector to look for

        Returns:
            Dictionary with HTML analysis results
        """
        try:
            full_path = self.project_root / html_path

            if not full_path.exists():
                return {
                    'success': False,
                    'error': f'HTML snapshot not found: {html_path}'
                }

            with open(full_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            result = {
                'success': True,
                'path': html_path,
                'total_elements': len(soup.find_all()),
                'interactive_elements': []
            }

            # Find interactive elements (buttons, inputs, dropdowns, etc.)
            interactive_tags = ['button', 'input', 'select', 'textarea', 'a']
            for tag in interactive_tags:
                elements = soup.find_all(tag)
                for elem in elements[:20]:  # Limit to first 20 of each type
                    elem_info = {
                        'tag': tag,
                        'id': elem.get('id'),
                        'class': elem.get('class'),
                        'data-testid': elem.get('data-testid'),
                        'name': elem.get('name'),
                        'text': elem.get_text(strip=True)[:50] if elem.get_text(strip=True) else None
                    }
                    result['interactive_elements'].append(elem_info)

            # If looking for specific selector
            if target_selector:
                result['target_selector_analysis'] = self._analyze_selector(soup, target_selector)

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'Error parsing HTML: {str(e)}'
            }

    def _analyze_selector(self, soup: BeautifulSoup, selector: str) -> Dict[str, Any]:
        """Analyze if a specific selector exists in HTML."""
        analysis = {
            'selector': selector,
            'found': False,
            'suggestions': []
        }

        try:
            # Try CSS selector
            elements = soup.select(selector)
            if elements:
                analysis['found'] = True
                analysis['count'] = len(elements)
                analysis['first_element'] = {
                    'tag': elements[0].name,
                    'attributes': dict(elements[0].attrs)
                }
            else:
                # Suggest alternatives
                # If it's an ID selector that failed
                if selector.startswith('#'):
                    id_value = selector[1:]
                    # Look for similar IDs
                    all_ids = [elem.get('id') for elem in soup.find_all(id=True)]
                    analysis['suggestions'].append({
                        'type': 'similar_ids',
                        'values': [f'#{id}' for id in all_ids if id and id_value.lower() in id.lower()]
                    })

                # Suggest data-testid alternatives
                testid_elements = soup.find_all(attrs={'data-testid': True})
                if testid_elements:
                    analysis['suggestions'].append({
                        'type': 'data-testid_available',
                        'values': [f'[data-testid="{elem.get("data-testid")}"]'
                                   for elem in testid_elements[:5]]
                    })

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def search_vector_db(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search vector database for similar past failures.

        Args:
            query: Semantic search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Dictionary with search results
        """
        if not self.collection:
            return {
                'success': False,
                'error': 'Vector DB not initialized'
            }

        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters if filters else None
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else None
                    })

            return {
                'success': True,
                'query': query,
                'results_count': len(formatted_results),
                'results': formatted_results
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Vector DB search error: {str(e)}'
            }

    def read_test_code(self, test_file_path: str, line_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Read and analyze test code.

        Args:
            test_file_path: Path to test file
            line_number: Optional specific line to focus on

        Returns:
            Dictionary with code analysis
        """
        try:
            full_path = self.project_root / test_file_path

            if not full_path.exists():
                return {
                    'success': False,
                    'error': f'Test file not found: {test_file_path}'
                }

            with open(full_path, 'r', encoding='utf-8') as f:
                code_lines = f.readlines()

            result = {
                'success': True,
                'path': test_file_path,
                'total_lines': len(code_lines),
                'code': ''.join(code_lines)
            }

            # If specific line requested
            if line_number and 0 < line_number <= len(code_lines):
                context_start = max(0, line_number - 5)
                context_end = min(len(code_lines), line_number + 5)

                result['focused_line'] = {
                    'line_number': line_number,
                    'line_content': code_lines[line_number - 1].strip(),
                    'context': ''.join(code_lines[context_start:context_end]),
                    'context_range': f'{context_start + 1}-{context_end}'
                }

            # Try to parse AST for structure analysis
            try:
                tree = ast.parse(''.join(code_lines))
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                result['functions'] = functions
            except:
                pass

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading test code: {str(e)}'
            }

    def analyze_sequential_context(self, session_data: Dict[str, Any], current_step: int) -> Dict[str, Any]:
        """
        Analyze what happened in previous test steps to understand context.

        Args:
            session_data: Current session execution data
            current_step: Current step number that failed

        Returns:
            Dictionary with sequential context analysis
        """
        try:
            context_window = self.config.get('agent_tools', {}).get('sequential_context', {}).get(
                'context_window_steps', 10
            )

            execution_sequence = session_data.get('execution_sequence', [])

            # Get relevant previous steps
            relevant_steps = [
                step for step in execution_sequence
                if step.get('step') < current_step
            ][-context_window:]

            analysis = {
                'success': True,
                'current_step': current_step,
                'previous_steps_analyzed': len(relevant_steps),
                'steps': []
            }

            for step in relevant_steps:
                step_info = {
                    'step_number': step.get('step'),
                    'action': step.get('action'),
                    'status': step.get('status'),
                    'attempts': step.get('attempt', 1)
                }

                # Check if step had issues
                if step.get('status') == 'FAIL':
                    step_info['had_failure'] = True
                    step_info['error'] = step.get('error')

                analysis['steps'].append(step_info)

            # Identify patterns
            analysis['patterns'] = {
                'consecutive_passes_before_failure': len([s for s in relevant_steps if s.get('status') == 'PASS']),
                'any_previous_failures': any(s.get('status') == 'FAIL' for s in relevant_steps),
                'last_action_before_failure': relevant_steps[-1].get('action') if relevant_steps else None
            }

            return analysis

        except Exception as e:
            return {
                'success': False,
                'error': f'Error analyzing sequential context: {str(e)}'
            }

    def validate_python_code(self, code: str) -> Dict[str, Any]:
        """
        Validate Python code syntax.

        Args:
            code: Python code string to validate

        Returns:
            Dictionary with validation results
        """
        try:
            ast.parse(code)
            return {
                'valid': True,
                'message': 'Code syntax is valid'
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset
            }

    def extract_code_pattern(self, insights: List[Dict[str, Any]], pattern_type: str) -> Dict[str, Any]:
        """
        Extract common patterns from multiple insights.

        Args:
            insights: List of insight dictionaries
            pattern_type: Type of pattern to extract (e.g., 'selector', 'timing', 'assertion')

        Returns:
            Dictionary with extracted pattern information
        """
        try:
            min_occurrences = self.config.get('vector_db', {}).get('pattern_extraction', {}).get(
                'min_occurrences', 3
            )

            # Group insights by similarity
            patterns = {}

            for insight in insights:
                # Extract pattern key based on type
                if pattern_type == 'selector':
                    key = self._extract_selector_pattern(insight)
                elif pattern_type == 'timing':
                    key = self._extract_timing_pattern(insight)
                else:
                    key = insight.get('failure_analysis', {}).get('root_cause', 'unknown')

                if key not in patterns:
                    patterns[key] = {
                        'pattern': key,
                        'occurrences': 0,
                        'examples': []
                    }

                patterns[key]['occurrences'] += 1
                patterns[key]['examples'].append({
                    'ticket': insight.get('metadata', {}).get('ticket_id'),
                    'solution': insight.get('solution', {}).get('new_code')
                })

            # Filter by minimum occurrences
            significant_patterns = {
                k: v for k, v in patterns.items()
                if v['occurrences'] >= min_occurrences
            }

            return {
                'success': True,
                'pattern_type': pattern_type,
                'patterns_found': len(significant_patterns),
                'patterns': significant_patterns
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error extracting patterns: {str(e)}'
            }

    def _extract_selector_pattern(self, insight: Dict[str, Any]) -> str:
        """Extract selector pattern from insight."""
        solution = insight.get('solution', {})
        new_code = solution.get('new_code', '')

        # Extract selector type
        if 'data-testid' in new_code:
            return 'data-testid'
        elif '[class=' in new_code or '.' in new_code:
            return 'css_class'
        elif '[id=' in new_code or '#' in new_code:
            return 'id'
        elif 'text=' in new_code or 'get_by_text' in new_code:
            return 'text_content'
        else:
            return 'other'

    def _extract_timing_pattern(self, insight: Dict[str, Any]) -> str:
        """Extract timing pattern from insight."""
        solution = insight.get('solution', {})
        new_code = solution.get('new_code', '')

        if 'wait_for_selector' in new_code:
            return 'wait_for_selector'
        elif 'wait_for_load_state' in new_code:
            return 'wait_for_load_state'
        elif 'timeout' in new_code:
            return 'timeout_increase'
        else:
            return 'other'


# Tool registry for agents
TOOLS_REGISTRY = {
    'read_screenshot': 'Analyze visual state from screenshot',
    'parse_html_snapshot': 'Extract DOM elements and attributes from HTML',
    'search_vector_db': 'Find semantically similar past failures',
    'read_test_code': 'Read and analyze test code',
    'analyze_sequential_context': 'Review previous test steps for context',
    'validate_python_code': 'Check Python syntax validity',
    'extract_code_pattern': 'Find common patterns across multiple insights'
}
