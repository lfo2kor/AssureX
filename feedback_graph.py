"""
Feedback Graph - LangGraph orchestrator for intelligent feedback system.
Connects 5 agents in a dynamic workflow based on confidence and completeness.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import yaml
from pathlib import Path

# Import all agents
from agents.agent_tools import AgentTools
from agents.input_parser import create_input_parser_agent
from agents.question_generator import create_question_generator_agent
from agents.solution_searcher import create_solution_searcher_agent
from agents.insight_enhancer import create_insight_enhancer_agent
from agents.quality_validator import create_quality_validator_agent


# Define graph state
class FeedbackState(TypedDict):
    """State that flows through the feedback graph."""

    # Input
    failure_context: Dict[str, Any]

    # Agent outputs
    parser_analysis: Dict[str, Any]
    questions: Dict[str, Any]
    tester_answers: Dict[str, Any]
    solution_analysis: Dict[str, Any]
    enhanced_insight: Dict[str, Any]
    validation_result: Dict[str, Any]

    # Control flow
    next_action: str
    retry_count: int

    # Final output
    final_insight: Dict[str, Any]
    success: bool


class FeedbackGraph:
    """
    LangGraph-based orchestrator for the feedback system.
    Routes between agents dynamically based on confidence and completeness.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize feedback graph with configuration.

        Args:
            config_path: Path to feedback_config.yaml
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'feedback_config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize shared tools
        self.tools = AgentTools(self.config)

        # Initialize agents
        self.input_parser = create_input_parser_agent(self.config, self.tools)
        self.question_generator = create_question_generator_agent(self.config)
        self.solution_searcher = create_solution_searcher_agent(self.config, self.tools)
        self.insight_enhancer = create_insight_enhancer_agent(self.config)
        self.quality_validator = create_quality_validator_agent(self.config)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Create graph
        workflow = StateGraph(FeedbackState)

        # Add nodes (agents)
        workflow.add_node("input_parser", self._run_input_parser)
        workflow.add_node("question_generator", self._run_question_generator)
        workflow.add_node("solution_searcher", self._run_solution_searcher)
        workflow.add_node("insight_enhancer", self._run_insight_enhancer)
        workflow.add_node("quality_validator", self._run_quality_validator)

        # Set entry point
        workflow.set_entry_point("input_parser")

        # Add conditional edges (routing logic)
        workflow.add_conditional_edges(
            "input_parser",
            self._route_after_parser,
            {
                "ask_questions": "question_generator",
                "solution_search": "solution_searcher"
            }
        )

        workflow.add_conditional_edges(
            "question_generator",
            self._route_after_questions,
            {
                "solution_search": "solution_searcher"
            }
        )

        workflow.add_edge("solution_searcher", "insight_enhancer")
        workflow.add_edge("insight_enhancer", "quality_validator")

        workflow.add_conditional_edges(
            "quality_validator",
            self._route_after_validation,
            {
                "approve": END,
                "approve_with_flags": END,
                "request_more_info": "question_generator",
                "reject": "input_parser"
            }
        )

        return workflow.compile()

    # Agent execution nodes

    def _run_input_parser(self, state: FeedbackState) -> FeedbackState:
        """Execute InputParser agent."""
        print("\n" + "="*60)
        print(" AGENT: InputParser")
        print("="*60)

        analysis = self.input_parser.analyze_failure(state['failure_context'])

        state['parser_analysis'] = analysis
        state['next_action'] = analysis.get('next_action', 'ask_questions')

        return state

    def _run_question_generator(self, state: FeedbackState) -> FeedbackState:
        """Execute QuestionGenerator agent."""
        print("\n" + "="*60)
        print(" AGENT: QuestionGenerator")
        print("="*60)

        questions_result = self.question_generator.generate_questions(
            state['failure_context'],
            state['parser_analysis']
        )

        state['questions'] = questions_result

        # If questions needed, collect answers from tester
        if questions_result.get('questions_needed'):
            answers = self.question_generator.collect_answers(
                questions_result.get('questions', [])
            )
            state['tester_answers'] = answers
        else:
            state['tester_answers'] = {}

        return state

    def _run_solution_searcher(self, state: FeedbackState) -> FeedbackState:
        """Execute SolutionSearcher agent."""
        print("\n" + "="*60)
        print(" AGENT: SolutionSearcher")
        print("="*60)

        solution_analysis = self.solution_searcher.search_and_analyze(
            state['failure_context'],
            state['parser_analysis'],
            state.get('tester_answers')
        )

        state['solution_analysis'] = solution_analysis

        return state

    def _run_insight_enhancer(self, state: FeedbackState) -> FeedbackState:
        """Execute InsightEnhancer agent."""
        print("\n" + "="*60)
        print(" AGENT: InsightEnhancer")
        print("="*60)

        enhanced = self.insight_enhancer.enhance_insight(
            state['failure_context'],
            state['parser_analysis'],
            state.get('tester_answers', {}),
            state['solution_analysis']
        )

        state['enhanced_insight'] = enhanced

        return state

    def _run_quality_validator(self, state: FeedbackState) -> FeedbackState:
        """Execute QualityValidator agent."""
        print("\n" + "="*60)
        print(" AGENT: QualityValidator")
        print("="*60)

        validation = self.quality_validator.validate_insight(
            state['enhanced_insight']
        )

        state['validation_result'] = validation
        state['next_action'] = validation.get('decision', 'approve')

        # If approved, set final insight
        if validation.get('validated'):
            state['final_insight'] = state['enhanced_insight']['insight']
            state['success'] = True
        else:
            state['success'] = False
            state['retry_count'] = state.get('retry_count', 0) + 1

        return state

    # Routing logic

    def _route_after_parser(self, state: FeedbackState) -> str:
        """Route after InputParser based on confidence and missing fields."""

        confidence = state['parser_analysis'].get('confidence', 0)
        missing_fields = state['parser_analysis'].get('missing_fields', [])

        auto_fix_threshold = self.config.get('confidence_thresholds', {}).get(
            'auto_fix_threshold', 0.85
        )

        # High confidence and no missing fields -> Skip questions
        if confidence >= auto_fix_threshold and not missing_fields:
            print(f"\n    High confidence ({confidence:.2f}) - Skipping questions")
            return "solution_search"
        else:
            print(f"\n    Need more info (confidence: {confidence:.2f}) - Asking questions")
            return "ask_questions"

    def _route_after_questions(self, state: FeedbackState) -> str:
        """Always proceed to solution search after questions."""
        return "solution_search"

    def _route_after_validation(self, state: FeedbackState) -> str:
        """Route after QualityValidator based on decision."""

        decision = state['validation_result'].get('decision', 'approve')
        retry_count = state.get('retry_count', 0)
        max_retries = 2

        # Prevent infinite loops
        if retry_count >= max_retries and decision in ['request_more_info', 'reject']:
            print(f"\n     Max retries ({max_retries}) reached - Approving with warnings")
            state['success'] = True
            state['final_insight'] = state['enhanced_insight']['insight']
            return "approve_with_flags"

        print(f"\n    Validation decision: {decision}")
        return decision

    def run(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the feedback graph to collect insight for a failure.

        Args:
            failure_context: Dictionary with failure information

        Returns:
            Dictionary with final insight and metadata
        """
        print("\n" + "="*60)
        print(" STARTING FEEDBACK GRAPH")
        print("="*60)
        print(f"Ticket: {failure_context.get('ticket_id')}")
        print(f"Failed Step: {failure_context.get('failed_step')} - {failure_context.get('step_name')}")
        print("="*60)

        # Initialize state
        initial_state = {
            'failure_context': failure_context,
            'parser_analysis': {},
            'questions': {},
            'tester_answers': {},
            'solution_analysis': {},
            'enhanced_insight': {},
            'validation_result': {},
            'next_action': '',
            'retry_count': 0,
            'final_insight': {},
            'success': False
        }

        # Run graph
        try:
            final_state = self.graph.invoke(initial_state)

            print("\n" + "="*60)
            print(" FEEDBACK GRAPH COMPLETE")
            print("="*60)
            print(f"Success: {final_state['success']}")
            print(f"Retries: {final_state['retry_count']}")
            print("="*60 + "\n")

            return {
                'success': final_state['success'],
                'insight': final_state.get('final_insight'),
                'validation': final_state.get('validation_result'),
                'metadata': {
                    'retries': final_state['retry_count'],
                    'tester_input_needed': bool(final_state.get('tester_answers')),
                    'confidence': final_state.get('validation_result', {}).get('confidence_score', 0)
                }
            }

        except Exception as e:
            print(f"\n FEEDBACK GRAPH ERROR: {e}")
            import traceback
            traceback.print_exc()

            return {
                'success': False,
                'error': str(e),
                'insight': None
            }


def create_feedback_graph(config_path: str = None) -> FeedbackGraph:
    """
    Factory function to create feedback graph.

    Args:
        config_path: Optional path to config file

    Returns:
        FeedbackGraph instance
    """
    return FeedbackGraph(config_path)


# CLI for testing
if __name__ == "__main__":
    import sys

    # Test with mock failure context
    mock_failure = {
        'ticket_id': 'RBPLCD-8002',
        'failed_step': 5,
        'step_name': 'Click department dropdown',
        'error_message': 'TimeoutError: Waiting for selector "#dept-dropdown" failed: 30000ms',
        'screenshot_path': 'Reports/RBPLCD-8002_step5_failure.png',
        'html_snapshot_path': 'Reports/RBPLCD-8002_step5.html',
        'test_file_path': 'Generated_Scripts/RBPLCD-8002_20251130_test.py',
        'failed_line_number': 45,
        'session_data': {
            'execution_sequence': [
                {'step': 1, 'action': 'Navigate', 'status': 'PASS', 'attempt': 1},
                {'step': 2, 'action': 'Login', 'status': 'PASS', 'attempt': 1},
                {'step': 3, 'action': 'Click Add User', 'status': 'PASS', 'attempt': 1},
                {'step': 4, 'action': 'Fill name', 'status': 'PASS', 'attempt': 1},
            ]
        },
        'session_id': 'RBPLCD-8002_20251130_140000'
    }

    print("Testing Feedback Graph with mock failure...")

    graph = create_feedback_graph()
    result = graph.run(mock_failure)

    if result['success']:
        print("\n Test successful!")
        print(f"Insight collected: {result['insight'].get('metadata', {}).get('ticket_id')}")
        print(f"Confidence: {result['metadata']['confidence']:.2f}")
        print(f"Tester input needed: {result['metadata']['tester_input_needed']}")
    else:
        print("\n Test failed!")
        print(f"Error: {result.get('error')}")
