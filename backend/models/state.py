"""
State schema for LangGraph workflow.

This module defines the TestAutomationState TypedDict that serves as the
shared state/memory for all agents in the workflow. State flows through
the LangGraph StateGraph and is updated by each agent.
"""

from typing import TypedDict, List, Dict, Optional


class TestAutomationState(TypedDict, total=False):
    """
    Shared state for the test automation workflow.

    This state is passed through all LangGraph nodes and updated by each agent.
    All inter-agent communication happens through this state dictionary.

    Attributes:
        Configuration:
            config: Loaded configuration from plcdtest_config.yaml
            ticket_number: Jira ticket number entered by user

        Jira Parser Output:
            jira_data: Complete parsed Jira ticket data
            module: Component/module from Jira (e.g., "Teststep")
            test_title: Test title extracted from ticket
            description: Test description
            steps: List of test steps with step numbers and text
            acceptance_criteria: Expected outcome criteria

        Vision Execution Output:
            execution_results: List of step execution results with screenshots
            screenshots: List of all screenshot file paths
            execution_start_time: ISO format timestamp of execution start
            execution_end_time: ISO format timestamp of execution end
            total_execution_time: Total time in seconds
            overall_status: "PASSED" or "FAILED"
            video_path: Path to recorded execution video

        Browser Context Memory:
            browser_context: Current browser state and page info
            current_step: Current step number being executed

        Conversation History:
            conversation_history: List of AI interactions and decisions
            errors: List of errors encountered during execution

        Report Output:
            report_path: Path to generated HTML report
            script_path: Path to generated Playwright script
            report_generation_status: "success" or "failed"
    """

    # Configuration
    config: Dict
    ticket_number: str

    # Jira Parser Output
    jira_data: Dict
    module: str
    test_title: str
    description: str
    steps: List[Dict]
    acceptance_criteria: str

    # Vision Execution Output
    execution_results: List[Dict]
    screenshots: List[str]
    execution_start_time: str
    execution_end_time: str
    total_execution_time: float
    overall_status: str
    video_path: str

    # Browser Context Memory
    browser_context: Dict
    current_step: int

    # Conversation History
    conversation_history: List[Dict]
    errors: List[Dict]

    # Report Output
    report_path: str
    script_path: str
    report_generation_status: str
