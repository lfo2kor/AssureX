"""
LangGraph Workflow for Test Automation

Orchestrates the end-to-end test automation workflow:
1. Load configuration
2. Get ticket input from user
3. Parse Jira ticket
4. Execute tests with vision
5. Generate reports
"""

import logging
from typing import Dict
from langgraph.graph import StateGraph, END
from models.state import TestAutomationState
from utils.config_loader import load_config
from agents.jira_parser_agent import jira_parser_agent
from agents.vision_executor_agent import vision_executor_agent
from agents.report_generator_agent import report_generator_agent


def load_config_node(state: TestAutomationState) -> TestAutomationState:
    """
    Load configuration from YAML file.

    Args:
        state: Current workflow state

    Returns:
        Updated state with config loaded
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("Loading configuration...")

    try:
        config = load_config("plcdtest_config.yaml")
        state['config'] = config
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    return state


def get_ticket_input_node(state: TestAutomationState) -> TestAutomationState:
    """
    Prompt user for Jira ticket number.

    Args:
        state: Current workflow state

    Returns:
        Updated state with ticket_number
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("Getting ticket number from user...")

    # Prompt for ticket number
    while True:
        ticket_number = input("\n Enter Jira ticket number (e.g., RBPLCD-8835): ").strip()

        # Validate format
        import re
        if re.match(r'^[A-Z]+-\d+$', ticket_number):
            state['ticket_number'] = ticket_number
            logger.info(f"Ticket number set: {ticket_number}")
            break
        else:
            print("Invalid ticket format. Expected format: PROJECT-NUMBER (e.g., RBPLCD-8835)")
            logger.warning(f"Invalid ticket format entered: {ticket_number}")

    return state


def create_workflow():
    """
    Create and compile the LangGraph workflow.

    Returns:
        Compiled workflow graph
    """
    # Create StateGraph with TestAutomationState
    workflow = StateGraph(TestAutomationState)

    # Add nodes
    workflow.add_node("load_config", load_config_node)
    workflow.add_node("get_ticket_input", get_ticket_input_node)
    workflow.add_node("jira_parser", jira_parser_agent)
    workflow.add_node("vision_executor", vision_executor_agent)
    workflow.add_node("report_generator", report_generator_agent)

    # Define linear flow
    workflow.set_entry_point("load_config")
    workflow.add_edge("load_config", "get_ticket_input")
    workflow.add_edge("get_ticket_input", "jira_parser")
    workflow.add_edge("jira_parser", "vision_executor")
    workflow.add_edge("vision_executor", "report_generator")
    workflow.add_edge("report_generator", END)

    # Compile workflow
    app = workflow.compile()

    return app
