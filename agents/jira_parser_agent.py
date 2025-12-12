"""
Jira Parser Agent

Parses Jira ticket files and extracts structured test data including:
- Ticket ID
- Module/Component
- Test title and description
- Test steps
- Acceptance criteria
"""

import re
import logging
from pathlib import Path
from typing import Dict, List
from models.state import TestAutomationState


def jira_parser_agent(state: TestAutomationState) -> TestAutomationState:
    """
    Parse Jira ticket file and extract test information.

    Reads the Jira ticket file, extracts structured data, and updates state
    with parsed information.

    Args:
        state: Current workflow state containing config and ticket_number

    Returns:
        Updated state with jira_data populated

    Raises:
        FileNotFoundError: If Jira ticket file doesn't exist
        ValueError: If ticket format is invalid
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("=" * 70)
    logger.info("JIRA PARSER AGENT - Starting")
    logger.info("=" * 70)

    # Get ticket number and file path
    ticket_number = state.get('ticket_number', '')
    if not ticket_number:
        error_msg = "No ticket number provided in state"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Parsing ticket: {ticket_number}")

    # Construct file path
    jira_folder = state['config']['folders']['jira']
    ticket_file = Path(jira_folder) / f"{ticket_number}.txt"

    logger.debug(f"Looking for file: {ticket_file}")

    # Check if file exists
    if not ticket_file.exists():
        error_msg = f"Jira ticket file not found: {ticket_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Read file content
    try:
        with open(ticket_file, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read ticket file ({len(content)} characters)")
    except Exception as e:
        error_msg = f"Error reading ticket file: {e}"
        logger.error(error_msg)
        raise

    # Parse ticket content
    try:
        jira_data = parse_ticket_content(content, logger)
        logger.info("Successfully parsed ticket content")
    except Exception as e:
        error_msg = f"Error parsing ticket content: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Update state
    state['jira_data'] = jira_data
    state['module'] = jira_data.get('module', 'Unknown')
    state['test_title'] = jira_data.get('title', '')
    state['description'] = jira_data.get('description', '')
    state['steps'] = jira_data.get('steps', [])
    state['acceptance_criteria'] = jira_data.get('acceptance_criteria', '')

    # Log summary
    logger.info("Jira Data Summary:")
    logger.info(f"  - Ticket ID: {jira_data.get('ticket_id', 'N/A')}")
    logger.info(f"  - Module: {jira_data.get('module', 'N/A')}")
    logger.info(f"  - Title: {jira_data.get('title', 'N/A')}")
    logger.info(f"  - Steps: {len(jira_data.get('steps', []))}")
    logger.info(f"  - Acceptance Criteria: {jira_data.get('acceptance_criteria', 'N/A')[:50]}...")

    logger.info("JIRA PARSER AGENT - Complete")
    logger.info("=" * 70)

    return state


def parse_ticket_content(content: str, logger: logging.Logger) -> Dict:
    """
    Parse Jira ticket content and extract structured data.

    Args:
        content: Raw ticket file content
        logger: Logger instance

    Returns:
        Dictionary containing:
            - ticket_id: Extracted ticket ID (e.g., "RBPLCD-8835")
            - module: Component/module name
            - title: Test title
            - description: Full description
            - steps: List of dicts with step_num and text
            - acceptance_criteria: Expected outcome

    Raises:
        ValueError: If required fields cannot be extracted
    """
    lines = content.split('\n')
    jira_data = {
        'ticket_id': '',
        'module': '',
        'title': '',
        'description': '',
        'steps': [],
        'acceptance_criteria': ''
    }

    # Extract ticket ID from title line
    # Expected format: [TICKET-ID] title text
    title_line = lines[0] if lines else ''
    ticket_id_match = re.search(r'\[([A-Z]+-\d+)\]', title_line)

    if ticket_id_match:
        jira_data['ticket_id'] = ticket_id_match.group(1)
        # Extract title (everything after the ticket ID)
        jira_data['title'] = title_line.split(']', 1)[1].strip() if ']' in title_line else ''
        logger.debug(f"Extracted ticket ID: {jira_data['ticket_id']}")
        logger.debug(f"Extracted title: {jira_data['title']}")
    else:
        logger.warning("Could not extract ticket ID from title line")

    # Extract module from Component/s line
    for line in lines:
        if line.startswith('Component/s:'):
            jira_data['module'] = line.split(':', 1)[1].strip()
            logger.debug(f"Extracted module: {jira_data['module']}")
            break

    # Extract steps from "Steps to Reproduce:" section
    steps_section_found = False
    step_lines = []

    for i, line in enumerate(lines):
        if 'Steps to Reproduce:' in line:
            steps_section_found = True
            logger.debug("Found 'Steps to Reproduce' section")
            continue

        if steps_section_found:
            # Stop at "Acceptance Criteria:" or empty line followed by a header
            if 'Acceptance Criteria:' in line or (line.strip() == '' and i + 1 < len(lines) and lines[i + 1].strip().endswith(':')):
                break

            # Parse step lines (format: "1. Step text" or "1) Step text")
            step_match = re.match(r'^(\d+)[\.\)]\s+(.+)$', line.strip())
            if step_match:
                step_num = int(step_match.group(1))
                step_text = step_match.group(2).strip()
                step_lines.append({'num': step_num, 'text': step_text})
                logger.debug(f"Extracted step {step_num}: {step_text[:50]}...")

    jira_data['steps'] = step_lines
    logger.info(f"Extracted {len(step_lines)} test steps")

    # Extract acceptance criteria
    criteria_section_found = False
    criteria_lines = []

    for line in lines:
        if 'Acceptance Criteria:' in line:
            criteria_section_found = True
            # Check if criteria is on the same line
            after_colon = line.split(':', 1)[1].strip()
            if after_colon:
                criteria_lines.append(after_colon)
            continue

        if criteria_section_found:
            # Stop at empty line or next section
            if line.strip() == '':
                break
            criteria_lines.append(line.strip())

    jira_data['acceptance_criteria'] = ' '.join(criteria_lines)
    logger.debug(f"Extracted acceptance criteria: {jira_data['acceptance_criteria'][:100]}...")

    # Build description from parsed data
    jira_data['description'] = f"{jira_data['title']} - {len(step_lines)} steps"

    # Validate required fields
    if not jira_data['ticket_id']:
        raise ValueError("Could not extract ticket ID from file")

    if not jira_data['steps']:
        raise ValueError("Could not extract test steps from file")

    return jira_data
