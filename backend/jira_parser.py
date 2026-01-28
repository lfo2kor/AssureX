"""
PLCD Testing Assistant - Jira Ticket Parser
Parses Jira ticket files and extracts test steps
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/jira_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JiraTicketParser:
    """
    Parse Jira ticket text files and extract structured information
    """

    def __init__(self, jira_folder: str = "Jira_Tickets"):
        """
        Initialize Jira parser

        Args:
            jira_folder: Folder containing Jira ticket files
        """
        self.jira_folder = Path(jira_folder)
        logger.info(f"Jira parser initialized with folder: {self.jira_folder}")


    def parse_ticket(self, ticket_id: str) -> Dict:
        """
        Parse a Jira ticket file

        Args:
            ticket_id: Jira ticket ID (e.g., "RBPLCD-8835")

        Returns:
            Dictionary containing:
            - ticket_id: Ticket ID
            - title: Ticket title
            - module: Target module
            - priority: Priority level
            - type: Test type
            - description: Ticket description
            - steps: List of test steps
            - expected_result: Expected results text
            - raw_content: Raw file content

        Raises:
            FileNotFoundError: If ticket file doesn't exist
        """
        ticket_path = self.jira_folder / f"{ticket_id}.txt"

        if not ticket_path.exists():
            raise FileNotFoundError(
                f"Jira ticket not found: {ticket_path.absolute()}\n"
                f"Please ensure the ticket file exists."
            )

        logger.info(f"Parsing Jira ticket: {ticket_id}")

        with open(ticket_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata
        ticket_data = {
            "ticket_id": ticket_id,
            "title": self._extract_field(content, "Title"),
            "module": self._extract_field(content, "Module") or self._extract_field(content, "Component/s"),
            "priority": self._extract_field(content, "Priority"),
            "type": self._extract_field(content, "Type"),
            "description": self._extract_section(content, "Description") or self._extract_section(content, "Problem Description"),
            "steps": self._extract_steps(content),
            "expected_result": self._extract_section(content, "Expected Result") or self._extract_section(content, "Acceptance Criteria"),
            "raw_content": content
        }

        logger.info(f"Parsed ticket: {ticket_data['title']}")
        logger.info(f"Module: {ticket_data['module']}")
        logger.info(f"Found {len(ticket_data['steps'])} test steps")

        return ticket_data


    def _extract_field(self, content: str, field_name: str) -> str:
        """
        Extract single-line field value

        Args:
            content: Ticket content
            field_name: Field name to extract

        Returns:
            Field value or empty string
        """
        pattern = rf"^{field_name}:\s*(.+)$"
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1).strip() if match else ""


    def _extract_section(self, content: str, section_name: str) -> str:
        """
        Extract multi-line section

        Args:
            content: Ticket content
            section_name: Section name

        Returns:
            Section content or empty string
        """
        # Match section header and capture until next section or end
        pattern = rf"^{section_name}:\s*$(.*?)(?=^[A-Z][a-z\s]+:|$)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            section_content = match.group(1).strip()
            return section_content
        return ""


    def _extract_steps(self, content: str) -> List[Dict]:
        """
        Extract test steps from content

        Args:
            content: Ticket content

        Returns:
            List of step dictionaries with:
            - step_number: Step number (1-indexed)
            - step_text: Step description
        """
        # Try pattern 1: "Step N: description"
        pattern1 = r"^Step (\d+):\s*(.+)$"
        matches = re.findall(pattern1, content, re.MULTILINE)

        if matches:
            steps = []
            for step_num, step_text in matches:
                steps.append({
                    "step_number": int(step_num),
                    "step_text": step_text.strip()
                })
            return steps

        # Try pattern 2: "N. description" (numbered list format)
        # Look for steps after "Steps to Reproduce:" section
        steps_section_match = re.search(
            r"Steps to Reproduce:\s*\n((?:\d+\.\s+.+\n?)+)",
            content,
            re.MULTILINE | re.IGNORECASE
        )

        if steps_section_match:
            steps_text = steps_section_match.group(1)
            pattern2 = r"^(\d+)\.\s+(.+)$"
            matches = re.findall(pattern2, steps_text, re.MULTILINE)

            steps = []
            for step_num, step_text in matches:
                steps.append({
                    "step_number": int(step_num),
                    "step_text": step_text.strip()
                })
            return steps

        return []


    def get_step_by_number(self, ticket_data: Dict, step_number: int) -> Optional[Dict]:
        """
        Get specific step from parsed ticket

        Args:
            ticket_data: Parsed ticket data
            step_number: Step number (1-indexed)

        Returns:
            Step dictionary or None if not found
        """
        for step in ticket_data['steps']:
            if step['step_number'] == step_number:
                return step
        return None


def test_jira_parser():
    """Test the Jira parser"""

    print("=" * 80)
    print("Testing Jira Ticket Parser")
    print("=" * 80)

    parser = JiraTicketParser()

    # Test with sample ticket
    ticket_id = "RBPLCD-8835"

    try:
        print(f"\nParsing ticket: {ticket_id}")
        print("-" * 80)

        ticket_data = parser.parse_ticket(ticket_id)

        print(f"\n[OK] Ticket parsed successfully!")
        print(f"\nTicket ID: {ticket_data['ticket_id']}")
        print(f"Title: {ticket_data['title']}")
        print(f"Module: {ticket_data['module']}")
        print(f"Priority: {ticket_data['priority']}")
        print(f"Type: {ticket_data['type']}")

        print(f"\nDescription:")
        print(f"  {ticket_data['description'][:100]}...")

        print(f"\nTest Steps ({len(ticket_data['steps'])} steps):")
        for step in ticket_data['steps']:
            print(f"  {step['step_number']}. {step['step_text']}")

        print(f"\nExpected Result:")
        result_lines = ticket_data['expected_result'].split('\n')
        for line in result_lines[:3]:
            if line.strip():
                print(f"  {line.strip()}")
        if len(result_lines) > 3:
            print(f"  ... ({len(result_lines) - 3} more lines)")

        print("\n" + "-" * 80)
        print(f"[OK] Successfully parsed {len(ticket_data['steps'])} steps")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[ERROR] Parser failed: {e}")
        logger.error(f"Parser failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_jira_parser()
