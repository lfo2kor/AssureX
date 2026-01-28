"""
Test script to verify login button click fix
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from workflows.test_workflow import create_workflow

def test_with_ticket(ticket_number="RBPLCD-8835"):
    """Test with a specific ticket number."""
    print("=" * 80)
    print(" " * 15 + "Testing Login Button Fix")
    print(" " * 20 + f"Ticket: {ticket_number}")
    print("=" * 80)
    print()

    # Initialize logger
    logger = setup_logger("Logs", ticket_number, logger_name="TA_AI_Project")

    try:
        # Create workflow
        logger.info("Creating workflow...")
        app = create_workflow()
        logger.info("Workflow created successfully")

        # Initialize state with ticket number pre-filled
        initial_state = {
            'ticket_number': ticket_number
        }

        # Execute workflow starting from load_config
        logger.info("Starting workflow execution...")
        print(f"Starting test automation for ticket {ticket_number}...\n")

        # We'll manually execute the steps to bypass user input
        from workflows.test_workflow import load_config_node
        from agents.jira_parser_agent import jira_parser_agent
        from agents.vision_executor_agent import vision_executor_agent
        from agents.report_generator_agent import report_generator_agent

        # Step 1: Load config
        state = load_config_node(initial_state)

        # Step 2: Skip ticket input (already in state)

        # Step 3: Parse Jira
        state = jira_parser_agent(state)

        # Step 4: Execute with vision
        state = vision_executor_agent(state)

        # Step 5: Generate report
        state = report_generator_agent(state)

        # Print summary
        overall_status = state.get('overall_status', 'UNKNOWN')
        execution_results = state.get('execution_results', [])
        passed = len([r for r in execution_results if r['status'] == 'PASSED'])
        failed = len([r for r in execution_results if r['status'] == 'FAILED'])

        print("\n" + "=" * 80)
        print(" " * 30 + "TEST RESULTS")
        print("=" * 80)
        print(f"\n✓ Overall Status:  {overall_status}")
        print(f"📊 Steps:           {passed} passed, {failed} failed")
        print(f"📄 Report:          {state.get('report_path', 'N/A')}")
        print(f"🎬 Video:           {state.get('video_path', 'N/A')}")
        print("=" * 80)

        return 0 if overall_status == 'PASSED' else 1

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    ticket = "RBPLCD-8835"
    if len(sys.argv) > 1:
        ticket = sys.argv[1]

    exit_code = test_with_ticket(ticket)
    sys.exit(exit_code)
