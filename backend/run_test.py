"""
Generic test runner for any Jira ticket
Usage: python run_test.py TICKET-ID [--no-cleanup]
Example: python run_test.py RBPLCD-8835
         python run_test.py RBPLCD-8835 --no-cleanup
"""
import sys
import os
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from workflows.test_workflow import create_workflow

def cleanup_old_files():
    """Remove old test artifacts before running new test"""
    print("Cleaning up old test files...")

    cleanup_targets = [
        ("Reports/*.html", "HTML reports"),
        ("Videos/*.webm", "Video recordings"),
        ("Screenshots/*.png", "Screenshots"),
        ("Generated_Scripts/*.py", "Generated Playwright scripts"),
    ]

    total_deleted = 0
    for pattern, description in cleanup_targets:
        files = glob.glob(pattern)
        if files:
            for file in files:
                try:
                    os.remove(file)
                    total_deleted += 1
                except Exception as e:
                    print(f"  Warning: Could not delete {file}: {e}")

    if total_deleted > 0:
        print(f"Removed {total_deleted} old files")
    print()

def run_ticket_test(ticket_id, skip_cleanup=False):
    """Run test automation for a Jira ticket"""

    # Validate ticket file exists
    ticket_file = f"Jira_Tickets/{ticket_id}.txt"
    if not os.path.exists(ticket_file):
        print(f"ERROR: Ticket file not found: {ticket_file}")
        print(f"Please create the file in Jira_Tickets/ folder")
        return 1

    print("="*80)
    print(f"           Test Automation for Jira Ticket: {ticket_id}")
    print("="*80)
    print()

    # Cleanup old files unless --no-cleanup flag is used
    if not skip_cleanup:
        cleanup_old_files()

    # Setup logger
    logger = setup_logger("Logs", ticket_id, logger_name="TA_AI_Project")

    try:
        # Execute workflow manually to bypass user input
        logger.info("Starting workflow execution...")

        from workflows.test_workflow import load_config_node
        from agents.jira_parser_agent import jira_parser_agent
        from agents.vision_executor_agent import vision_executor_agent
        from agents.report_generator_agent import report_generator_agent

        # Initialize state with ticket number
        state = {'ticket_number': ticket_id}

        # Step 1: Load config
        state = load_config_node(state)

        # Step 2: Skip ticket input (already in state)

        # Step 3: Parse Jira
        state = jira_parser_agent(state)

        # Step 4: Execute with vision
        state = vision_executor_agent(state)

        # Step 5: Generate report
        state = report_generator_agent(state)

        logger.info("Workflow execution completed")

        # Extract results
        overall_status = state.get('overall_status', 'UNKNOWN')
        execution_results = state.get('execution_results', [])

        # Count passed/failed steps
        if isinstance(execution_results, list):
            steps_passed = len([r for r in execution_results if r.get('status') == 'PASSED'])
            total_steps = len(execution_results)
        else:
            steps_passed = 0
            total_steps = 0

        # Display results
        print()
        print("="*80)
        print("                        TEST RESULTS")
        print("="*80)
        print(f"Overall Status:  {overall_status}")
        print(f"Steps Passed:    {steps_passed}/{total_steps}")
        print()

        if execution_results:
            print("Step Details:")
            for step_result in execution_results:
                step_num = step_result.get('step_num', '?')
                step_text = step_result.get('step_text', 'N/A')
                status = step_result.get('status', 'UNKNOWN')
                selector = step_result.get('selector_used', 'N/A')
                level = step_result.get('level_used', 'N/A')

                status_symbol = "[OK]" if status == "PASSED" else "[FAIL]"
                print(f"  Step {step_num}: {status_symbol} {step_text[:60]}...")
                print(f"           Selector: {selector}")
                print(f"           Level: {level}")
                print()

        # Show generated files
        if state.get('report_path'):
            print(f"HTML Report: {state['report_path']}")
        if state.get('script_path'):
            print(f"Playwright Script: {state['script_path']}")

        # Video path might be in state directly
        video_path = state.get('video_path')
        if video_path:
            print(f"Video Recording: {video_path}")

        print("="*80)

        return 0 if overall_status == "PASSED" else 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        print(f"\nERROR: Test execution failed")
        print(f"Error: {e}")
        print("\nCheck the log file for details")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_test.py TICKET-ID [--no-cleanup]")
        print("Example: python run_test.py RBPLCD-8835")
        print("         python run_test.py RBPLCD-8835 --no-cleanup")
        print()
        print("The ticket file should be placed in Jira_Tickets/ folder")
        print("Example: Jira_Tickets/RBPLCD-8835.txt")
        print()
        print("By default, old test files are cleaned up before running.")
        print("Use --no-cleanup to skip cleanup.")
        sys.exit(1)

    ticket_id = sys.argv[1]
    skip_cleanup = "--no-cleanup" in sys.argv

    exit_code = run_ticket_test(ticket_id, skip_cleanup)
    sys.exit(exit_code)
