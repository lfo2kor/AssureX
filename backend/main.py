"""
Main Entry Point - AI-Powered Vision-Based Test Automation

Usage:
    python main.py

The tool will:
1. Load configuration
2. Prompt for Jira ticket number
3. Parse ticket and execute tests
4. Generate HTML report and Playwright script
"""

import sys
import os
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.logger import setup_logger
from workflows.test_workflow import create_workflow


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print(" " * 15 + "AI-Powered Vision-Based Test Automation")
    print(" " * 30 + "Version 1.0 PoC")
    print("=" * 80)
    print()


def print_summary(state: dict):
    """
    Print execution summary.

    Args:
        state: Final workflow state
    """
    jira_data = state.get('jira_data', {})
    overall_status = state.get('overall_status', 'UNKNOWN')
    total_time = state.get('total_execution_time', 0)
    execution_results = state.get('execution_results', [])

    print("\n" + "=" * 80)
    print(" " * 30 + "EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\n📋 Ticket ID:       {jira_data.get('ticket_id', 'N/A')}")
    print(f"📦 Module:          {jira_data.get('module', 'N/A')}")
    print(f"📝 Title:           {jira_data.get('title', 'N/A')}")

    # Status with color
    status_symbol = "✓" if overall_status == "PASSED" else "✗"
    print(f"\n{status_symbol} Overall Status:  {overall_status}")

    # Step results
    passed = len([r for r in execution_results if r['status'] == 'PASSED'])
    failed = len([r for r in execution_results if r['status'] == 'FAILED'])
    print(f"📊 Steps:           {passed} passed, {failed} failed (total: {len(execution_results)})")

    # Timing
    print(f"⏱️  Execution Time:  {total_time:.2f} seconds")

    # Output files
    print(f"\n📄 HTML Report:     {state.get('report_path', 'Not generated')}")
    print(f"🎬 Video:           {state.get('video_path', 'Not available')}")
    print(f"📜 Script:          {state.get('script_path', 'Not generated')}")

    print("\n" + "=" * 80)

    if overall_status == "PASSED":
        print(" " * 25 + "🎉 Test Execution Successful!")
    else:
        print(" " * 25 + "⚠️  Test Execution Had Failures")

    print("=" * 80)


def main():
    """Main entry point."""
    # Print banner
    print_banner()

    # Initialize logger (temporary, will be replaced after getting ticket number)
    temp_logger = setup_logger("Logs", "TEMP", logger_name="TA_AI_Project")

    try:
        # Create workflow
        temp_logger.info("Creating workflow...")
        app = create_workflow()
        temp_logger.info("Workflow created successfully")

        # Initialize empty state
        initial_state = {}

        # Execute workflow
        temp_logger.info("Starting workflow execution...")
        print("Starting test automation workflow...\n")

        final_state = app.invoke(initial_state)

        # Print summary
        print_summary(final_state)

        # Return exit code based on status
        overall_status = final_state.get('overall_status', 'UNKNOWN')
        return 0 if overall_status == 'PASSED' else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user (Ctrl+C)")
        temp_logger.warning("Execution interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        temp_logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
