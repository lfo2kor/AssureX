"""
Test Jira Parser Agent
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils.config_loader import load_config
from utils.logger import setup_logger
from agents.jira_parser_agent import jira_parser_agent
from models.state import TestAutomationState

print("=" * 70)
print("TESTING JIRA PARSER AGENT")
print("=" * 70)

# Load config
config = load_config("plcdtest_config.yaml")
print("✓ Config loaded")

# Setup logger
logger = setup_logger(config['folders']['logs'], "JIRA-TEST")
print("✓ Logger initialized")

# Create test state
state: TestAutomationState = {
    'config': config,
    'ticket_number': 'RBPLCD-8835'
}

print(f"\n✓ Testing with ticket: {state['ticket_number']}")

# Run Jira Parser Agent
try:
    updated_state = jira_parser_agent(state)
    print("\n✓ Jira Parser Agent executed successfully")

    # Verify results
    print("\n" + "=" * 70)
    print("PARSED RESULTS:")
    print("=" * 70)

    jira_data = updated_state.get('jira_data', {})

    print(f"\nTicket ID: {jira_data.get('ticket_id')}")
    print(f"Module: {jira_data.get('module')}")
    print(f"Title: {jira_data.get('title')}")
    print(f"Description: {jira_data.get('description')}")
    print(f"\nSteps ({len(jira_data.get('steps', []))}):")
    for step in jira_data.get('steps', []):
        print(f"  {step['num']}. {step['text']}")

    print(f"\nAcceptance Criteria:")
    print(f"  {jira_data.get('acceptance_criteria')}")

    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION:")
    print("=" * 70)

    assert jira_data.get('ticket_id') == 'RBPLCD-8835', "Ticket ID mismatch"
    print("✓ Ticket ID correct")

    assert jira_data.get('module') == 'Teststep', "Module mismatch"
    print("✓ Module correct")

    assert len(jira_data.get('steps', [])) == 8, "Step count mismatch"
    print("✓ Step count correct (8 steps)")

    assert jira_data.get('acceptance_criteria'), "No acceptance criteria"
    print("✓ Acceptance criteria extracted")

    # Check state fields
    assert updated_state.get('module') == 'Teststep', "State module not set"
    print("✓ State module field set")

    assert updated_state.get('test_title'), "State test_title not set"
    print("✓ State test_title field set")

    assert updated_state.get('steps'), "State steps not set"
    print("✓ State steps field set")

    print("\n" + "=" * 70)
    print("🎉 JIRA PARSER AGENT TEST PASSED!")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
