"""
PLCD Testing Assistant - Integration Test
Tests all components working together without browser execution
"""

from config_loader import load_config
from jira_parser import JiraTicketParser
from agent1_selector_discovery import Agent1SelectorDiscovery


def test_integration():
    """Test integration of all components"""

    print("=" * 80)
    print("PLCD Testing Assistant - Integration Test")
    print("=" * 80)

    # Step 1: Load configuration
    print("\n[1/4] Loading configuration...")
    config = load_config()
    print(f"[OK] Configuration loaded")
    print(f"     Azure endpoint: {config['azure_openai']['endpoint']}")
    print(f"     ChromaDB path: {config['vector_database']['persist_directory']}")

    # Step 2: Parse Jira ticket
    print("\n[2/4] Parsing Jira ticket...")
    parser = JiraTicketParser(config['folders']['jira'])
    ticket_data = parser.parse_ticket("RBPLCD-8835")
    print(f"[OK] Ticket: {ticket_data['title']}")
    print(f"     Module: {ticket_data['module']}")
    print(f"     Steps: {len(ticket_data['steps'])}")

    # Step 3: Initialize Agent 1
    print("\n[3/4] Initializing Agent 1...")
    agent1 = Agent1SelectorDiscovery(config)
    print(f"[OK] Agent 1 initialized")
    print(f"     Collection: {agent1.collection.name}")
    print(f"     Selectors: {agent1.collection.count()}")

    # Step 4: Test selector discovery for each step
    print("\n[4/4] Testing selector discovery for all steps...")
    print("-" * 80)

    results = []
    for step in ticket_data['steps']:
        step_number = step['step_number']
        step_text = step['step_text']

        print(f"\nStep {step_number}: {step_text}")

        result = agent1.discover_selector(
            step_text=step_text,
            current_module=ticket_data['module']
        )

        if result['selector_result']:
            selector_result = result['selector_result']
            print(f"  [OK] {selector_result['selector']}")
            print(f"       Confidence: {selector_result['confidence']:.3f}")
            print(f"       Agent: {selector_result['agent_used']}")
            print(f"       Module: {selector_result['metadata']['module']}")

            results.append({
                "step": step_number,
                "text": step_text,
                "selector": selector_result['selector'],
                "confidence": selector_result['confidence'],
                "success": True
            })
        else:
            print(f"  [WARNING] No selector found")
            results.append({
                "step": step_number,
                "text": step_text,
                "success": False
            })

    # Summary
    print("\n" + "-" * 80)
    print("\nIntegration Test Summary:")
    print("=" * 80)

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nTotal Steps: {total}")
    print(f"Successful Discoveries: {successful}")
    print(f"Failed Discoveries: {total - successful}")

    if successful > 0:
        avg_confidence = sum(r['confidence'] for r in results if r['success']) / successful
        print(f"Average Confidence: {avg_confidence:.3f}")

    print("\nDetailed Results:")
    for r in results:
        status = "[OK]" if r['success'] else "[FAILED]"
        if r['success']:
            print(f"  {status} Step {r['step']}: {r['selector']} (conf: {r['confidence']:.3f})")
        else:
            print(f"  {status} Step {r['step']}: No selector found")

    print("\n" + "=" * 80)

    if successful == total:
        print("[OK] All components integrated successfully!")
    elif successful > total // 2:
        print("[WARNING] Partial integration - some selectors not found")
    else:
        print("[ERROR] Integration test failed - most selectors not found")

    print("=" * 80)

    return results


if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
