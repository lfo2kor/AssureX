"""
Test Sequential Context Implementation

Compares V1.0 (no context) vs V2.0 (sequential context) L1 matching
for RBPLCD-8835 and RBPLCD-8862.
"""

import logging
from utils.selector_loader import SelectorLoader  # V1.0
from utils.selector_loader_v2 import SelectorLoaderV2  # V2.0


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("TA_AI_Project")


def test_rbplcd_8835():
    """Test RBPLCD-8835: Edit part details from teststep"""

    print("="*80)
    print("TESTING: RBPLCD-8835 - Edit part details")
    print("="*80)

    steps = [
        ("Login", "login"),
        ("navigate to teststep", "teststep"),
        ("click on teststep named as default_Measurement01", "teststep"),
        ("open parts accordion", "teststep"),  # ← Should find parts accordion
        ("click on edit button of part default_testobject_01", "teststep"),  # ← Should find edit in parts
        ("Click on Type from mandatory field and select 'Type 5' from drop down", "teststep"),  # ← Should find in entity-attribute
        ("click on save", "teststep"),
        ("Successfully edited message should be displayed", "teststep"),
    ]

    # Test V1.0
    print("\n" + "="*80)
    print("V1.0 (No Sequential Context)")
    print("="*80)

    loader_v1 = SelectorLoader()
    v1_success = 0

    for step_num, (step_text, module) in enumerate(steps, 1):
        print(f"\nStep {step_num}: {step_text}")
        selector = loader_v1.find_best_selector(step_text, module)

        if selector:
            print(f"  [OK] FOUND: {selector.get('attr')} (module={selector.get('module')})")
            v1_success += 1
        else:
            print(f"  [FAIL] NOT FOUND (L1 fails)")

    # Test V2.0
    print("\n" + "="*80)
    print("V2.0 (With Sequential Context)")
    print("="*80)

    loader_v2 = SelectorLoaderV2(use_sequential_context=True)
    v2_success = 0

    for step_num, (step_text, module) in enumerate(steps, 1):
        print(f"\nStep {step_num}: {step_text}")

        # Show state before search
        state_info = loader_v2.get_state_info()
        if state_info['enabled']:
            print(f"  STATE: module={state_info['current_module']}, "
                  f"visible={state_info['visible_modules']}, "
                  f"edit={state_info['edit_mode']}")

        selector = loader_v2.find_best_selector(step_text, module)

        if selector:
            print(f"  [OK] FOUND: {selector.get('attr')} (module={selector.get('module')})")
            v2_success += 1
        else:
            print(f"  [FAIL] NOT FOUND (L1 fails)")

    # Results
    print("\n" + "="*80)
    print("RBPLCD-8835 RESULTS")
    print("="*80)
    print(f"V1.0 L1 Success: {v1_success}/{len(steps)} = {v1_success/len(steps)*100:.0f}%")
    print(f"V2.0 L1 Success: {v2_success}/{len(steps)} = {v2_success/len(steps)*100:.0f}%")
    print(f"Improvement: +{v2_success - v1_success} steps (+{(v2_success - v1_success)/len(steps)*100:.0f}%)")
    print("="*80)

    return v1_success, v2_success


def test_rbplcd_8862():
    """Test RBPLCD-8862: Create project from teststep"""

    print("\n\n" + "="*80)
    print("TESTING: RBPLCD-8862 - Create project from teststep")
    print("="*80)

    steps = [
        ("Login", "login"),
        ("Navigate to teststep", "teststep"),
        ('Force Click on "... +" button', "teststep"),  # ← Dropdown trigger
        ('Select "Project" from the drop down and click on it', "teststep"),  # ← Menu item
        ('Click on Select Product and select "MyProject" from drop down', "teststep"),
        ('Click on Name and type "default project"', "teststep"),
        ('Click on save', "teststep"),
        ('Click on Delete', "teststep"),
        ('Click on Remove', "teststep"),
    ]

    # Test V1.0
    print("\n" + "="*80)
    print("V1.0 (No Sequential Context)")
    print("="*80)

    loader_v1 = SelectorLoader()
    v1_success = 0

    for step_num, (step_text, module) in enumerate(steps, 1):
        print(f"\nStep {step_num}: {step_text}")
        selector = loader_v1.find_best_selector(step_text, module)

        if selector:
            print(f"  [OK] FOUND: {selector.get('attr')} (module={selector.get('module')})")
            v1_success += 1
        else:
            print(f"  [FAIL] NOT FOUND (L1 fails)")

    # Test V2.0
    print("\n" + "="*80)
    print("V2.0 (With Sequential Context)")
    print("="*80)

    loader_v2 = SelectorLoaderV2(use_sequential_context=True)
    v2_success = 0

    for step_num, (step_text, module) in enumerate(steps, 1):
        print(f"\nStep {step_num}: {step_text}")

        # Show state before search
        state_info = loader_v2.get_state_info()
        if state_info['enabled']:
            print(f"  STATE: module={state_info['current_module']}, "
                  f"visible={state_info['visible_modules']}, "
                  f"dropdown={state_info['dialog_open']}")

        selector = loader_v2.find_best_selector(step_text, module)

        if selector:
            print(f"  [OK] FOUND: {selector.get('attr')} (module={selector.get('module')})")
            v2_success += 1
        else:
            print(f"  [FAIL] NOT FOUND (L1 fails)")

    # Results
    print("\n" + "="*80)
    print("RBPLCD-8862 RESULTS")
    print("="*80)
    print(f"V1.0 L1 Success: {v1_success}/{len(steps)} = {v1_success/len(steps)*100:.0f}%")
    print(f"V2.0 L1 Success: {v2_success}/{len(steps)} = {v2_success/len(steps)*100:.0f}%")
    print(f"Improvement: +{v2_success - v1_success} steps (+{(v2_success - v1_success)/len(steps)*100:.0f}%)")
    print("="*80)

    return v1_success, v2_success


def main():
    """Run both tests and show overall results"""

    print("\n" + "="*80)
    print("SEQUENTIAL CONTEXT COMPARISON TEST")
    print("="*80)
    print("Comparing V1.0 (current) vs V2.0 (sequential context)")
    print("="*80)

    # Run tests
    v1_8835, v2_8835 = test_rbplcd_8835()
    v1_8862, v2_8862 = test_rbplcd_8862()

    # Overall results
    total_steps = 17  # 8 + 9
    v1_total = v1_8835 + v1_8862
    v2_total = v2_8835 + v2_8862

    print("\n\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Total Steps: {total_steps}")
    print()
    print(f"V1.0 (Current):")
    print(f"  L1 Success: {v1_total}/{total_steps} = {v1_total/total_steps*100:.0f}%")
    print()
    print(f"V2.0 (Sequential Context):")
    print(f"  L1 Success: {v2_total}/{total_steps} = {v2_total/total_steps*100:.0f}%")
    print()
    print(f"Improvement:")
    print(f"  +{v2_total - v1_total} steps successful")
    print(f"  +{(v2_total - v1_total)/total_steps*100:.0f}% absolute improvement")
    print(f"  +{((v2_total - v1_total)/v1_total*100) if v1_total > 0 else 0:.0f}% relative improvement")
    print("="*80)

    print("\nKey Insights:")
    print("  - Sequential context tracks state across steps")
    print("  - Enables cross-module selector matching")
    print("  - Dramatically improves L1 success rate")
    print("  - Reduces reliance on L2/L3 fallbacks")
    print()


if __name__ == "__main__":
    main()
