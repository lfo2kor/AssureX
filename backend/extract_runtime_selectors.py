"""
Runtime Selector Extraction Script

Runs a test in "extraction mode" to capture selectors from the running application.

Usage:
    python extract_runtime_selectors.py RBPLCD-8835
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright
from agents.jira_parser_agent import jira_parser_agent
from agents.vision_executor_agent import auto_login, initialize_browser
from utils.runtime_selector_extractor import RuntimeSelectorExtractor
from utils.step_executor import StepExecutor
from utils.vision_helper import AzureVisionClient
from utils.selector_loader import SelectorLoader
from utils.module_mapper import ModuleMapper
from models.state import TestAutomationState
import yaml


def setup_logger(ticket_id: str) -> logging.Logger:
    """Setup logger for extraction."""
    logger = logging.getLogger("RuntimeExtraction")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = Path("Logs") / f"runtime_extraction_{ticket_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def extract_target_text(step_text: str) -> str:
    """
    Extract target text from step description.

    Examples:
        "click on save" → "save"
        "open parts accordion" → "parts"
        "Click on Type from mandatory field" → "Type"
        "navigate to teststep" → "teststep"
    """
    import re

    step_lower = step_text.lower()

    # Check for quoted text first
    quoted_match = re.search(r'["\']([^"\']+)["\']', step_text)
    if quoted_match:
        return quoted_match.group(1)

    # Extract key nouns/verbs
    if 'accordion' in step_lower:
        # "open parts accordion" → "parts"
        match = re.search(r'(\w+)\s+accordion', step_lower)
        if match:
            return match.group(1)

    if 'navigate' in step_lower or 'go to' in step_lower:
        # "navigate to teststep" → "teststep"
        match = re.search(r'(?:navigate to|go to)\s+(\w+)', step_lower)
        if match:
            return match.group(1)

    if 'click on' in step_lower:
        # "click on save" → "save"
        match = re.search(r'click on\s+(\w+)', step_lower)
        if match:
            return match.group(1)

    if 'select' in step_lower:
        # "select Type from dropdown" → "Type"
        match = re.search(r'select\s+(\w+)', step_lower)
        if match:
            return match.group(1)

    # Fallback: first significant word
    words = step_text.split()
    for word in words:
        if len(word) > 3 and word.lower() not in ['click', 'open', 'from', 'the', 'and']:
            return word

    return ""


def extract_selectors_from_test(ticket_id: str):
    """
    Main extraction function.

    Args:
        ticket_id: Jira ticket ID (e.g., "RBPLCD-8835")
    """
    logger = setup_logger(ticket_id)

    logger.info("=" * 80)
    logger.info(f"   Runtime Selector Extraction for {ticket_id}")
    logger.info("=" * 80)

    # Load config
    config_file = "plcdtest_config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Parse Jira ticket
    logger.info("\n[1/5] Parsing Jira ticket...")
    state = TestAutomationState(config=config, ticket_number=ticket_id)
    state = jira_parser_agent(state)
    jira_data = state['jira_data']

    steps = jira_data.get('steps', [])
    module = jira_data.get('module', '')

    logger.info(f"  Ticket: {ticket_id}")
    logger.info(f"  Module: {module}")
    logger.info(f"  Steps: {len(steps)}")

    # Initialize browser
    logger.info("\n[2/5] Initializing browser...")
    playwright = None
    browser = None
    context = None
    page = None

    try:
        playwright, browser, context, page = initialize_browser(config, logger)

        # Initialize components
        vision_client = AzureVisionClient(config, logger)

        # Use SelectorLoaderV2 for extraction (has update_state_for_step method)
        from utils.selector_loader_v2 import SelectorLoaderV2
        selector_loader = SelectorLoaderV2(
            selectors_file="Selectors_Folder/selectors_enriched_all_modules.json",
            use_sequential_context=False  # Don't need context for extraction
        )

        module_mapper = ModuleMapper(config)

        # Initialize runtime extractor
        extractor = RuntimeSelectorExtractor(page, config, logger)

        # Initialize step executor
        step_executor = StepExecutor(
            page=page,
            vision_client=vision_client,
            selector_loader=selector_loader,
            config=config,
            logger=logger,
            module=module
        )

        # Auto-login
        logger.info("\n[3/5] Performing auto-login...")
        login_success = auto_login(page, config, vision_client, logger)

        if not login_success:
            raise Exception("Auto-login failed")

        logger.info("  Login successful")

        # Execute steps and extract selectors
        logger.info(f"\n[4/5] Executing {len(steps)} steps and extracting selectors...")

        for step in steps:
            step_num = step['num']
            step_text = step['text']

            # Skip login step
            if step_num == 1 and 'login' in step_text.lower():
                logger.info(f"\nStep {step_num}: {step_text} [SKIPPED - login already done]")
                continue

            logger.info(f"\nStep {step_num}: {step_text}")

            # Prepare step context for extraction
            step_context = {
                'step_num': step_num,
                'step_text': step_text,
                'module': module,
                'action': 'unknown'
            }

            # Extract target text (what we're looking for)
            target_text = extract_target_text(step_text)
            if target_text:
                logger.info(f"  Target text: '{target_text}'")

            # Extract selectors BEFORE executing step (current page state)
            logger.info(f"  Extracting selectors from page...")
            selectors_before = extractor.extract_from_current_page(
                step_context,
                target_text=target_text
            )

            # Execute the step using existing StepExecutor
            logger.info(f"  Executing step...")
            result = step_executor.execute_step(step)

            if result['status'] == 'PASSED':
                logger.info(f"  ✅ Step passed using: {result['selector_used']} (Level: {result['level_used']})")

                # If L2 succeeded, extract selector from the element we just clicked
                if 'Level 2' in result['level_used']:
                    logger.info(f"  Learning from L2 success...")
                    learned_selector = extractor.extract_specific_element(
                        result['selector_used'],
                        step_context
                    )
                    if learned_selector:
                        logger.info(f"    ✅ Learned: [{learned_selector['attr']}=\"{learned_selector['value']}\"]")
            else:
                logger.warning(f"  ❌ Step failed: {result.get('error', 'Unknown error')}")

            # Wait for page to stabilize
            page.wait_for_timeout(config['wait_times']['after_click'])

        # Save extracted selectors
        logger.info(f"\n[5/5] Saving extracted selectors...")
        output_file = f"Selectors_Folder/runtime_selectors_{ticket_id}.json"
        extractor.save_to_json(output_file)

        logger.info(f"\n" + "=" * 80)
        logger.info(f"   Extraction Complete!")
        logger.info(f"=" * 80)
        logger.info(f"Total selectors extracted: {extractor.get_extracted_count()}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review: {output_file}")
        logger.info(f"  2. Merge with existing selectors: python merge_selectors.py")
        logger.info(f"  3. Re-test with merged selectors: python run_test.py {ticket_id}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise

    finally:
        if browser:
            browser.close()
        if playwright:
            playwright.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_runtime_selectors.py TICKET_ID")
        print("Example: python extract_runtime_selectors.py RBPLCD-8835")
        sys.exit(1)

    ticket_id = sys.argv[1]
    extract_selectors_from_test(ticket_id)
