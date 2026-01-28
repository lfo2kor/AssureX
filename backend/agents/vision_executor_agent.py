"""
Vision Executor Agent

Executes test steps using GPT-4o vision and Playwright browser automation.
Uses coordinate-based clicking (no selectors) guided by AI vision.

Components:
- Browser setup and initialization
- Auto-login with vision guidance
- Test step execution with retry logic
- Screenshot capture and video recording
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from utils.vision_helper import AzureVisionClient
from utils.selector_loader_v2 import SelectorLoaderV2
from utils.step_executor import StepExecutor
from models.state import TestAutomationState


def vision_executor_agent(state: TestAutomationState) -> TestAutomationState:
    """
    Main vision executor agent function.

    Orchestrates entire test execution:
    1. Initialize browser
    2. Auto-login
    3. Execute all test steps with retry logic
    4. Record video
    5. Capture screenshots
    6. Update state with results

    Args:
        state: Current workflow state with config and jira_data

    Returns:
        Updated state with execution results
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("=" * 70)
    logger.info("VISION EXECUTOR AGENT - Starting")
    logger.info("=" * 70)

    config = state['config']
    jira_data = state.get('jira_data', {})
    steps = jira_data.get('steps', [])

    if not steps:
        raise ValueError("No test steps found in jira_data")

    logger.info(f"Executing {len(steps)} test steps")

    # Initialize vision client
    vision_client = AzureVisionClient(config, logger)

    # Initialize selector loader V2 with sequential context tracking
    selector_loader = SelectorLoaderV2(
        selectors_file="Selectors_Folder/selectors_merged_runtime_fixed.json",
        use_sequential_context=True
    )

    # Reset state for new test
    selector_loader.reset_state()

    # Get module context from Jira data
    module = jira_data.get('module', '')
    logger.info(f"Module context: {module}")

    # Initialize results tracking
    execution_results = []
    screenshots = []
    errors = []
    execution_start_time = datetime.now()

    playwright = None
    browser = None
    context = None
    page = None

    try:
        # Initialize browser
        logger.info("Initializing browser...")
        playwright, browser, context, page = initialize_browser(config, logger)
        logger.info("Browser initialized successfully")

        # Auto-login
        logger.info("Performing auto-login...")
        login_success = auto_login(page, config, vision_client, logger)

        if not login_success:
            raise Exception("Auto-login failed")

        logger.info("Login successful")

        # Initialize browser context in state
        state['browser_context'] = {
            'current_page': 'main',
            'last_action': 'login'
        }

        # Initialize step executor with 3-level selector strategy
        step_executor = StepExecutor(
            page=page,
            vision_client=vision_client,
            selector_loader=selector_loader,
            config=config,
            logger=logger,
            module=module
        )

        # Execute each test step (skip step 1 "Login" since already done)
        for step in steps:
            # Skip login step since we already did auto-login
            if step['num'] == 1 and 'login' in step['text'].lower():
                logger.info(f"Skipping Step 1 (Login) - already completed in auto_login()")
                # Add a PASSED result for login step
                execution_results.append({
                    'step_num': 1,
                    'step_text': step['text'],
                    'status': 'PASSED',
                    'selector_used': 'auto_login()',
                    'level_used': 'Built-in login',
                    'confidence': 1.0,
                    'execution_time': 0.0,
                    'screenshot_before': '',
                    'screenshot_after': '',
                    'error': ''
                })
                continue

            logger.info(f"\n{'=' * 70}")
            logger.info(f"Executing Step {step['num']}: {step['text']}")
            logger.info(f"{'=' * 70}")

            state['current_step'] = step['num']

            # Execute step using 3-level selector strategy
            result = step_executor.execute_step(step)

            execution_results.append(result)

            # Collect screenshots
            if result.get('screenshot_before'):
                screenshots.append(result['screenshot_before'])
            if result.get('screenshot_after'):
                screenshots.append(result['screenshot_after'])

            # Check if step failed
            if result['status'] == 'FAILED':
                logger.warning(f"Step {step['num']} failed")
                logger.warning(f"Error: {result.get('error', 'Unknown error')}")
                errors.append({
                    'step_num': step['num'],
                    'error': result.get('error', 'Unknown error')
                })

                # TERMINATE TEST IMMEDIATELY if all 3 levels failed
                if result.get('error') == 'All 3 selector levels (L1, L2, L3) failed':
                    logger.error("")
                    logger.error("="*80)
                    logger.error("TEST EXECUTION TERMINATED")
                    logger.error("="*80)
                    logger.error(f"Reason: Step {step['num']} failed at all selector levels")
                    logger.error(f"Screenshot: {result.get('screenshot_after', 'N/A')}")
                    logger.error("")
                    logger.error("Please fix the issue above and restart the test.")
                    logger.error("="*80)

                    # Raise exception to stop execution
                    raise Exception(f"Test terminated at Step {step['num']}: {result.get('error')}")

            # Update browser context
            state['browser_context']['last_action'] = step['text']

            logger.info(f"Step {step['num']} status: {result['status']}")
            logger.info(f"Selector used: {result.get('selector_used', 'N/A')}")
            logger.info(f"Level used: {result.get('level_used', 'N/A')}")

    except Exception as e:
        logger.error(f"Fatal error during execution: {e}")
        import traceback
        traceback.print_exc()
        errors.append({'fatal_error': str(e)})

    finally:
        # Record execution end time
        execution_end_time = datetime.now()
        total_execution_time = (execution_end_time - execution_start_time).total_seconds()

        logger.info(f"\nTotal execution time: {total_execution_time:.2f} seconds")

        # Save video if recording
        video_path = ""
        if context and config['execution'].get('record_video', True):
            try:
                video_path = save_video(context, config, state.get('ticket_number', 'unknown'), logger)
                logger.info(f"Video saved: {video_path}")
            except Exception as e:
                logger.error(f"Error saving video: {e}")

        # Close browser
        if context:
            context.close()
        if browser:
            browser.close()
        if playwright:
            playwright.stop()

        logger.info("Browser closed")

    # Determine overall status
    failed_steps = [r for r in execution_results if r['status'] == 'FAILED']
    overall_status = 'FAILED' if failed_steps else 'PASSED'

    # Update state
    state['execution_results'] = execution_results
    state['screenshots'] = screenshots
    state['execution_start_time'] = execution_start_time.isoformat()
    state['execution_end_time'] = execution_end_time.isoformat()
    state['total_execution_time'] = total_execution_time
    state['overall_status'] = overall_status
    state['video_path'] = video_path
    state['errors'] = errors

    logger.info("=" * 70)
    logger.info(f"VISION EXECUTOR AGENT - Complete")
    logger.info(f"Overall Status: {overall_status}")
    logger.info(f"Steps Passed: {len(execution_results) - len(failed_steps)}/{len(execution_results)}")
    logger.info("=" * 70)

    return state


def initialize_browser(
    config: Dict,
    logger: logging.Logger
) -> Tuple[any, Browser, BrowserContext, Page]:
    """
    Initialize Playwright browser with video recording.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Tuple of (playwright, browser, context, page)
    """
    playwright = sync_playwright().start()

    # Select browser
    browser_type = config.get('browser', 'edge').lower()
    if browser_type == 'edge':
        browser = playwright.chromium.launch(
            channel='msedge',
            headless=config['execution'].get('headless', False)
        )
    elif browser_type == 'chromium':
        browser = playwright.chromium.launch(headless=config['execution'].get('headless', False))
    elif browser_type == 'firefox':
        browser = playwright.firefox.launch(headless=config['execution'].get('headless', False))
    else:
        browser = playwright.chromium.launch(headless=config['execution'].get('headless', False))

    logger.info(f"Launched {browser_type} browser")

    # Create context with video recording
    context_options = {
        'viewport': {'width': 1920, 'height': 1080}
    }

    if config['execution'].get('record_video', True):
        video_folder = Path(config['folders']['videos'])
        video_folder.mkdir(parents=True, exist_ok=True)
        context_options['record_video_dir'] = str(video_folder)
        context_options['record_video_size'] = {'width': 1920, 'height': 1080}

    context = browser.new_context(**context_options)
    page = context.new_page()

    # Navigate to web URL
    web_url = config.get('web_url', '')
    logger.info(f"Navigating to: {web_url}")
    page.goto(web_url, wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(config['wait_times']['page_load'])

    return playwright, browser, context, page


def auto_login(
    page: Page,
    config: Dict,
    vision_client: AzureVisionClient,
    logger: logging.Logger
) -> bool:
    """
    Perform auto-login using CV-guided selector-based element detection.

    Uses Computer Vision to identify the BEST CSS selectors for login elements,
    then uses Playwright's selector-based methods (NOT coordinate clicking).

    Args:
        page: Playwright page object
        config: Configuration dictionary
        vision_client: Azure vision client
        logger: Logger instance

    Returns:
        True if login successful, False otherwise
    """
    try:
        # Take screenshot of login page
        screenshot = page.screenshot()
        logger.info("Captured login page screenshot")

        # Get credentials
        username = config['login']['username']
        password = config['login']['password']

        # METHOD 1: Try standard HTML selectors first (fast, no CV needed)
        logger.info("Attempting login with standard selectors...")
        standard_selectors_worked = False

        try:
            # Try common standard selectors
            username_count = page.locator('input[type="text"]').count()
            password_count = page.locator('input[type="password"]').count()
            button_count = page.locator('button[type="submit"]').count()

            if username_count > 0 and password_count > 0 and button_count > 0:
                logger.info(f"Found standard elements: {username_count} text input(s), {password_count} password input(s), {button_count} submit button(s)")

                # Fill username
                page.locator('input[type="text"]').first.fill(username)
                logger.info("Username filled using standard selector")

                # Fill password
                page.locator('input[type="password"]').first.fill(password)
                logger.info("Password filled using standard selector")

                # Click submit button
                page.locator('button[type="submit"]').first.click()
                logger.info("Login button clicked using standard selector")

                standard_selectors_worked = True
            else:
                logger.info("Standard selectors not found, will use CV-guided approach")

        except Exception as e:
            logger.info(f"Standard selectors failed: {e}, falling back to CV-guided selectors")

        # METHOD 2: Use CV to identify SPECIFIC selectors (if standard didn't work)
        if not standard_selectors_worked:
            logger.info("Using CV to identify specific selectors...")

            # Call CV to get best selectors
            selector_result = vision_client.identify_login_selectors(screenshot)
            logger.info(f"CV identified selectors with confidence: {selector_result.get('confidence', 0)}")
            logger.info(f"CV reasoning: {selector_result.get('reasoning', 'N/A')}")

            # Extract selectors
            username_selector = selector_result.get('username_selector', '')
            password_selector = selector_result.get('password_selector', '')
            button_selector = selector_result.get('button_selector', '')

            # Get fallback selectors
            username_fallback = selector_result.get('username_fallback', 'input[type="text"]')
            password_fallback = selector_result.get('password_fallback', 'input[type="password"]')
            button_fallback = selector_result.get('button_fallback', 'button[type="submit"]')

            logger.info(f"Primary selectors - Username: {username_selector}, Password: {password_selector}, Button: {button_selector}")

            # Try primary selectors first
            try:
                # Fill username
                if page.locator(username_selector).count() > 0:
                    page.locator(username_selector).first.fill(username)
                    logger.info(f"Username filled using CV selector: {username_selector}")
                else:
                    logger.warning(f"Primary username selector not found, trying fallback: {username_fallback}")
                    page.locator(username_fallback).first.fill(username)
                    logger.info(f"Username filled using fallback selector")

                # Fill password
                if page.locator(password_selector).count() > 0:
                    page.locator(password_selector).first.fill(password)
                    logger.info(f"Password filled using CV selector: {password_selector}")
                else:
                    logger.warning(f"Primary password selector not found, trying fallback: {password_fallback}")
                    page.locator(password_fallback).first.fill(password)
                    logger.info(f"Password filled using fallback selector")

                # Click button
                if page.locator(button_selector).count() > 0:
                    page.locator(button_selector).first.click()
                    logger.info(f"Login button clicked using CV selector: {button_selector}")
                else:
                    logger.warning(f"Primary button selector not found, trying fallback: {button_fallback}")
                    page.locator(button_fallback).first.click()
                    logger.info(f"Login button clicked using fallback selector")

            except Exception as e:
                logger.error(f"CV-guided selectors failed: {e}")
                # Last resort: press Enter
                logger.info("Trying Enter key as last resort...")
                page.keyboard.press('Enter')

        # Wait for navigation
        page.wait_for_timeout(config['wait_times']['after_login'])

        # Verify login success by checking if URL changed
        current_url = page.url
        logger.info(f"Current URL after login: {current_url}")

        # Check if we're still on login page
        if 'login' in current_url.lower():
            logger.warning("Still on login page - login may have failed")
            # Take screenshot for debugging
            page.screenshot(path='login_failed_debug.png')
            logger.info("Debug screenshot saved: login_failed_debug.png")
            return False

        logger.info("Login successful - URL changed from login page")
        return True

    except Exception as e:
        logger.error(f"Auto-login error: {e}")
        import traceback
        traceback.print_exc()
        return False


def execute_with_retry(
    step: Dict,
    page: Page,
    vision_client: AzureVisionClient,
    config: Dict,
    state: TestAutomationState,
    logger: logging.Logger
) -> Dict:
    """
    Execute a test step with retry logic.

    Attempts execution up to max_retries times with prompt refinement.

    Args:
        step: Step dictionary with 'num' and 'text'
        page: Playwright page object
        vision_client: Azure vision client
        config: Configuration dictionary
        state: Current workflow state
        logger: Logger instance

    Returns:
        Execution result dictionary
    """
    max_retries = config['execution'].get('max_retries', 3)

    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}")

        result = execute_test_step(
            step=step,
            page=page,
            vision_client=vision_client,
            config=config,
            state=state,
            attempt=attempt,
            logger=logger
        )

        # Check if successful and confident
        if result['status'] == 'PASSED' and result.get('confidence', 0) >= 0.85:
            result['retries'] = attempt - 1
            return result

        if attempt < max_retries:
            logger.warning(f"Step failed or low confidence. Retrying...")
            time.sleep(1)  # Brief pause before retry

    # Max retries reached
    result['status'] = 'FAILED'
    result['retries'] = max_retries
    logger.error(f"Step failed after {max_retries} attempts")

    return result


def execute_test_step(
    step: Dict,
    page: Page,
    vision_client: AzureVisionClient,
    config: Dict,
    state: TestAutomationState,
    attempt: int,
    logger: logging.Logger
) -> Dict:
    """
    Execute a single test step using vision-guided automation.

    Args:
        step: Step dictionary with 'num' and 'text'
        page: Playwright page object
        vision_client: Azure vision client
        config: Configuration dictionary
        state: Current workflow state
        attempt: Current attempt number (for prompt refinement)
        logger: Logger instance

    Returns:
        Execution result dictionary with:
            - step_num, step_text, status, coordinates, confidence,
              execution_time, screenshot_before, screenshot_after, error
    """
    step_start_time = time.time()
    result = {
        'step_num': step['num'],
        'step_text': step['text'],
        'status': 'FAILED',
        'coordinates': {},
        'confidence': 0.0,
        'execution_time': 0.0,
        'screenshot_before': '',
        'screenshot_after': '',
        'error': ''
    }

    try:
        # Take screenshot before action
        screenshots_folder = Path(config['folders']['reports']) / 'screenshots'
        screenshots_folder.mkdir(parents=True, exist_ok=True)

        screenshot_before_path = screenshots_folder / f"step_{step['num']}_before.png"
        screenshot_before = page.screenshot(path=str(screenshot_before_path))
        result['screenshot_before'] = str(screenshot_before_path)
        logger.debug(f"Screenshot before: {screenshot_before_path}")

        # Build context for vision API
        context = {
            'module': state.get('module', ''),
            'current_page': state.get('browser_context', {}).get('current_page', ''),
            'previous_action': state.get('browser_context', {}).get('last_action', ''),
            'attempt': attempt
        }

        # Call vision API with context
        vision_result = vision_client.call_vision_with_context(
            screenshot=screenshot_before,
            task=step['text'],
            context=context
        )

        logger.info(f"Vision result: {vision_result}")

        # Extract action details
        action_type = vision_result.get('action_type', 'click')
        coordinates = vision_result.get('coordinates', {})
        value = vision_result.get('value', '')
        confidence = vision_result.get('confidence', 0.0)

        result['coordinates'] = coordinates
        result['confidence'] = confidence

        logger.info(f"Action: {action_type} at ({coordinates.get('x')}, {coordinates.get('y')}) with confidence {confidence}")

        # Execute action
        if action_type == 'click':
            # Try clicking with mouse
            page.mouse.click(coordinates['x'], coordinates['y'])
            page.wait_for_timeout(config['wait_times']['after_click'])

            # For button clicks, also try pressing Enter as fallback
            if 'button' in step['text'].lower() or 'btn' in step['text'].lower():
                logger.debug("Button detected - pressing Enter as fallback")
                page.keyboard.press('Enter')
                page.wait_for_timeout(500)

        elif action_type == 'type':
            page.mouse.click(coordinates['x'], coordinates['y'])
            page.wait_for_timeout(config['wait_times']['after_click'])
            page.keyboard.type(value)
            page.wait_for_timeout(config['wait_times']['after_type'])

        elif action_type == 'dropdown':
            page.mouse.click(coordinates['x'], coordinates['y'])
            page.wait_for_timeout(config['wait_times']['after_dropdown'])
            # For dropdown, value might contain the option to select
            if value:
                # Simplified: just type the value (in real scenario, might need more logic)
                page.keyboard.type(value)
                page.wait_for_timeout(config['wait_times']['after_type'])
                page.keyboard.press('Enter')

        # Take screenshot after action
        screenshot_after_path = screenshots_folder / f"step_{step['num']}_after.png"
        page.screenshot(path=str(screenshot_after_path))
        result['screenshot_after'] = str(screenshot_after_path)
        logger.debug(f"Screenshot after: {screenshot_after_path}")

        # Mark as passed
        result['status'] = 'PASSED'

    except Exception as e:
        logger.error(f"Error executing step: {e}")
        result['error'] = str(e)
        result['status'] = 'FAILED'

    finally:
        execution_time = time.time() - step_start_time
        result['execution_time'] = execution_time
        logger.info(f"Step execution time: {execution_time:.2f}s")

    return result


def save_video(
    context: BrowserContext,
    config: Dict,
    ticket_id: str,
    logger: logging.Logger
) -> str:
    """
    Save the recorded video.

    Args:
        context: Browser context with video recording
        config: Configuration dictionary
        ticket_id: Ticket ID for filename
        logger: Logger instance

    Returns:
        Path to saved video file
    """
    try:
        # Close context to finalize video
        video = context.pages[0].video
        if video:
            video_path = video.path()
            logger.info(f"Video recorded at: {video_path}")
            return video_path
        else:
            logger.warning("No video recording found")
            return ""
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        return ""
