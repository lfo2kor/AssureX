"""
PLCD Testing Assistant - Main Orchestrator
Executes automated tests using AI-powered selector discovery
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from playwright.sync_api import sync_playwright, Page, Browser

from config_loader import load_config
from jira_parser import JiraTicketParser
from agent1_selector_discovery import Agent1SelectorDiscovery
from agent2_dom_discovery import Agent2DOMDiscovery
from report_generator import generate_html_report
from script_generator import generate_playwright_script, generate_pytest_config, generate_readme
from learning_system import LearningSystem


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/plcd_ta.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestExecutionState:
    """Holds state during test execution"""

    def __init__(self, ticket_id: str, config: Dict):
        self.ticket_id = ticket_id
        self.config = config
        self.ticket_data = None
        self.current_step = 0
        self.step_results = []
        self.overall_status = "PASSED"
        self.execution_start_time = time.time()
        self.page: Optional[Page] = None
        self.browser: Optional[Browser] = None


class PLCDTestingAssistant:
    """
    Main orchestrator for PLCD test automation
    """

    def __init__(self, config: Dict):
        """
        Initialize testing assistant

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.jira_parser = JiraTicketParser(config['folders']['jira'])
        self.agent1 = Agent1SelectorDiscovery(config)
        self.agent2 = Agent2DOMDiscovery(config)
        self.learning_system = LearningSystem(config)

        logger.info("PLCD Testing Assistant initialized")


    def execute_test(self, ticket_id: str) -> Dict:
        """
        Execute test for a Jira ticket

        Args:
            ticket_id: Jira ticket ID (e.g., "RBPLCD-8835")

        Returns:
            Execution results dictionary
        """
        print("\n" + "=" * 80)
        print(f"PLCD Testing Assistant - Executing Test: {ticket_id}")
        print("=" * 80)

        # Initialize state
        state = TestExecutionState(ticket_id, self.config)

        try:
            # Step 1: Parse Jira ticket
            state.ticket_data = self._parse_ticket(state)

            # Step 2: Initialize browser
            with sync_playwright() as playwright:
                state.browser = self._init_browser(playwright, state)

                # Create context without viewport to allow maximized window
                context = state.browser.new_context(
                    no_viewport=True  # Allow browser to use full screen
                )
                state.page = context.new_page()

                # Step 3: Login
                self._login(state)

                # Step 3.5: Navigate to module page
                self._navigate_to_module(state)

                # Step 4: Execute test steps
                self._execute_steps(state)

                # Step 5: Cleanup
                state.page.close()
                state.browser.close()

            # Step 6: Generate summary
            results = self._generate_summary(state)

            print("\n" + "=" * 80)
            print("Test Execution Complete!")
            print("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            state.overall_status = "FAILED"
            print(f"\n[ERROR] Test execution failed: {e}")
            return self._generate_summary(state)


    def _parse_ticket(self, state: TestExecutionState) -> Dict:
        """Parse Jira ticket"""
        print("\n[1/5] Parsing Jira ticket...")
        ticket_data = self.jira_parser.parse_ticket(state.ticket_id)
        print(f"[OK] Ticket: {ticket_data['title']}")
        print(f"[OK] Module: {ticket_data['module']}")
        print(f"[OK] Steps: {len(ticket_data['steps'])}")
        return ticket_data


    def _init_browser(self, playwright, state: TestExecutionState) -> Browser:
        """Initialize Playwright browser with maximized viewport"""
        print("\n[2/5] Initializing browser...")

        browser_type = self.config['browser']
        headless = self.config['execution']['headless']

        if browser_type == 'edge':
            browser = playwright.chromium.launch(
                headless=headless,
                channel='msedge',
                args=['--start-maximized']
            )
        elif browser_type == 'chrome':
            browser = playwright.chromium.launch(
                headless=headless,
                args=['--start-maximized']
            )
        else:
            browser = playwright.chromium.launch(
                headless=headless,
                args=['--start-maximized']
            )

        print(f"[OK] Browser: {browser_type} (headless: {headless})")
        return browser


    def _login(self, state: TestExecutionState):
        """Login to application using Agent 1 for button discovery"""
        print("\n[3/5] Logging in...")

        page = state.page
        web_url = self.config['web_url']
        username = self.config['login']['username']
        password = self.config['login']['password']

        # Navigate to login page
        page.goto(web_url)
        page.wait_for_load_state('networkidle')

        # Wait for page to render
        page.wait_for_timeout(2000)

        try:
            # Fill username (use .first to get first matching element)
            page.locator('input[type="text"]').first.fill(username)
            logger.info("Username filled")

            # Fill password
            page.locator('input[type="password"]').first.fill(password)
            logger.info("Password filled")

            # Use Agent 1 to discover login button selector
            print("  Discovering login button selector...")
            result = self.agent1.discover_selector(
                step_text="Click login button",
                current_module="Auth"
            )

            if result['selector_result']:
                login_button_selector = result['selector_result']['selector']
                confidence = result['selector_result']['confidence']
                print(f"  Found login button: {login_button_selector} (conf: {confidence:.2f})")

                page.click(login_button_selector)
                logger.info(f"Login button clicked: {login_button_selector}")
            else:
                # Fallback: try any button
                print("  Using fallback selector: button")
                page.locator('button').first.click()
                logger.info("Login button clicked using fallback")

            # Wait for login to complete
            page.wait_for_timeout(self.config['wait_times']['after_login'])

            print(f"[OK] Logged in as: {username}")

        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise Exception(f"Login failed: {e}. Check if login page structure has changed.")


    def _navigate_to_module(self, state: TestExecutionState):
        """Navigate to the module page based on ticket module"""
        module = state.ticket_data.get('module', '')

        if not module:
            print("\n[3.5/5] Skipping module navigation (no module specified)")
            return

        # Check if first step is already a navigation step
        steps = state.ticket_data.get('steps', [])
        if steps and len(steps) > 0:
            first_step_text = steps[0]['step_text'].lower()
            if 'navigate' in first_step_text or 'go to' in first_step_text:
                print(f"\n[3.5/5] Skipping auto-navigation (Step 1 handles navigation)")
                return

        print(f"\n[3.5/5] Navigating to module: {module}...")

        page = state.page

        try:
            # Map module names to navigation text
            module_nav_map = {
                'Teststep': 'Navigate to Runs',
                'Tests': 'Navigate to Tasks',
                'Parts': 'Navigate to Parts',
                'Equipment': 'Navigate to Equipment',
                'Projects': 'Navigate to Projects',
            }

            nav_text = module_nav_map.get(module, f'Navigate to {module}')

            # Use Agent 1 to discover navigation selector
            result = self.agent1.discover_selector(
                step_text=nav_text,
                current_module='Common'
            )

            if result['selector_result']:
                nav_selector = result['selector_result']['selector']
                confidence = result['selector_result']['confidence']
                print(f"  Found navigation: {nav_selector} (conf: {confidence:.2f})")

                # Click navigation
                page.click(nav_selector)
                page.wait_for_timeout(self.config['wait_times']['after_navigation'])

                print(f"[OK] Navigated to {module}")
            else:
                print(f"[WARNING] Could not find navigation for {module}, continuing anyway...")

        except Exception as e:
            logger.error(f"Module navigation failed: {e}")
            print(f"[WARNING] Navigation failed: {e}, continuing anyway...")


    def _execute_steps(self, state: TestExecutionState):
        """Execute all test steps"""
        print("\n[4/5] Executing test steps...")
        print("-" * 80)

        total_steps = len(state.ticket_data['steps'])

        for step_data in state.ticket_data['steps']:
            step_number = step_data['step_number']
            step_text = step_data['step_text']

            print(f"\nStep {step_number}/{total_steps}: {step_text}")

            try:
                # Try runtime learned selectors first (highest priority)
                learned_selector = self.learning_system.get_learned_selector(
                    step_text=step_text,
                    module=state.ticket_data['module']
                )

                if learned_selector and learned_selector['confidence'] > 0.85:
                    # Use learned selector with high confidence
                    selector_result = learned_selector
                    confidence = learned_selector['confidence']
                    agent_used = "Runtime"
                    print(f"  [Runtime Learned: {confidence:.2f}] (from {learned_selector['original_ticket']})")
                else:
                    # Agent 1: Try semantic search
                    result = self.agent1.discover_selector(
                        step_text=step_text,
                        current_module=state.ticket_data['module']
                    )

                    selector_result = result['selector_result']
                    agent_used = "L1"

                    if selector_result is None:
                        raise Exception("No selector found by Agent 1")

                    confidence = selector_result['confidence']

                    # Confidence-based routing: L1 → L2 → L3 (only if not using runtime)
                    confidence_threshold = self.config['agent1_selector_discovery']['retrieval']['confidence_threshold']
                    retry_threshold = self.config['agent1_selector_discovery']['retrieval'].get('retry_threshold', 0.70)

                    if confidence < retry_threshold:
                        # Low confidence: Try Agent 2 (DOM Discovery)
                        logger.info(f"Low confidence ({confidence:.2f}), trying Agent 2...")
                        print(f"  [L1 Low Confidence: {confidence:.2f}] Trying Agent 2 (DOM)...")

                        agent2_result = self.agent2.discover_selector(
                            page=state.page,
                            step_text=step_text,
                            agent1_result=result
                        )

                        if agent2_result['selector_result']:
                            agent2_confidence = agent2_result['selector_result']['confidence']

                            # Use Agent 2 result if better confidence
                            if agent2_confidence > confidence:
                                selector_result = agent2_result['selector_result']
                                confidence = agent2_confidence
                                agent_used = "L2"
                                print(f"  [L2 Found Better: {confidence:.2f}]")
                            else:
                                print(f"  [L2 No Improvement: {agent2_confidence:.2f}, using L1]")

                selector = selector_result['selector']

                # Execute action
                action_type = self._detect_action_type(step_text)
                execution_success = self._execute_action(
                    state.page,
                    selector,
                    action_type,
                    step_text
                )

                # Record result
                step_result = {
                    "step_number": step_number,
                    "step_text": step_text,
                    "selector": selector,
                    "confidence": confidence,
                    "agent_used": agent_used,
                    "action_type": action_type,
                    "status": "PASSED" if execution_success else "FAILED"
                }

                state.step_results.append(step_result)

                status_icon = "[OK]" if execution_success else "[FAILED]"
                print(f"{status_icon} {selector} (conf: {confidence:.2f}, agent: {agent_used})")

                if not execution_success:
                    state.overall_status = "FAILED"

                    # Fail-fast mode: stop execution if enabled
                    if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                        print(f"\n[FAIL-FAST] Stopping execution due to step failure")
                        break

            except Exception as e:
                logger.error(f"Step {step_number} failed: {e}")
                state.step_results.append({
                    "step_number": step_number,
                    "step_text": step_text,
                    "status": "FAILED",
                    "error": str(e)
                })
                state.overall_status = "FAILED"
                print(f"[FAILED] Error: {e}")

                # Fail-fast mode: stop execution if enabled
                if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                    print(f"\n[FAIL-FAST] Stopping execution due to error")
                    break

        print("-" * 80)


    def _detect_action_type(self, step_text: str) -> str:
        """
        Detect action type from step text

        Args:
            step_text: Step description

        Returns:
            Action type: click, type, select, navigate, verify
        """
        step_lower = step_text.lower()

        if any(word in step_lower for word in ['click', 'press', 'select']):
            return 'click'
        elif any(word in step_lower for word in ['enter', 'type', 'input', 'fill']):
            return 'type'
        elif 'navigate' in step_lower or 'go to' in step_lower:
            return 'navigate'
        elif 'verify' in step_lower or 'check' in step_lower or 'wait' in step_lower:
            return 'verify'
        elif 'clear' in step_lower:
            return 'clear'
        else:
            return 'click'  # Default


    def _execute_action(
        self,
        page: Page,
        selector: str,
        action_type: str,
        step_text: str
    ) -> bool:
        """
        Execute Playwright action

        Args:
            page: Playwright page object
            selector: Element selector
            action_type: Type of action
            step_text: Original step text

        Returns:
            True if successful, False otherwise
        """
        try:
            # Wait for element
            page.wait_for_selector(
                selector,
                timeout=self.config['performance']['element_wait_timeout']
            )

            if action_type == 'click':
                page.click(selector)
                page.wait_for_timeout(self.config['wait_times']['after_click'])

            elif action_type == 'type':
                # Extract text to type (simplified - would need better parsing)
                text_to_type = self._extract_text_to_type(step_text)
                page.fill(selector, text_to_type)
                page.wait_for_timeout(self.config['wait_times']['after_type'])

            elif action_type == 'clear':
                page.fill(selector, '')
                page.wait_for_timeout(self.config['wait_times']['after_type'])

            elif action_type == 'verify':
                # Just check if element exists
                page.wait_for_timeout(self.config['wait_times']['after_click'])

            elif action_type == 'navigate':
                page.click(selector)
                page.wait_for_timeout(self.config['wait_times']['after_navigation'])

            return True

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False


    def _extract_text_to_type(self, step_text: str) -> str:
        """
        Extract text to type from step description

        Args:
            step_text: Step text

        Returns:
            Text to type (extracted from quotes or defaults)
        """
        import re

        # Try to find text in quotes
        match = re.search(r'"([^"]+)"', step_text)
        if match:
            return match.group(1)

        # Default value
        return "Test Value"


    def _generate_summary(self, state: TestExecutionState) -> Dict:
        """Generate execution summary and HTML report"""
        print("\n[5/5] Generating summary and report...")

        execution_time = time.time() - state.execution_start_time

        passed_steps = sum(1 for r in state.step_results if r.get('status') == 'PASSED')
        failed_steps = sum(1 for r in state.step_results if r.get('status') == 'FAILED')

        summary = {
            "ticket_id": state.ticket_id,
            "overall_status": state.overall_status,
            "total_steps": len(state.step_results),
            "passed_steps": passed_steps,
            "failed_steps": failed_steps,
            "execution_time": f"{execution_time:.1f}s",
            "step_results": state.step_results
        }

        # Generate HTML report
        try:
            report_path = generate_html_report(
                ticket_id=state.ticket_id,
                ticket_data=state.ticket_data,
                step_results=state.step_results,
                overall_status=state.overall_status,
                execution_time=execution_time,
                config=self.config
            )
            summary['report_path'] = report_path
            print(f"[OK] HTML Report: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            print(f"[WARNING] Report generation failed: {e}")

        # Generate Playwright test script (only if test passed or has successful steps)
        if passed_steps > 0:
            try:
                script_path = generate_playwright_script(
                    ticket_id=state.ticket_id,
                    ticket_data=state.ticket_data,
                    step_results=state.step_results,
                    config=self.config
                )
                summary['script_path'] = script_path
                print(f"[OK] Playwright Script: {script_path}")

                # Generate pytest config and README (only once)
                scripts_folder = Path(self.config['folders']['generated_scripts'])
                conftest_path = scripts_folder / "conftest.py"
                readme_path = scripts_folder / "README.md"

                if not conftest_path.exists():
                    generate_pytest_config(self.config)
                    print(f"[OK] Generated conftest.py for pytest")

                if not readme_path.exists():
                    generate_readme(self.config)
                    print(f"[OK] Generated README.md")

            except Exception as e:
                logger.error(f"Failed to generate Playwright script: {e}")
                print(f"[WARNING] Script generation failed: {e}")

        # Learn from execution (save successful selectors to runtime collection)
        if passed_steps > 0:
            try:
                learning_stats = self.learning_system.learn_from_execution(
                    ticket_id=state.ticket_id,
                    ticket_data=state.ticket_data,
                    step_results=state.step_results
                )
                summary['learning_stats'] = learning_stats
                print(f"[OK] Learned {learning_stats['selectors_learned']} selectors for future use")
            except Exception as e:
                logger.error(f"Failed to save learned selectors: {e}")
                print(f"[WARNING] Learning system failed: {e}")

        print(f"\n[OK] Summary:")
        print(f"     Status: {summary['overall_status']}")
        print(f"     Total Steps: {summary['total_steps']}")
        print(f"     Passed: {summary['passed_steps']}")
        print(f"     Failed: {summary['failed_steps']}")
        print(f"     Execution Time: {summary['execution_time']}")

        return summary


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python plcd_ta.py <TICKET_ID>")
        print("Example: python plcd_ta.py RBPLCD-8835")
        sys.exit(1)

    ticket_id = sys.argv[1]

    try:
        # Load configuration
        config = load_config()

        # Initialize assistant
        assistant = PLCDTestingAssistant(config)

        # Execute test
        results = assistant.execute_test(ticket_id)

        # Exit with appropriate code
        sys.exit(0 if results['overall_status'] == 'PASSED' else 1)

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
