"""
Pytest configuration for generated Playwright tests
Browser: edge
"""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="function")
def page():
    """
    Playwright page fixture - creates browser and page for each test
    """
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(
            channel='msedge',
            headless=False,
            args=['--start-maximized']
        )

        # Create context without viewport to allow maximized window
        context = browser.new_context(no_viewport=True)

        # Create page
        page = context.new_page()

        # Set default timeout
        page.set_default_timeout(30000)

        yield page

        # Cleanup
        context.close()
        browser.close()
