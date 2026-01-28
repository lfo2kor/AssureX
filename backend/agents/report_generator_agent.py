"""
Report Generator Agent

Generates:
1. HTML report with embedded screenshots
2. Playwright Python script for test reproduction
"""

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
from jinja2 import Template
from models.state import TestAutomationState


def report_generator_agent(state: TestAutomationState) -> TestAutomationState:
    """
    Generate HTML report and Playwright script.

    Args:
        state: Current workflow state with execution results

    Returns:
        Updated state with report_path and script_path
    """
    logger = logging.getLogger("TA_AI_Project")
    logger.info("=" * 70)
    logger.info("REPORT GENERATOR AGENT - Starting")
    logger.info("=" * 70)

    config = state['config']
    template_path = Path("templates") / "report_template.html"

    try:
        # Generate HTML report
        logger.info("Generating HTML report...")
        report_path = generate_html_report(state, str(template_path), logger)
        logger.info(f"HTML report generated: {report_path}")

        # Generate Playwright script
        logger.info("Generating Playwright script...")
        script_path = generate_playwright_script(state, logger)
        logger.info(f"Playwright script generated: {script_path}")

        # Update state
        state['report_path'] = report_path
        state['script_path'] = script_path
        state['report_generation_status'] = 'success'

        logger.info("REPORT GENERATOR AGENT - Complete")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        state['report_generation_status'] = 'failed'
        import traceback
        traceback.print_exc()

    return state


def generate_html_report(
    state: TestAutomationState,
    template_path: str,
    logger: logging.Logger
) -> str:
    """
    Generate HTML report using Jinja2 template.

    Args:
        state: Workflow state with execution results
        template_path: Path to HTML template
        logger: Logger instance

    Returns:
        Path to generated HTML report
    """
    config = state['config']
    jira_data = state.get('jira_data', {})
    execution_results = state.get('execution_results', [])

    # Load template
    if not Path(template_path).exists():
        logger.warning(f"Template not found: {template_path}, using built-in template")
        template_content = get_builtin_template()
    else:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

    template = Template(template_content)

    # Embed screenshots as base64
    for result in execution_results:
        if result.get('screenshot_before'):
            result['screenshot_before_base64'] = embed_screenshot(result['screenshot_before'], logger)
        if result.get('screenshot_after'):
            result['screenshot_after_base64'] = embed_screenshot(result['screenshot_after'], logger)

    # Prepare context
    context = {
        'ticket_id': jira_data.get('ticket_id', 'Unknown'),
        'module': jira_data.get('module', 'Unknown'),
        'test_title': jira_data.get('title', 'Unknown'),
        'description': jira_data.get('description', ''),
        'acceptance_criteria': jira_data.get('acceptance_criteria', ''),
        'overall_status': state.get('overall_status', 'UNKNOWN'),
        'execution_start_time': state.get('execution_start_time', ''),
        'execution_end_time': state.get('execution_end_time', ''),
        'total_execution_time': state.get('total_execution_time', 0),
        'execution_results': execution_results,
        'video_path': state.get('video_path', ''),
        'script_path': state.get('script_path', ''),
        'steps_count': len(execution_results),
        'passed_count': len([r for r in execution_results if r['status'] == 'PASSED']),
        'failed_count': len([r for r in execution_results if r['status'] == 'FAILED']),
    }

    # Render template
    html_content = template.render(**context)

    # Save report
    reports_folder = Path(config['folders']['reports'])
    reports_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{jira_data.get('ticket_id', 'TEST')}_report_{timestamp}.html"
    report_path = reports_folder / report_filename

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Report saved: {report_path}")
    return str(report_path)


def generate_playwright_script(
    state: TestAutomationState,
    logger: logging.Logger
) -> str:
    """
    Generate Playwright Python script from execution results.

    Args:
        state: Workflow state with execution results
        logger: Logger instance

    Returns:
        Path to generated script file
    """
    config = state['config']
    jira_data = state.get('jira_data', {})
    execution_results = state.get('execution_results', [])

    # Build script content
    script_lines = []
    script_lines.append('"""')
    script_lines.append(f'Generated Playwright script for {jira_data.get("ticket_id", "TEST")}')
    script_lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    script_lines.append('"""')
    script_lines.append('')
    script_lines.append('from playwright.sync_api import sync_playwright')
    script_lines.append('import time')
    script_lines.append('')
    script_lines.append('def run():')
    script_lines.append('    with sync_playwright() as playwright:')
    script_lines.append('        # Launch browser')
    script_lines.append(f'        browser = playwright.chromium.launch(channel="msedge", headless=False)')
    script_lines.append('        context = browser.new_context(viewport={"width": 1920, "height": 1080})')
    script_lines.append('        page = context.new_page()')
    script_lines.append('')
    script_lines.append(f'        # Navigate to URL')
    script_lines.append(f'        page.goto("{config.get("web_url", "")}")')
    script_lines.append(f'        time.sleep({config["wait_times"]["page_load"] / 1000})')
    script_lines.append('')
    script_lines.append('        # TODO: Add login coordinates here')
    script_lines.append('')

    # Add steps
    for result in execution_results:
        step_num = result.get('step_num', 0)
        step_text = result.get('step_text', '')
        coordinates = result.get('coordinates', {})
        action_type = result.get('action_type', 'click')

        script_lines.append(f'        # Step {step_num}: {step_text}')

        if coordinates:
            x = coordinates.get('x', 0)
            y = coordinates.get('y', 0)
            script_lines.append(f'        page.mouse.click({x}, {y})')
            wait_time = config['wait_times']['after_click'] / 1000
            script_lines.append(f'        time.sleep({wait_time})')
        else:
            script_lines.append(f'        # WARNING: No coordinates available for this step')

        script_lines.append('')

    script_lines.append('        # Close browser')
    script_lines.append('        context.close()')
    script_lines.append('        browser.close()')
    script_lines.append('')
    script_lines.append('if __name__ == "__main__":')
    script_lines.append('    run()')

    script_content = '\n'.join(script_lines)

    # Save script
    scripts_folder = Path(config['folders']['scripts'])
    scripts_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"{jira_data.get('ticket_id', 'TEST')}_script_{timestamp}.py"
    script_path = scripts_folder / script_filename

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    logger.info(f"Script saved: {script_path}")
    return str(script_path)


def embed_screenshot(screenshot_path: str, logger: logging.Logger) -> str:
    """
    Embed screenshot as base64 data URI.

    Args:
        screenshot_path: Path to screenshot file
        logger: Logger instance

    Returns:
        Base64 encoded data URI string
    """
    try:
        if not Path(screenshot_path).exists():
            logger.warning(f"Screenshot not found: {screenshot_path}")
            return ""

        with open(screenshot_path, 'rb') as f:
            image_data = f.read()

        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    except Exception as e:
        logger.error(f"Error embedding screenshot: {e}")
        return ""


def get_builtin_template() -> str:
    """
    Get built-in HTML template as fallback.

    Returns:
        HTML template string
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {{ ticket_id }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }
        .status-passed { background-color: #28a745; }
        .status-failed { background-color: #dc3545; }
        .summary {
            background-color: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }
        .summary-item {
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .screenshot {
            max-width: 300px;
            cursor: pointer;
            border: 1px solid #ddd;
        }
        .screenshot:hover {
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Execution Report</h1>

        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-item"><strong>Ticket ID:</strong> {{ ticket_id }}</div>
            <div class="summary-item"><strong>Module:</strong> {{ module }}</div>
            <div class="summary-item"><strong>Title:</strong> {{ test_title }}</div>
            <div class="summary-item">
                <strong>Overall Status:</strong>
                <span class="status-badge status-{{ overall_status|lower }}">{{ overall_status }}</span>
            </div>
            <div class="summary-item"><strong>Execution Time:</strong> {{ "%.2f"|format(total_execution_time) }} seconds</div>
            <div class="summary-item"><strong>Steps:</strong> {{ passed_count }}/{{ steps_count }} passed</div>
        </div>

        <h2>Test Information</h2>
        <p><strong>Description:</strong> {{ description }}</p>
        <p><strong>Acceptance Criteria:</strong> {{ acceptance_criteria }}</p>

        <h2>Step Results</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Description</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Time (s)</th>
                <th>Screenshot Before</th>
                <th>Screenshot After</th>
            </tr>
            {% for result in execution_results %}
            <tr>
                <td>{{ result.step_num }}</td>
                <td>{{ result.step_text }}</td>
                <td><span class="status-badge status-{{ result.status|lower }}">{{ result.status }}</span></td>
                <td>{{ "%.2f"|format(result.confidence) }}</td>
                <td>{{ "%.2f"|format(result.execution_time) }}</td>
                <td>
                    {% if result.screenshot_before_base64 %}
                    <img src="{{ result.screenshot_before_base64 }}" class="screenshot" alt="Before">
                    {% else %}
                    N/A
                    {% endif %}
                </td>
                <td>
                    {% if result.screenshot_after_base64 %}
                    <img src="{{ result.screenshot_after_base64 }}" class="screenshot" alt="After">
                    {% else %}
                    N/A
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>

        <h2>Media Files</h2>
        <p><strong>Video:</strong> {{ video_path if video_path else 'Not available' }}</p>
        <p><strong>Script:</strong> {{ script_path if script_path else 'Not available' }}</p>

        <hr>
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Generated by AI-Powered Vision-Based Test Automation v1.0
        </p>
    </div>
</body>
</html>
"""
