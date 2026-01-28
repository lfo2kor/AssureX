"""
PLCD Testing Assistant - HTML Report Generator
Generates execution reports
"""

import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_html_report(
    ticket_id: str,
    ticket_data: Dict,
    step_results: List[Dict],
    overall_status: str,
    execution_time: float,
    config: Dict,
    timestamp: str = None
) -> str:
    """
    Generate HTML report for test execution

    Args:
        ticket_id: Jira ticket ID
        ticket_data: Parsed ticket data
        step_results: List of step execution results
        overall_status: PASSED/FAILED
        execution_time: Total execution time
        config: Configuration dictionary
        timestamp: Optional pre-generated timestamp (for consistent naming across artifacts)

    Returns:
        Path to generated HTML report
    """
    # Create reports folder
    reports_folder = Path(config['folders']['reports'])
    reports_folder.mkdir(parents=True, exist_ok=True)

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{ticket_id}_{timestamp}_report.html"
    report_path = reports_folder / report_filename

    # Count statistics
    total_steps = len(step_results)
    passed_steps = sum(1 for r in step_results if r.get('status') == 'PASSED')
    failed_steps = sum(1 for r in step_results if r.get('status') == 'FAILED')

    # Agent usage statistics
    l1_count = sum(1 for r in step_results if r.get('agent_used') == 'L1')
    l2_count = sum(1 for r in step_results if r.get('agent_used') == 'L2')
    l3_count = sum(1 for r in step_results if r.get('agent_used') == 'L3')

    # Average confidence
    confidences = [r.get('confidence', 0) for r in step_results if 'confidence' in r]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {ticket_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2 px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .status-PASSED {{ color: #28a745; }}
        .status-FAILED {{ color: #dc3545; }}
        .step-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .step-table th {{
            background-color: #007acc;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .step-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        .step-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-PASSED {{
            background-color: #28a745;
            color: white;
        }}
        .badge-FAILED {{
            background-color: #dc3545;
            color: white;
        }}
        .badge-L1 {{
            background-color: #17a2b8;
            color: white;
        }}
        .badge-L2 {{
            background-color: #ffc107;
            color: #333;
        }}
        .badge-L3 {{
            background-color: #6f42c1;
            color: white;
        }}
        .selector {{
            font-family: 'Courier New', monospace;
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Execution Report</h1>
        <p><strong>Ticket:</strong> {ticket_id} - {ticket_data.get('title', 'N/A')}</p>
        <p><strong>Module:</strong> {ticket_data.get('module', 'N/A')}</p>
        <p><strong>Execution Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-card">
                <h3>Overall Status</h3>
                <div class="value status-{overall_status}">{overall_status}</div>
            </div>
            <div class="summary-card">
                <h3>Total Steps</h3>
                <div class="value">{total_steps}</div>
            </div>
            <div class="summary-card">
                <h3>Passed</h3>
                <div class="value status-PASSED">{passed_steps}</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="value status-FAILED">{failed_steps}</div>
            </div>
            <div class="summary-card">
                <h3>Execution Time</h3>
                <div class="value">{execution_time:.1f}s</div>
            </div>
            <div class="summary-card">
                <h3>Avg Confidence</h3>
                <div class="value">{avg_confidence:.2f}</div>
            </div>
        </div>

        <h2>Agent Usage</h2>
        <p>L1 (Semantic Search): {l1_count} | L2 (DOM Discovery): {l2_count} | L3 (Vision): {l3_count}</p>

        <h2>Step Details</h2>
        <table class="step-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Step</th>
                    <th>Selector</th>
                    <th>Agent</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add step rows
    for step_result in step_results:
        step_num = step_result.get('step_number', '?')
        step_text = step_result.get('step_text', 'N/A')
        selector = step_result.get('selector', 'N/A')
        agent = step_result.get('agent_used', 'N/A')
        confidence = step_result.get('confidence', 0)
        status = step_result.get('status', 'UNKNOWN')

        # Make skipped login steps more readable
        if status == 'skipped' and step_text in ['N/A', 'Login', 'login']:
            step_num = '1'
            step_text = 'Login (already authenticated)'
            selector = 'N/A - Session reused'
            agent = 'Auth'

        html_content += f"""
                <tr>
                    <td>{step_num}</td>
                    <td>{step_text}</td>
                    <td><span class="selector">{selector}</span></td>
                    <td><span class="badge badge-{agent}">{agent}</span></td>
                    <td>{confidence:.2f}</td>
                    <td><span class="badge badge-{status}">{status}</span></td>
                </tr>
"""

    html_content += f"""
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by PLCD Testing Assistant | Powered by Azure OpenAI & ChromaDB</p>
        </div>
    </div>
</body>
</html>
"""

    # Write to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(report_path)
