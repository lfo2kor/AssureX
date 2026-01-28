"""
Failure Analyzer - Analyzes why all 3 selector levels failed.

Provides detailed breakdown and actionable recommendations.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class FailureAnalyzer:
    """Analyzes test step failures and provides actionable guidance."""

    def __init__(self, selector_loader, page, module: str, logger: logging.Logger):
        """
        Initialize failure analyzer.

        Args:
            selector_loader: SelectorLoader instance
            page: Playwright page object
            module: Current test module
            logger: Logger instance
        """
        self.selector_loader = selector_loader
        self.page = page
        self.module = module
        self.logger = logger

    def analyze_failure(self, step_text: str, step_num: int) -> Dict[str, Any]:
        """
        Analyze why all 3 levels failed.

        Args:
            step_text: Test step description
            step_num: Step number

        Returns:
            Dictionary with detailed failure analysis
        """
        analysis = {
            'step_number': step_num,
            'step_text': step_text,
            'module': self.module,
            'failure_reason': 'Unknown',
            'l1_details': {},
            'l2_details': {},
            'l3_details': {},
            'recommendations': [],
            'missing_selector_template': None
        }

        # Analyze each level
        analysis['l1_details'] = self._analyze_l1_failure(step_text)
        analysis['l2_details'] = self._analyze_l2_failure(step_text)
        analysis['l3_details'] = self._analyze_l3_failure(step_text)

        # Determine primary failure reason
        if analysis['l1_details']['reason'] == 'no_selectors_found':
            analysis['failure_reason'] = 'MISSING_SELECTOR_IN_JSON'
            analysis['recommendations'].append(
                "Add the missing selector to JSON file manually"
            )
            analysis['recommendations'].append(
                "Or run: python extract_runtime_selectors.py <TICKET_ID>"
            )
            analysis['missing_selector_template'] = self._generate_selector_template(step_text)

        elif analysis['l1_details']['reason'] == 'selector_ambiguous':
            analysis['failure_reason'] = 'AMBIGUOUS_SELECTOR'
            analysis['recommendations'].append(
                f"Selector [{analysis['l1_details']['selector']}] matches {analysis['l1_details']['count_on_page']} elements"
            )
            analysis['recommendations'].append(
                "Add tag/class to make it unique (e.g., input.mat-input[data-attr='value'])"
            )
            analysis['recommendations'].append(
                "Or use a more specific data attribute value"
            )

        elif analysis['l1_details']['reason'] == 'selector_not_on_page':
            analysis['failure_reason'] = 'SELECTOR_NOT_VISIBLE'
            analysis['recommendations'].append(
                f"Selector exists in JSON but count on page: {analysis['l1_details']['count_on_page']}"
            )
            analysis['recommendations'].append(
                f"Current module: {self.module} - Check if this is correct"
            )
            analysis['recommendations'].append(
                "Element might be in a different module or not loaded yet"
            )

        elif analysis['l2_details']['reason'] == 'no_pattern_matched':
            analysis['failure_reason'] = 'NO_GENERIC_PATTERN_AVAILABLE'
            analysis['recommendations'].append(
                "This element type doesn't have a generic L2 pattern"
            )
            analysis['recommendations'].append(
                "You MUST add a custom selector to JSON for this element"
            )

        else:
            analysis['failure_reason'] = 'ELEMENT_NOT_DETECTABLE'
            analysis['recommendations'].append(
                "Element might be hidden, in iframe, or requires scroll/wait"
            )
            analysis['recommendations'].append(
                "Check the failure screenshot for visual clues"
            )
            analysis['recommendations'].append(
                "Element might need JavaScript interaction or different approach"
            )

        return analysis

    def _analyze_l1_failure(self, step_text: str) -> Dict[str, Any]:
        """
        Analyze why L1 failed.

        Returns:
            {
                'tried': bool,
                'reason': str,  # 'no_selectors_found' | 'selector_ambiguous' | 'selector_not_on_page'
                'keywords_extracted': list,
                'selectors_found': int,
                'selector': str | None,
                'count_on_page': int
            }
        """
        details = {
            'tried': True,
            'reason': 'unknown',
            'keywords_extracted': [],
            'selectors_found': 0,
            'selector': None,
            'count_on_page': 0
        }

        # Extract keywords that L1 would use
        try:
            keywords = self.selector_loader._extract_keywords(step_text)
            details['keywords_extracted'] = keywords
        except:
            details['keywords_extracted'] = []

        # Find what L1 would search for
        try:
            selector_obj = self.selector_loader.find_best_selector(step_text, self.module)

            if not selector_obj:
                details['reason'] = 'no_selectors_found'
                return details

            details['selectors_found'] = 1

            # Build selector and check count
            selector_str = self.selector_loader.build_selector(selector_obj)
            details['selector'] = selector_str

            # Check count on page
            count = self.page.locator(selector_str).count()
            details['count_on_page'] = count

            if count == 0:
                details['reason'] = 'selector_not_on_page'
            elif count > 1:
                details['reason'] = 'selector_ambiguous'
            else:
                details['reason'] = 'found_but_action_failed'

        except Exception as e:
            details['reason'] = 'error_during_analysis'
            details['error'] = str(e)

        return details

    def _analyze_l2_failure(self, step_text: str) -> Dict[str, Any]:
        """
        Analyze why L2 failed.

        Returns:
            {
                'tried': bool,
                'reason': str,
                'patterns_tried': list,
                'action_type': str
            }
        """
        details = {
            'tried': True,
            'reason': 'no_pattern_matched',
            'patterns_tried': [],
            'action_type': 'unknown'
        }

        step_lower = step_text.lower()

        # Determine action type and patterns that would be tried
        if 'navigate' in step_lower or ('click' in step_lower and any(w in step_lower for w in ['button', 'link', 'menu'])):
            details['action_type'] = 'navigation/button'
            details['patterns_tried'] = [
                "button:has-text('...')",
                "a:has-text('...')",
                "[role='link']:has-text('...')",
                "[role='button']:has-text('...')"
            ]

        elif 'select' in step_lower and 'dropdown' in step_lower:
            details['action_type'] = 'dropdown'
            details['patterns_tried'] = [
                "[data-attribute='Field']",
                "input.mat-mdc-autocomplete-trigger",
                ".mat-select",
                "[role='combobox']"
            ]

        elif any(word in step_lower for word in ['enter', 'type', 'input']):
            details['action_type'] = 'text_input'
            details['patterns_tried'] = [
                "input[name='...']",
                "input[placeholder='...']",
                "input[type='text']"
            ]

        elif 'accordion' in step_lower or 'panel' in step_lower:
            details['action_type'] = 'accordion'
            details['patterns_tried'] = [
                "[role='button'][aria-expanded='false']",
                ".mat-expansion-panel-header:has-text('...')"
            ]

        elif 'message' in step_lower or 'displayed' in step_lower:
            details['action_type'] = 'verification'
            details['patterns_tried'] = [
                ":has-text('...')",
                "[role='alert']",
                ".notification"
            ]

        else:
            details['action_type'] = 'generic_action'
            details['patterns_tried'] = [
                "Generic button/link patterns"
            ]

        return details

    def _analyze_l3_failure(self, step_text: str) -> Dict[str, Any]:
        """
        Analyze why L3 (CV) failed.

        Returns:
            {
                'tried': bool,
                'reason': str,
                'possible_causes': list
            }
        """
        details = {
            'tried': True,
            'reason': 'cv_failed',
            'possible_causes': []
        }

        # L3 failure usually means:
        details['possible_causes'] = [
            "Element not visible in screenshot",
            "Element in iframe or shadow DOM",
            "Element requires scroll to be visible",
            "Element hidden by overlay or modal",
            "CV couldn't identify unique selector",
            "Network/timing issue - element not loaded"
        ]

        return details

    def _generate_selector_template(self, step_text: str) -> Dict[str, Any]:
        """
        Generate JSON template for missing selector.

        Args:
            step_text: Step description

        Returns:
            JSON template dictionary
        """
        step_lower = step_text.lower()

        # Determine element type from step text
        if 'button' in step_lower or 'click' in step_lower:
            tag = 'button'
            context = ['btn', 'button', 'click']
        elif 'input' in step_lower or 'enter' in step_lower or 'type' in step_lower:
            tag = 'input'
            context = ['input', 'field', 'text']
        elif 'dropdown' in step_lower or 'select' in step_lower:
            tag = 'mat-select'
            context = ['dropdown', 'select', 'option']
        elif 'accordion' in step_lower or 'panel' in step_lower:
            tag = 'div'
            context = ['accordion', 'panel', 'expand']
        else:
            tag = 'UNKNOWN_TAG'
            context = []

        template = {
            "attr": "data-FIXME-ATTRIBUTE-NAME",
            "value": "FIXME-ATTRIBUTE-VALUE",
            "tagName": tag,
            "className": "FIXME-CLASS-NAME",
            "module": self.module,
            "context": context,
            "priority": 20,
            "isClickable": True,
            "isVisible": True,
            "source": "manual_fix",
            "label": f"FIXME: {step_text[:60]}"
        }

        return template

    def save_failure_report(self, step_num: int, analysis: Dict[str, Any], screenshot_path: str = None) -> str:
        """
        Save detailed failure report to file.

        Args:
            step_num: Step number
            analysis: Analysis dictionary
            screenshot_path: Path to failure screenshot

        Returns:
            Path to text report file
        """
        # Create Failure_Reports directory
        reports_dir = Path('Failure_Reports')
        reports_dir.mkdir(exist_ok=True)

        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = reports_dir / f'failure_step_{step_num}_{timestamp}.json'
        text_path = reports_dir / f'failure_step_{step_num}_{timestamp}.txt'

        # Save JSON report
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Save human-readable text report
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"FAILURE ANALYSIS - Step {step_num}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Step Text: {analysis['step_text']}\n")
            f.write(f"Module: {analysis['module']}\n")
            f.write(f"Failure Reason: {analysis['failure_reason']}\n\n")

            f.write("="*80 + "\n")
            f.write("L1 ANALYSIS (Custom Selectors from JSON):\n")
            f.write("="*80 + "\n")
            l1 = analysis['l1_details']
            f.write(f"Keywords extracted: {l1.get('keywords_extracted', [])}\n")
            f.write(f"Selectors found in JSON: {l1.get('selectors_found', 0)}\n")
            f.write(f"Reason: {l1.get('reason', 'unknown')}\n")
            if l1.get('selector'):
                f.write(f"Selector tried: {l1['selector']}\n")
                f.write(f"Count on page: {l1.get('count_on_page', 0)}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("L2 ANALYSIS (Generic HTML Patterns):\n")
            f.write("="*80 + "\n")
            l2 = analysis['l2_details']
            f.write(f"Action type: {l2.get('action_type', 'unknown')}\n")
            f.write(f"Patterns tried:\n")
            for pattern in l2.get('patterns_tried', []):
                f.write(f"  - {pattern}\n")
            f.write(f"Reason: {l2.get('reason', 'unknown')}\n\n")

            f.write("="*80 + "\n")
            f.write("L3 ANALYSIS (CV-Guided):\n")
            f.write("="*80 + "\n")
            l3 = analysis['l3_details']
            f.write(f"Reason: {l3.get('reason', 'unknown')}\n")
            f.write(f"Possible causes:\n")
            for cause in l3.get('possible_causes', []):
                f.write(f"  - {cause}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("="*80 + "\n")
            for i, rec in enumerate(analysis.get('recommendations', []), 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            if analysis.get('missing_selector_template'):
                f.write("="*80 + "\n")
                f.write("MISSING SELECTOR TEMPLATE:\n")
                f.write("="*80 + "\n")
                f.write("Add this to your selectors JSON file:\n\n")
                f.write(json.dumps(analysis['missing_selector_template'], indent=2))
                f.write("\n\n")

            if screenshot_path:
                f.write("="*80 + "\n")
                f.write("SCREENSHOT:\n")
                f.write("="*80 + "\n")
                f.write(f"{screenshot_path}\n\n")

            f.write("="*80 + "\n")
            f.write("NEXT STEPS:\n")
            f.write("="*80 + "\n")
            f.write("1. Review the failure analysis above\n")
            f.write("2. Check the screenshot for visual confirmation\n")
            f.write("3. Add/fix the selector based on recommendations\n")
            f.write("4. Re-run the test: python run_test.py <TICKET_ID>\n")

        self.logger.error(f"Detailed failure report saved to: {text_path}")
        return str(text_path)
