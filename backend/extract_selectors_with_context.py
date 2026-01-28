"""
Generic Selector Extractor with Context (Scalable for Any Web Application)

Extracts data-* selectors from HTML files and enriches them with context
derived ONLY from the web application source code (no JIRA/test dependencies).

Works with: Angular, React, Vue, plain HTML
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup


class SelectorContextExtractor:
    """
    Extracts selectors with context from HTML/TypeScript source files.

    Context is derived from:
    1. HTML structure (parent elements, siblings)
    2. Element attributes (class, role, aria-*)
    3. TypeScript/JavaScript code (function names, variables)
    4. Angular/React directives
    5. Folder structure
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.all_selectors = []

    def extract_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Extract selectors from all HTML files in a folder.

        Args:
            folder_path: Relative path to component folder (e.g., 'src/app/create-new')

        Returns:
            List of enriched selector dictionaries
        """
        folder = self.base_path / folder_path
        module_name = folder.name

        print(f"\n=== Extracting from module: {module_name} ===")

        # Find all HTML files in this folder
        html_files = list(folder.glob('*.html'))
        ts_files = list(folder.glob('*.ts'))

        print(f"Found {len(html_files)} HTML files, {len(ts_files)} TypeScript files")

        selectors = []

        for html_file in html_files:
            # Get corresponding TypeScript file
            ts_file = html_file.with_suffix('.ts')

            # Extract from HTML
            html_selectors = self.extract_from_html(
                html_file,
                module_name,
                ts_file if ts_file.exists() else None
            )

            selectors.extend(html_selectors)

        print(f"Extracted {len(selectors)} selectors with context")

        return selectors

    def extract_from_html(
        self,
        html_file: Path,
        module_name: str,
        ts_file: Path = None
    ) -> List[Dict[str, Any]]:
        """
        Extract selectors from a single HTML file.

        Args:
            html_file: Path to HTML file
            module_name: Module name (from folder)
            ts_file: Optional TypeScript file path

        Returns:
            List of selector dictionaries with context
        """
        print(f"  Processing: {html_file.name}")

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse TypeScript if available
        ts_context = self._parse_typescript(ts_file) if ts_file else {}

        # Parse HTML
        selectors = []
        lines = html_content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Extract data-* attributes
            # Pattern 1: data-attr="value"
            static_matches = re.finditer(r'data-([a-zA-Z0-9\-_]+)="([^"]*)"', line)
            for match in static_matches:
                attr_name = match.group(1)
                attr_value = match.group(2)

                selector = self._build_selector_object(
                    attr=f'data-{attr_name}',
                    value=attr_value,
                    module=module_name,
                    html_line=line,
                    line_number=line_num,
                    is_dynamic=False,
                    html_file=html_file,
                    ts_context=ts_context
                )
                selectors.append(selector)

            # Pattern 2: [attr.data-attr]="variable"
            dynamic_matches = re.finditer(r'\[attr\.data-([a-zA-Z0-9\-_]+)\]="([^"]*)"', line)
            for match in dynamic_matches:
                attr_name = match.group(1)
                attr_value = match.group(2)

                selector = self._build_selector_object(
                    attr=f'attr.data-{attr_name}',
                    value=attr_value,
                    module=module_name,
                    html_line=line,
                    line_number=line_num,
                    is_dynamic=True,
                    html_file=html_file,
                    ts_context=ts_context
                )
                selectors.append(selector)

        return selectors

    def _build_selector_object(
        self,
        attr: str,
        value: str,
        module: str,
        html_line: str,
        line_number: int,
        is_dynamic: bool,
        html_file: Path,
        ts_context: Dict
    ) -> Dict[str, Any]:
        """
        Build enriched selector object with context.

        Context is derived from:
        - HTML structure analysis
        - TypeScript function names
        - Element attributes
        - Angular directives
        """
        # Analyze HTML line for context
        context = self._extract_context_from_html_line(html_line, attr, value)

        # Get element type
        element_type = self._extract_element_type(html_line)

        # Calculate priority based on context
        priority = self._calculate_priority(context, attr, element_type, ts_context)

        # Extract label from HTML
        label = self._extract_label(html_line)

        # Build usage scenario from context
        usage_scenario = self._build_usage_scenario(context, element_type, value, ts_context)

        # Get parent component from folder structure
        parent_component = self._get_parent_component(html_file)

        # Extract condition if any
        condition = self._extract_condition(html_line)

        return {
            'attr': attr,
            'value': value,
            'module': module,
            'context': context,
            'priority': priority,
            'usage_scenario': usage_scenario,
            'parentComponent': parent_component,
            'filePath': str(html_file.relative_to(self.base_path)).replace('\\', '/'),
            'dynamic': is_dynamic,
            'label': label,
            'lineNumber': line_number,
            'elementType': element_type,
            'condition': condition if condition else None
        }

    def _extract_context_from_html_line(self, html_line: str, attr: str, value: str) -> List[str]:
        """
        Extract context from HTML line analysis.

        Context keywords derived from:
        1. Element type (button, input, div, mat-icon, etc.)
        2. Angular directives (@if, @for, [matMenuTriggerFor], etc.)
        3. Click handlers (openDialog, route, create, etc.)
        4. CSS classes (mat-*, btn-*, form-*, etc.)
        5. ARIA attributes (role, aria-label, etc.)
        6. Attribute names (create, show, more, vertical, etc.)
        """
        context = []
        line_lower = html_line.lower()

        # 1. Detect element type
        if '<button' in line_lower:
            context.append('button')
        elif '<input' in line_lower:
            context.append('input')
        elif '<mat-icon' in line_lower or 'fonticon' in line_lower:
            context.append('icon')
        elif '<mat-menu' in line_lower:
            context.append('menu')
        elif '<ng-container' in line_lower:
            context.append('container')
        elif '<div' in line_lower:
            context.append('div')

        # 2. Detect Angular/Material components
        if 'mat-menu-item' in line_lower:
            context.extend(['menu-item', 'dropdown'])
        if 'matmenutriggerfor' in line_lower:
            context.extend(['menu-trigger', 'dropdown'])
        if 'mat-raised-button' in line_lower or 'color="primary"' in line_lower:
            context.append('primary-action')
        if 'mat-select' in line_lower or 'autocomplete' in line_lower:
            context.append('dropdown')
        if 'mat-expansion-panel' in line_lower or 'accordion' in line_lower:
            context.append('accordion')
        if 'mat-dialog' in line_lower or 'dialog' in line_lower:
            context.append('dialog')

        # 3. Detect actions from click handlers
        if '(click)="open' in line_lower:
            context.append('clickable')
            if 'opendialog' in line_lower or 'opencreatedialog' in line_lower:
                context.extend(['create', 'dialog'])
            if 'opendetailview' in line_lower or 'routetodetailview' in line_lower:
                context.extend(['detail-view'])

        # 4. Detect from attribute name and value
        attr_lower = attr.lower()
        value_lower = value.lower()

        if 'create' in attr_lower or 'create' in value_lower:
            context.append('create')
        if 'show' in attr_lower or 'show' in value_lower:
            context.append('show')
        if 'more' in attr_lower or 'more' in value_lower:
            context.extend(['more-options'])
        if 'vertical' in attr_lower or 'vertical' in value_lower:
            context.append('more-vertical')
        if 'dropdown' in attr_lower or 'dropdown' in value_lower:
            context.append('dropdown')
        if 'edit' in attr_lower or 'edit' in value_lower:
            context.append('edit')
        if 'delete' in attr_lower or 'delete' in value_lower:
            context.append('delete')
        if 'save' in attr_lower or 'save' in value_lower:
            context.append('save')
        if 'close' in attr_lower or 'close' in value_lower or 'cancel' in attr_lower:
            context.append('close')
        if 'add' in attr_lower or 'addicon' in attr_lower:
            context.append('add')
        if 'accordion' in attr_lower or 'panel' in attr_lower:
            context.append('accordion')
        if 'table' in attr_lower or 'row' in attr_lower:
            context.append('table')

        # 5. Detect from ARIA/role attributes
        if 'role="button"' in html_line or '[role="button"]' in html_line:
            context.append('button')
        if 'role="menu"' in html_line or 'role="menuitem"' in html_line:
            context.append('menu')

        # 6. Detect conditional context
        if '@if' in html_line:
            if '_detailviewcondition' in line_lower:
                context.append('detail-view')
            if 'iscreatebtnenable' in line_lower:
                context.append('create')

        # 7. Detect from translation keys
        if 'translate' in html_line:
            match = re.search(r"'([^']*)'\\s*\|\\s*translate", html_line)
            if match:
                translation_key = match.group(1).lower()
                if 'create' in translation_key:
                    context.append('create')
                if 'save' in translation_key:
                    context.append('save')
                if 'cancel' in translation_key:
                    context.append('cancel')

        # Remove duplicates and return
        return list(dict.fromkeys(context))  # Preserves order

    def _extract_element_type(self, html_line: str) -> str:
        """Extract HTML element type from line."""
        match = re.search(r'<([a-z\-]+)', html_line.lower())
        if match:
            return match.group(1)
        return 'unknown'

    def _calculate_priority(
        self,
        context: List[str],
        attr: str,
        element_type: str,
        ts_context: Dict
    ) -> int:
        """
        Calculate selector priority (1-10).

        Priority rules:
        10 - Critical actions (primary create/save, main triggers)
        9  - Important actions (detail view, menu items)
        8  - Supporting elements (containers, secondary buttons)
        7  - Decorative/icon elements
        5  - Default
        3  - Unknown/low confidence
        """
        priority = 5  # Default

        # High priority for test-specific attributes
        if 'data-testid' in attr or 'data-cy' in attr or 'data-test' in attr:
            return 10

        # High priority for primary actions
        if 'primary-action' in context or 'create' in context:
            priority = 9
            if 'button' in context and element_type == 'button':
                priority = 10  # Primary action button

        # High priority for menu triggers (dropdown buttons)
        if 'menu-trigger' in context or ('dropdown' in context and 'button' in context):
            priority = 10

        # Medium-high for dialog/detail-view actions
        if 'dialog' in context or 'detail-view' in context:
            priority = 9

        # Medium for menu items
        if 'menu-item' in context:
            priority = 9

        # Medium for containers
        if 'container' in context:
            priority = 8

        # Lower for icons (unless they're action icons)
        if 'icon' in context:
            priority = 7
            if 'add' in context or 'edit' in context or 'delete' in context:
                priority = 8  # Action icons

        # Boost if function name appears in TypeScript
        attr_clean = attr.lower().replace('data-', '').replace('attr.', '')
        if ts_context.get('functions'):
            for func_name in ts_context['functions']:
                if attr_clean in func_name.lower():
                    priority = min(priority + 1, 10)
                    break

        return priority

    def _extract_label(self, html_line: str) -> str:
        """Extract visible label/text from HTML line."""
        # Pattern 1: {{variable | translate}}
        match = re.search(r'{{([^}]+)}}', html_line)
        if match:
            return match.group(1).strip()

        # Pattern 2: Text content between tags
        match = re.search(r'>([^<]+)</', html_line)
        if match:
            text = match.group(1).strip()
            if text and not text.startswith('{') and len(text) < 100:
                return text

        return ''

    def _build_usage_scenario(
        self,
        context: List[str],
        element_type: str,
        value: str,
        ts_context: Dict
    ) -> str:
        """
        Build human-readable usage scenario from context.

        Example outputs:
        - "Primary create button that opens dialog"
        - "Dropdown menu trigger button"
        - "Add icon on create button"
        """
        parts = []

        # Start with element type
        if 'button' in context:
            if 'primary-action' in context:
                parts.append('Primary')
            if 'menu-trigger' in context:
                parts.append('Dropdown menu trigger')
            elif 'menu-item' in context:
                parts.append('Menu item')
            else:
                parts.append('Button')
        elif 'icon' in context:
            parts.append('Icon')
        elif 'input' in context:
            parts.append('Input field')
        elif 'container' in context:
            parts.append('Container')

        # Add action context
        actions = []
        if 'create' in context:
            actions.append('create')
        if 'edit' in context:
            actions.append('edit')
        if 'delete' in context:
            actions.append('delete')
        if 'save' in context:
            actions.append('save')
        if 'close' in context:
            actions.append('close')

        if actions:
            parts.append(' / '.join(actions))

        # Add target context
        if 'dialog' in context:
            parts.append('opens dialog')
        if 'detail-view' in context:
            parts.append('in detail view')
        if 'dropdown' in context and 'menu-trigger' not in context:
            parts.append('in dropdown menu')

        # Combine parts
        if not parts:
            parts = [f'Element with value: {value}']

        return ' '.join(parts)

    def _get_parent_component(self, html_file: Path) -> str:
        """Get parent component name from file path."""
        # Example: create-new/create-ae-name/create-ae-name.component.html
        #   -> parent: create-ae-name/create-ae-name

        parts = html_file.relative_to(self.base_path).parts
        if len(parts) >= 2:
            # If in subfolder: parent/child/file.html
            if len(parts) >= 3 and parts[-2] == parts[-3]:
                return f'{parts[-3]}/{parts[-2]}'
            else:
                return parts[-2]
        return html_file.parent.name

    def _extract_condition(self, html_line: str) -> str:
        """Extract @if condition from Angular template."""
        match = re.search(r'@if\s*\(([^)]+)\)', html_line)
        if match:
            return match.group(1).strip()
        return None

    def _parse_typescript(self, ts_file: Path) -> Dict[str, Any]:
        """
        Parse TypeScript file for additional context.

        Extracts:
        - Function names
        - Variable names
        - Enums/constants
        """
        if not ts_file.exists():
            return {}

        with open(ts_file, 'r', encoding='utf-8') as f:
            ts_content = f.read()

        context = {
            'functions': [],
            'variables': [],
            'enums': []
        }

        # Extract function names
        func_matches = re.finditer(r'\\b(\\w+)\\s*\\([^)]*\\)\\s*{', ts_content)
        for match in func_matches:
            context['functions'].append(match.group(1))

        # Extract variable declarations
        var_matches = re.finditer(r'\\b(\\w+):\\s*string\\s*=', ts_content)
        for match in var_matches:
            context['variables'].append(match.group(1))

        return context


def main():
    """
    Main execution function.

    Usage:
        python extract_selectors_with_context.py
    """
    # Base path to web application source
    base_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client"

    # Initialize extractor
    extractor = SelectorContextExtractor(base_path)

    # Extract from create-new module (test case)
    selectors = extractor.extract_from_folder('src/app/create-new')

    # Save enriched selectors
    output_file = Path('Selectors_Folder/create-new_enriched_scalable.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selectors, f, indent=2)

    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total selectors extracted: {len(selectors)}")
    print(f"Saved to: {output_file}")

    # Show sample
    print("\n=== SAMPLE ENRICHED SELECTORS ===")
    for selector in selectors[:3]:
        print(f"\nSelector: {selector['attr']}")
        print(f"  Context: {selector['context']}")
        print(f"  Priority: {selector['priority']}")
        print(f"  Usage: {selector['usage_scenario']}")


if __name__ == "__main__":
    main()
