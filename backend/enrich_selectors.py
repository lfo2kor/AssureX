"""
Selector Enrichment Script

Enriches selectors.json by extracting context, priority, and metadata
from Angular component HTML files.

Usage:
    python enrich_selectors.py

Input:
    - Selectors_Folder/selectors.json (current selectors)
    - C:/Projects/AI_Chat/PLCD/cri-webapp/client/src/app/ (HTML files)

Output:
    - Selectors_Folder/selectors_enriched.json (enriched selectors)
"""

import json
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SelectorEnricher:
    """Enriches selectors with context, priority, and metadata from HTML"""

    def __init__(self, webapp_path: str):
        self.webapp_path = Path(webapp_path)
        self.html_cache = {}  # Cache HTML files for performance

    def enrich_selectors(self, selectors: list) -> list:
        """
        Enrich list of selectors with context and metadata.

        Args:
            selectors: List of selector dictionaries

        Returns:
            List of enriched selector dictionaries
        """
        enriched = []
        total = len(selectors)

        for idx, selector in enumerate(selectors, 1):
            logger.info(f"Processing {idx}/{total}: {selector.get('attr', 'unknown')}")

            try:
                enriched_selector = self._enrich_single_selector(selector)
                enriched.append(enriched_selector)
            except Exception as e:
                logger.warning(f"Could not enrich {selector.get('attr')}: {e}")
                # Keep original if enrichment fails
                enriched.append(selector)

        return enriched

    def _enrich_single_selector(self, selector: dict) -> dict:
        """Enrich a single selector with context and metadata"""

        # Copy original
        enriched = selector.copy()

        # Get HTML content
        file_path = selector.get('filePath', '')
        if not file_path:
            logger.warning(f"No filePath for {selector.get('attr')}")
            return self._add_minimal_enrichment(enriched)

        html_content = self._load_html_file(file_path)
        if not html_content:
            logger.warning(f"Could not load HTML: {file_path}")
            return self._add_minimal_enrichment(enriched)

        # Find the selector's HTML line
        attr = selector.get('attr', '').replace('attr.', '').replace('data-', '')
        value = selector.get('value', '')

        html_line = self._find_selector_in_html(html_content, attr, value)
        if not html_line:
            logger.debug(f"Could not find selector in HTML: {attr}")
            return self._add_minimal_enrichment(enriched)

        # Extract context from HTML
        context = self._extract_context_from_html(html_line, attr, value)
        enriched['context'] = context

        # Calculate priority
        element_type = self._extract_element_type(html_line)
        enriched['elementType'] = element_type
        enriched['priority'] = self._calculate_priority(context, element_type)

        # Generate usage scenario
        enriched['usage_scenario'] = self._build_usage_scenario(context, element_type, value)

        # Extract line number
        line_num = self._find_line_number(html_content, attr, value)
        if line_num:
            enriched['lineNumber'] = line_num

        logger.info(f"  ✅ Enriched with {len(context)} context keywords, priority={enriched['priority']}")

        return enriched

    def _load_html_file(self, file_path: str) -> str:
        """Load HTML file content (with caching)"""

        if file_path in self.html_cache:
            return self.html_cache[file_path]

        # Try to find file
        full_path = self.webapp_path / file_path.replace('\\', '/')

        if not full_path.exists():
            # Try without src/ prefix
            full_path = self.webapp_path / file_path.replace('src/', '').replace('\\', '/')

        if not full_path.exists():
            logger.debug(f"File not found: {full_path}")
            return ""

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.html_cache[file_path] = content
                return content
        except Exception as e:
            logger.warning(f"Error reading {full_path}: {e}")
            return ""

    def _find_selector_in_html(self, html_content: str, attr: str, value: str) -> str:
        """Find the HTML line containing the selector"""

        # Build search patterns
        patterns = [
            f'{attr}="{value}"',
            f'{attr}=\'{value}\'',
            f'data-{attr}="{value}"',
            f'data-{attr}=\'{value}\'',
            f'[attr.data-{attr}]="{value}"',
            f'attr.data-{attr}',
        ]

        for pattern in patterns:
            if pattern in html_content:
                # Find the line containing this pattern
                lines = html_content.split('\n')
                for line in lines:
                    if pattern in line:
                        return line

        return ""

    def _extract_context_from_html(self, html_line: str, attr: str, value: str) -> list:
        """Extract context keywords from HTML line"""

        context = []
        line_lower = html_line.lower()

        # 1. Element type detection
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

        # 2. Angular Material directives
        if 'matmenutriggerfor' in line_lower:
            context.extend(['menu-trigger', 'dropdown'])

        if 'mat-menu-item' in line_lower:
            context.extend(['menu-item', 'dropdown'])

        if 'mat-raised-button' in line_lower or 'color="primary"' in line_lower:
            context.append('primary-action')

        if 'mat-select' in line_lower or 'autocomplete' in line_lower:
            context.append('dropdown')

        if 'mat-expansion-panel' in line_lower or 'accordion' in line_lower:
            context.append('accordion')

        if 'mat-dialog' in line_lower or 'dialog' in line_lower:
            context.append('dialog')

        # 3. Click handlers
        if '(click)="open' in line_lower:
            context.append('clickable')

            if 'opendialog' in line_lower or 'opencreatedialog' in line_lower:
                context.extend(['create', 'dialog'])

            if 'opendetailview' in line_lower or 'routetodetailview' in line_lower:
                context.append('detail-view')

        # 4. Attribute name analysis
        attr_lower = attr.lower()
        value_lower = value.lower()

        keyword_mapping = {
            'create': 'create',
            'show': 'show',
            'more': 'more-options',
            'vertical': 'more-vertical',
            'dropdown': 'dropdown',
            'edit': 'edit',
            'delete': 'delete',
            'save': 'save',
            'close': 'close',
            'cancel': 'cancel',
            'add': 'add',
            'accordion': 'accordion',
            'panel': 'accordion',
            'table': 'table',
            'row': 'table',
        }

        for keyword, context_value in keyword_mapping.items():
            if keyword in attr_lower or keyword in value_lower:
                if context_value not in context:
                    context.append(context_value)

        # 5. Icon analysis
        if 'fonticon' in line_lower:
            icon_match = re.search(r'fonticon="([^"]+)"', line_lower)
            if icon_match:
                icon_name = icon_match.group(1)
                if 'add' in icon_name:
                    context.append('add')
                if 'edit' in icon_name:
                    context.append('edit')
                if 'delete' in icon_name or 'remove' in icon_name:
                    context.append('delete')
                if 'more' in icon_name or 'vertical' in icon_name:
                    context.append('more-options')

        # Remove duplicates while preserving order
        return list(dict.fromkeys(context))

    def _extract_element_type(self, html_line: str) -> str:
        """Extract HTML element type from line"""

        match = re.search(r'<([a-z\-]+)', html_line.lower())
        if match:
            return match.group(1)
        return 'unknown'

    def _calculate_priority(self, context: list, element_type: str) -> int:
        """Calculate priority score (1-10) based on context and element type"""

        priority = 5  # Default

        # High priority for critical actions
        if 'menu-trigger' in context or 'primary-action' in context:
            priority = 10

        elif 'create' in context and 'button' in context:
            priority = 9

        elif 'dialog' in context or 'detail-view' in context:
            priority = 9

        elif 'menu-item' in context:
            priority = 9

        elif 'accordion' in context:
            priority = 8

        elif 'container' in context:
            priority = 7

        elif 'icon' in context:
            priority = 7
            # Boost for action icons
            if 'add' in context or 'edit' in context or 'delete' in context:
                priority = 8

        return priority

    def _build_usage_scenario(self, context: list, element_type: str, value: str) -> str:
        """Build human-readable usage scenario description"""

        parts = []

        # Element type description
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

        # Action verbs
        actions = []
        if 'create' in context:
            actions.append('create')
        if 'edit' in context:
            actions.append('edit')
        if 'delete' in context:
            actions.append('delete')
        if 'save' in context:
            actions.append('save')

        if actions:
            parts.append(' / '.join(actions))

        # Target context
        if 'dialog' in context:
            parts.append('opens dialog')
        if 'detail-view' in context:
            parts.append('in detail view')
        if 'dropdown' in context and 'menu-trigger' not in context:
            parts.append('in dropdown menu')

        # Value hint
        if value and value.lower() != 'unknown':
            parts.append(f"({value})")

        scenario = ' '.join(parts)
        return scenario if scenario else "UI element"

    def _find_line_number(self, html_content: str, attr: str, value: str) -> int:
        """Find line number of selector in HTML"""

        patterns = [
            f'{attr}="{value}"',
            f'data-{attr}="{value}"',
            f'attr.data-{attr}',
        ]

        lines = html_content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if pattern in line:
                    return line_num

        return 0

    def _add_minimal_enrichment(self, selector: dict) -> dict:
        """Add minimal enrichment when HTML not available"""

        # Extract keywords from attr/value
        attr = selector.get('attr', '').lower()
        value = selector.get('value', '').lower()

        context = []
        if 'btn' in attr or 'button' in attr:
            context.append('button')
        if 'icon' in attr:
            context.append('icon')
        if 'dropdown' in attr or 'menu' in attr:
            context.append('dropdown')
        if 'accordion' in attr or 'panel' in attr:
            context.append('accordion')

        selector['context'] = context
        selector['priority'] = 5
        selector['usage_scenario'] = "UI element"
        selector['elementType'] = 'unknown'

        return selector


def main():
    """Main enrichment process"""

    # Paths
    selectors_file = Path("Selectors_Folder/selectors.json")
    output_file = Path("Selectors_Folder/selectors_enriched.json")
    webapp_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src/app"

    logger.info("="*60)
    logger.info("SELECTOR ENRICHMENT SCRIPT")
    logger.info("="*60)

    # Load current selectors
    logger.info(f"\nLoading selectors from: {selectors_file}")
    try:
        with open(selectors_file, 'r', encoding='utf-8') as f:
            selectors = json.load(f)
        logger.info(f"✅ Loaded {len(selectors)} selectors")
    except Exception as e:
        logger.error(f"❌ Error loading selectors: {e}")
        return

    # Initialize enricher
    logger.info(f"\nWeb app path: {webapp_path}")
    enricher = SelectorEnricher(webapp_path)

    # Enrich selectors
    logger.info(f"\n{'='*60}")
    logger.info("ENRICHING SELECTORS")
    logger.info("="*60)

    enriched_selectors = enricher.enrich_selectors(selectors)

    # Save enriched selectors
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving enriched selectors to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_selectors, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(enriched_selectors)} enriched selectors")
    except Exception as e:
        logger.error(f"❌ Error saving: {e}")
        return

    # Statistics
    logger.info(f"\n{'='*60}")
    logger.info("ENRICHMENT STATISTICS")
    logger.info("="*60)

    enriched_count = sum(1 for s in enriched_selectors if s.get('context'))
    has_priority = sum(1 for s in enriched_selectors if 'priority' in s)
    has_usage = sum(1 for s in enriched_selectors if 'usage_scenario' in s)

    logger.info(f"Total selectors: {len(enriched_selectors)}")
    logger.info(f"With context: {enriched_count} ({enriched_count/len(enriched_selectors)*100:.1f}%)")
    logger.info(f"With priority: {has_priority} ({has_priority/len(enriched_selectors)*100:.1f}%)")
    logger.info(f"With usage scenario: {has_usage} ({has_usage/len(enriched_selectors)*100:.1f}%)")

    logger.info(f"\n✅ ENRICHMENT COMPLETE!")
    logger.info(f"Review: {output_file}")


if __name__ == "__main__":
    main()
