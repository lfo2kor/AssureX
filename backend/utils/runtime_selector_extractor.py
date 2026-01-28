"""
Runtime Selector Extractor

Extracts selectors from the RUNNING APPLICATION instead of source code.
Uses Playwright to navigate through test steps and capture actual DOM attributes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from playwright.sync_api import Page


class RuntimeSelectorExtractor:
    """
    Extracts data-* selectors from running application.

    Process:
    1. Navigate through test steps using L2 (generic patterns)
    2. At each step, extract ALL data-* attributes from page
    3. Record which elements are clickable, visible, etc.
    4. Save to JSON with step context
    """

    def __init__(self, page: Page, config: Dict, logger: logging.Logger):
        """
        Initialize runtime extractor.

        Args:
            page: Playwright page object
            config: Configuration dictionary
            logger: Logger instance
        """
        self.page = page
        self.config = config
        self.logger = logger
        self.extracted_selectors = []

    def extract_from_current_page(
        self,
        step_context: Dict[str, Any],
        target_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all data-* selectors from current page.

        Args:
            step_context: Context about current step
                {
                    'step_num': 4,
                    'step_text': 'open parts accordion',
                    'module': 'DetailView',
                    'action': 'expand'
                }
            target_text: Optional text to identify target element
                         (e.g., 'Save', 'Parts', 'Type')

        Returns:
            List of extracted selectors with metadata
        """
        self.logger.info(f"Extracting selectors from page for step {step_context['step_num']}")

        # JavaScript to extract selectors from DOM
        extraction_script = """
            (targetText) => {
                const selectors = [];

                // Find all elements with data-* attributes
                const allElements = document.querySelectorAll('*');

                allElements.forEach(el => {
                    // Get all data-* attributes
                    const dataAttrs = Array.from(el.attributes)
                        .filter(attr => attr.name.startsWith('data-'));

                    if (dataAttrs.length === 0) return;

                    // Get element properties
                    const rect = el.getBoundingClientRect();
                    const isVisible = (
                        rect.width > 0 &&
                        rect.height > 0 &&
                        el.offsetParent !== null &&
                        window.getComputedStyle(el).visibility !== 'hidden' &&
                        window.getComputedStyle(el).display !== 'none'
                    );

                    // Check if clickable
                    const isClickable = (
                        el.matches('button, a, input, select, textarea, [role="button"], [role="link"], [role="tab"]') ||
                        el.onclick !== null ||
                        window.getComputedStyle(el).cursor === 'pointer' ||
                        el.hasAttribute('ng-click') ||
                        el.hasAttribute('(click)')
                    );

                    // Get text content
                    const textContent = el.textContent.trim();
                    const innerText = el.innerText ? el.innerText.trim() : '';

                    // Check if this element contains target text
                    const containsTarget = targetText ?
                        (textContent.toLowerCase().includes(targetText.toLowerCase()) ||
                         innerText.toLowerCase().includes(targetText.toLowerCase())) : false;

                    // Extract each data-* attribute
                    dataAttrs.forEach(attr => {
                        selectors.push({
                            attr: attr.name,
                            value: attr.value,
                            tagName: el.tagName.toLowerCase(),
                            className: el.className,
                            id: el.id || null,
                            textContent: textContent.substring(0, 100),
                            innerText: innerText.substring(0, 100),
                            isVisible: isVisible,
                            isClickable: isClickable,
                            role: el.getAttribute('role'),
                            ariaLabel: el.getAttribute('aria-label'),
                            type: el.getAttribute('type'),
                            placeholder: el.getAttribute('placeholder'),
                            name: el.getAttribute('name'),
                            containsTargetText: containsTarget,
                            // Position/size
                            width: Math.round(rect.width),
                            height: Math.round(rect.height),
                            top: Math.round(rect.top),
                            left: Math.round(rect.left)
                        });
                    });
                });

                return selectors;
            }
        """

        try:
            # Execute extraction
            selectors = self.page.evaluate(extraction_script, target_text)

            # Enrich with step context
            for selector in selectors:
                selector.update({
                    'step_num': step_context['step_num'],
                    'step_text': step_context['step_text'],
                    'module': step_context.get('module', ''),
                    'action': step_context.get('action', ''),
                    'extractedDate': datetime.now().isoformat(),
                    'extractionMode': 'runtime',
                    'pageUrl': self.page.url
                })

            # Filter and prioritize
            filtered = self._filter_and_prioritize(selectors, step_context, target_text)

            self.extracted_selectors.extend(filtered)

            self.logger.info(f"  Extracted {len(selectors)} selectors, kept {len(filtered)} after filtering")

            # Log top candidates if target text provided
            if target_text and filtered:
                self.logger.info(f"  Top candidates for '{target_text}':")
                for sel in filtered[:3]:
                    self.logger.info(f"    [{sel['attr']}=\"{sel['value']}\"] "
                                   f"(clickable={sel['isClickable']}, visible={sel['isVisible']}, "
                                   f"tag={sel['tagName']}, text='{sel['textContent'][:30]}')")

            return filtered

        except Exception as e:
            self.logger.error(f"Error extracting selectors: {e}")
            return []

    def _filter_and_prioritize(
        self,
        selectors: List[Dict],
        step_context: Dict,
        target_text: Optional[str]
    ) -> List[Dict]:
        """
        Filter and prioritize extracted selectors.

        Filtering rules:
        - Keep only visible OR clickable elements (hidden elements are useless)
        - If target_text provided, prioritize elements containing that text
        - Remove duplicates

        Args:
            selectors: Raw extracted selectors
            step_context: Step context
            target_text: Target text to search for

        Returns:
            Filtered and prioritized list
        """
        filtered = []
        seen = set()

        for sel in selectors:
            # Create unique key
            key = f"{sel['attr']}={sel['value']}"

            if key in seen:
                continue

            # Filter rules
            if not sel['isVisible'] and not sel['isClickable']:
                # Skip hidden, non-interactive elements
                continue

            # Prioritize elements with target text
            if target_text:
                sel['priority'] = 100 if sel['containsTargetText'] else 50
            else:
                sel['priority'] = 50

            # Boost priority for clickable elements
            if sel['isClickable']:
                sel['priority'] += 20

            # Boost priority for visible elements
            if sel['isVisible']:
                sel['priority'] += 10

            # Boost priority for certain tag types
            if sel['tagName'] in ['button', 'a', 'input']:
                sel['priority'] += 15

            filtered.append(sel)
            seen.add(key)

        # Sort by priority
        filtered.sort(key=lambda x: x['priority'], reverse=True)

        return filtered

    def extract_specific_element(
        self,
        selector_string: str,
        step_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract data-* attributes from a specific element.

        Useful when L2 succeeds - we can extract the exact selector used.

        Args:
            selector_string: Playwright selector that worked (e.g., "button:has-text('Save')")
            step_context: Step context

        Returns:
            Extracted selector with metadata, or None
        """
        try:
            element = self.page.locator(selector_string).first

            # Extract all attributes from this specific element
            data = element.evaluate("""
                el => {
                    const rect = el.getBoundingClientRect();
                    const dataAttrs = Array.from(el.attributes)
                        .filter(attr => attr.name.startsWith('data-'));

                    return {
                        dataAttrs: dataAttrs.map(a => ({name: a.name, value: a.value})),
                        tagName: el.tagName.toLowerCase(),
                        className: el.className,
                        id: el.id || null,
                        textContent: el.textContent.trim().substring(0, 100),
                        isVisible: el.offsetParent !== null,
                        isClickable: true,  // We know it's clickable since we clicked it!
                        role: el.getAttribute('role'),
                        ariaLabel: el.getAttribute('aria-label'),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    };
                }
            """)

            if not data['dataAttrs']:
                self.logger.warning(f"Element has no data-* attributes: {selector_string}")
                return None

            # Create selector entry (use first data-* attribute)
            primary_attr = data['dataAttrs'][0]

            selector_entry = {
                'attr': primary_attr['name'],
                'value': primary_attr['value'],
                'tagName': data['tagName'],
                'className': data['className'],
                'id': data['id'],
                'textContent': data['textContent'],
                'isVisible': data['isVisible'],
                'isClickable': data['isClickable'],
                'role': data['role'],
                'ariaLabel': data['ariaLabel'],
                'width': data['width'],
                'height': data['height'],
                'step_num': step_context['step_num'],
                'step_text': step_context['step_text'],
                'module': step_context.get('module', ''),
                'action': step_context.get('action', ''),
                'extractedDate': datetime.now().isoformat(),
                'extractionMode': 'runtime_specific',
                'pageUrl': self.page.url,
                'learned_from': selector_string,
                'priority': 100,  # High priority since we know it works
                'allDataAttrs': data['dataAttrs']  # Keep all data-* for reference
            }

            self.extracted_selectors.append(selector_entry)

            self.logger.info(f"  Extracted from working selector: [{primary_attr['name']}=\"{primary_attr['value']}\"]")

            return selector_entry

        except Exception as e:
            self.logger.error(f"Error extracting specific element: {e}")
            return None

    def save_to_json(self, output_file: str):
        """
        Save extracted selectors to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Deduplicate by attr+value
        unique_selectors = {}
        for sel in self.extracted_selectors:
            key = f"{sel['attr']}={sel['value']}"
            if key not in unique_selectors or sel['priority'] > unique_selectors[key]['priority']:
                unique_selectors[key] = sel

        data = {
            'metadata': {
                'extractionDate': datetime.now().isoformat(),
                'extractionMode': 'runtime',
                'testUrl': self.config.get('test_url', 'unknown'),
                'totalSelectors': len(unique_selectors),
                'totalSteps': len(set(s['step_num'] for s in self.extracted_selectors))
            },
            'selectors': sorted(unique_selectors.values(), key=lambda x: x['step_num'])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Runtime selectors saved to: {output_path}")
        self.logger.info(f"  Total unique selectors: {len(unique_selectors)}")

    def get_extracted_count(self) -> int:
        """Get number of extracted selectors."""
        return len(self.extracted_selectors)
