"""
Selector Loader Utility

Loads and searches custom selectors from selectors.json file.
Provides methods to find selectors by keywords, module, and action type.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any


class SelectorLoader:
    """
    Utility class to load and search custom selectors from JSON file.
    """

    def __init__(self, selectors_file: str = "Selectors_Folder/selectors.json"):
        """
        Initialize selector loader.

        Args:
            selectors_file: Path to selectors.json file
        """
        self.logger = logging.getLogger("TA_AI_Project")
        self.selectors_file = Path(selectors_file)
        self.selectors = []
        self.load_selectors()

    def load_selectors(self):
        """Load selectors from JSON file."""
        try:
            if not self.selectors_file.exists():
                self.logger.warning(f"Selectors file not found: {self.selectors_file}")
                return

            with open(self.selectors_file, 'r', encoding='utf-8') as f:
                self.selectors = json.load(f)

            self.logger.info(f"Loaded {len(self.selectors)} selectors from {self.selectors_file}")

        except Exception as e:
            self.logger.error(f"Error loading selectors: {e}")
            self.selectors = []

    def search_by_keywords(
        self,
        keywords: List[str],
        module: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search selectors by keywords in attr, value, or label fields.

        Args:
            keywords: List of keywords to search for
            module: Optional module name to filter by

        Returns:
            List of matching selectors
        """
        matches = []

        for selector in self.selectors:
            # Filter by module if specified
            if module:
                selector_module = selector.get('module', '').lower()
                if module.lower() not in selector_module:
                    continue

            # Check if any keyword matches
            attr = selector.get('attr', '').lower()
            value = selector.get('value', '').lower()
            label = selector.get('label', '').lower()

            for keyword in keywords:
                kw = keyword.lower()
                if kw in attr or kw in value or kw in label:
                    matches.append(selector)
                    break

        return matches

    def search_by_action(
        self,
        action_keywords: List[str],
        module: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search selectors by action type (button, input, dropdown, etc.).

        Args:
            action_keywords: Keywords like ['save', 'button'], ['edit', 'btn']
            module: Optional module name to filter

        Returns:
            List of matching selectors
        """
        return self.search_by_keywords(action_keywords, module)

    def build_selector(self, selector_obj: Dict[str, Any]) -> str:
        """
        Build Playwright selector string from selector object.

        Args:
            selector_obj: Selector dictionary from JSON

        Returns:
            CSS selector string for Playwright
        """
        attr = selector_obj.get('attr', '')
        value = selector_obj.get('value', '')
        is_dynamic = selector_obj.get('dynamic', False)

        # Handle attribute selectors
        if attr.startswith('attr.'):
            # Dynamic attribute - just use attribute name
            attr = attr.replace('attr.', '')
            if is_dynamic:
                # Dynamic value - just check attribute exists
                return f"[{attr}]"
            else:
                # Static value
                return f'[{attr}="{value}"]'
        else:
            # Regular data attribute
            if is_dynamic:
                return f"[{attr}]"
            else:
                return f'[{attr}="{value}"]'

    def find_best_selector(
        self,
        step_text: str,
        module: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching selector for a test step.

        Args:
            step_text: Test step description
            module: Module context from Jira

        Returns:
            Best matching selector object or None
        """
        # Extract keywords from step text
        keywords = self._extract_keywords(step_text)

        # Try without module filter first (modals/dialogs can be from different modules)
        matches = self.search_by_keywords(keywords)

        # If too many matches, try with module filter to narrow down
        if len(matches) > 3:
            matches_with_module = self.search_by_keywords(keywords, module)
            if matches_with_module:
                matches = matches_with_module

        if matches:
            # Return first match (could be improved with scoring)
            return matches[0]

        return None

    def _extract_keywords(self, step_text: str) -> List[str]:
        """
        Extract keywords from step text.

        Args:
            step_text: Test step description

        Returns:
            List of keywords
        """
        # Common action words to extract
        step_lower = step_text.lower()
        keywords = []

        # Extract specific patterns (SPECIFIC keywords first, then GENERIC)
        # Specific UI elements
        if '... +' in step_text or 'more' in step_lower or 'vertical' in step_lower:
            keywords.extend(['showmoreverticalbtn', 'showmore', 'vertical', 'more'])
        if 'project' in step_lower or 'product' in step_lower:
            keywords.extend(['selectproject', 'project', 'product'])
        if 'accordion' in step_lower:
            keywords.extend(['accordion', 'panel'])
        if 'parts' in step_lower:
            keywords.append('parts')
        if 'teststep' in step_lower:
            keywords.append('teststep')
        if 'name' in step_lower and 'type' in step_lower:
            keywords.extend(['name', 'input'])

        # Action buttons (specific)
        if 'save' in step_lower:
            keywords.extend(['save', 'btn', 'button'])
        if 'edit' in step_lower:
            keywords.extend(['edit', 'btn', 'button'])
        if 'delete' in step_lower:
            keywords.extend(['delete', 'btn', 'button'])
        if 'remove' in step_lower:
            keywords.extend(['remove', 'btn', 'button'])
        if 'close' in step_lower or 'cancel' in step_lower:
            keywords.extend(['close', 'cancel', 'btn'])

        # Generic patterns (only if no specific match above)
        if 'navigate' in step_lower:
            keywords.extend(['navigate', 'nav', 'menu'])
        if 'dropdown' in step_lower or 'select' in step_lower:
            # Only add generic dropdown keywords if we haven't already added specific ones
            if 'selectproject' not in keywords:
                keywords.extend(['dropdown', 'select', 'type'])

        return keywords

    def get_selectors_for_module(self, module: str) -> List[Dict[str, Any]]:
        """
        Get all selectors for a specific module.

        Args:
            module: Module name

        Returns:
            List of selectors for the module
        """
        return [
            s for s in self.selectors
            if module.lower() in s.get('module', '').lower()
        ]
