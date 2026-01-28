"""
Selector Loader V2 - With Sequential Context Support

Enhanced version that uses sequential context tracking for improved L1 matching.
Implements score-based ranking instead of first-match approach.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from utils.sequential_context import SequentialContextTracker


class SelectorLoaderV2:
    """
    Enhanced selector loader with sequential context support.
    Provides score-based matching using state tracking across steps.
    """

    def __init__(
        self,
        selectors_file: str = "Selectors_Folder/selectors.json",
        use_sequential_context: bool = True
    ):
        """
        Initialize selector loader V2.

        Args:
            selectors_file: Path to selectors.json file
            use_sequential_context: Enable sequential context tracking
        """
        self.logger = logging.getLogger("TA_AI_Project")
        self.selectors_file = Path(selectors_file)
        self.selectors = []
        self.metadata = {}
        self.use_sequential_context = use_sequential_context

        # Initialize context tracker
        if use_sequential_context:
            self.context_tracker = SequentialContextTracker()
            self.logger.info("Sequential context tracking ENABLED")
        else:
            self.context_tracker = None
            self.logger.info("Sequential context tracking DISABLED (V1.0 mode)")

        self.load_selectors()

    def load_selectors(self):
        """Load selectors from JSON file."""
        try:
            if not self.selectors_file.exists():
                self.logger.warning(f"Selectors file not found: {self.selectors_file}")
                return

            with open(self.selectors_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Support both formats:
            # 1. New enriched format: {"metadata": {...}, "selectors": [...]}
            # 2. Old format: [...]
            if isinstance(data, dict) and 'selectors' in data:
                self.selectors = data['selectors']
                self.metadata = data.get('metadata', {})
                phase = self.metadata.get('phase', 'unknown')
                self.logger.info(f"Loaded enriched selectors (Phase: {phase})")
            else:
                # Old format: array of selectors
                self.selectors = data
                self.metadata = {}
                self.logger.info(f"Loaded selectors (old format)")

            self.logger.info(f"Total selectors loaded: {len(self.selectors)}")

        except Exception as e:
            self.logger.error(f"Error loading selectors: {e}")
            self.selectors = []
            self.metadata = {}

    def find_best_selector(
        self,
        step_text: str,
        module: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching selector for a test step.

        V2 Implementation:
        - Uses sequential context if enabled
        - Score-based ranking (not first-match)
        - State-aware module scope

        Args:
            step_text: Test step description
            module: Module context from Jira

        Returns:
            Best matching selector object or None
        """
        # Extract keywords from step text
        keywords = self._extract_keywords(step_text)

        # Get search scope (V2: from state, V1: from module)
        # State is already updated by step executor before this is called
        if self.use_sequential_context and self.context_tracker:
            search_modules = self.context_tracker.get_search_scope(module)
        else:
            search_modules = [module] if module else []

        self.logger.info(f"L1 Search: keywords={keywords}, modules={search_modules}")

        # Find and score candidates
        candidates = self._find_and_score_candidates(
            keywords,
            search_modules,
            step_text
        )

        if not candidates:
            self.logger.warning(f"L1 FAIL: No selectors found for keywords={keywords}")
            return None

        # Return best candidate
        best_selector, best_score = candidates[0]

        self.logger.info(f"L1 SUCCESS: Found '{best_selector.get('attr')}' "
                        f"(score={best_score}, module={best_selector.get('module')})")

        return best_selector

    def _find_and_score_candidates(
        self,
        keywords: List[str],
        search_modules: List[str],
        step_text: str
    ) -> List[tuple[Dict[str, Any], int]]:
        """
        Find and score selector candidates.

        Returns:
            List of (selector, score) tuples, sorted by score descending
        """
        candidates = []

        for selector in self.selectors:
            # Check module scope
            selector_module = selector.get('module', '').lower()

            if search_modules:
                # V2: Check if in search scope
                if not any(sm.lower() in selector_module for sm in search_modules):
                    continue
            # else: No module filter (search all)

            # Calculate score
            score = self._calculate_selector_score(selector, keywords, step_text)

            if score > 0:
                candidates.append((selector, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        self.logger.debug(f"Found {len(candidates)} candidates")
        if candidates:
            # Show top 3
            for i, (sel, score) in enumerate(candidates[:3]):
                self.logger.debug(f"  #{i+1}: {sel.get('attr')} (score={score}, module={sel.get('module')})")

        return candidates

    def _calculate_selector_score(
        self,
        selector: Dict[str, Any],
        keywords: List[str],
        step_text: str
    ) -> int:
        """
        Calculate match score for a selector.

        Scoring components:
        1. Keyword matches in attr/value/label (5 points each)
        2. Keyword matches in context (8 points each) - if available
        3. Priority from selector (0-10 points) - if available
        4. State-based boost (0-30 points) - if sequential context enabled

        Returns:
            Total score
        """
        score = 0

        # 1. Keyword matching in basic fields
        attr = selector.get('attr', '').lower()
        value = selector.get('value', '').lower()
        label = selector.get('label', '').lower()

        for keyword in keywords:
            kw = keyword.lower()
            if kw in attr or kw in value or kw in label:
                score += 5

        # 2. Keyword matching in context (V2.0 enrichment)
        context = selector.get('context', [])
        if context:
            for keyword in keywords:
                if keyword.lower() in [c.lower() for c in context]:
                    score += 8  # Context matches are stronger

        # 3. Priority boost (V2.0 enrichment)
        priority = selector.get('priority', 0)
        if priority:
            score += priority

        # 4. State-based boost (V2 sequential context)
        if self.use_sequential_context and self.context_tracker:
            state_boost = self.context_tracker.get_state_score_boost(selector)
            score += state_boost

        return score

    def _extract_keywords(self, step_text: str) -> List[str]:
        """
        Extract keywords from step text.

        Args:
            step_text: Test step description

        Returns:
            List of keywords
        """
        # Same as V1.0 for compatibility
        step_lower = step_text.lower()
        keywords = []

        # CHECK FOR MESSAGE/NOTIFICATION VERIFICATION FIRST (highest priority)
        is_verification = False
        if 'message should be displayed' in step_lower or 'should display' in step_lower:
            keywords.extend(['message', 'notification', 'alert', 'snackbar', 'success', 'toast'])
            is_verification = True
        elif 'displayed' in step_lower or 'appears' in step_lower or 'shown' in step_lower:
            keywords.extend(['message', 'notification', 'alert'])
            is_verification = True

        # Extract specific patterns (SPECIFIC keywords first, then GENERIC)
        # Specific UI elements
        if '... +' in step_text or 'more' in step_lower or 'vertical' in step_lower:
            keywords.extend(['showmoreverticalbtn', 'showmore', 'vertical', 'more'])
        # Only add 'selectproject' if NOT a dropdown/select step
        if ('project' in step_lower or 'product' in step_lower) and not ('dropdown' in step_lower or 'select' in step_lower):
            keywords.extend(['selectproject', 'project', 'product'])
        elif 'project' in step_lower or 'product' in step_lower:
            # For dropdown/select steps, just add project without selectproject
            keywords.extend(['project', 'product'])
        if 'accordion' in step_lower:
            keywords.extend(['accordion', 'panel'])
        if 'parts' in step_lower:
            keywords.append('parts')
        if 'teststep' in step_lower:
            keywords.append('teststep')
        if 'name' in step_lower and 'type' in step_lower:
            keywords.extend(['name', 'input'])

        # Action buttons (specific) - SKIP if this is a verification step
        if not is_verification:
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
            # Always add dropdown keywords for dropdown/select steps
            keywords.extend(['dropdown', 'select', 'type'])

        return keywords

    def build_selector(self, selector_obj: Dict[str, Any], value_override: str = None) -> str:
        """
        Build Playwright selector string from selector object.

        Args:
            selector_obj: Selector dictionary from JSON
            value_override: Optional value to use instead of selector's value
                          (used when trying possibleValues for dynamic selectors)

        Returns:
            CSS selector string for Playwright
        """
        attr = selector_obj.get('attr', '')
        value = value_override or selector_obj.get('value', '')
        tagName = selector_obj.get('tagName', '').lower()
        className = selector_obj.get('className', '')

        # Support both 'isDynamic' (new enriched format) and 'dynamic' (old format)
        is_dynamic = selector_obj.get('isDynamic', selector_obj.get('dynamic', False))

        # Handle 'attr.' prefix from old format (clean it up)
        if attr.startswith('attr.'):
            attr = attr.replace('attr.', '')

        # Build base attribute selector
        if is_dynamic and not value_override:
            # Dynamic selector without specific value - use wildcard
            attr_selector = f"[{attr}]"
        else:
            # Static selector OR dynamic with specific value
            # Remove {{...}} markers if present
            clean_value = value.replace('{{', '').replace('}}', '')
            attr_selector = f'[{attr}="{clean_value}"]'

        # Enhance with tag and class if available (for better specificity)
        # This prevents ambiguous matches
        if tagName and className:
            # Extract first significant class (e.g., "mat-mdc-autocomplete-trigger")
            classes = className.split()
            # Prioritize specific classes
            specific_classes = [c for c in classes if 'trigger' in c or 'btn' in c or 'input' in c]
            if specific_classes:
                return f"{tagName}.{specific_classes[0]}{attr_selector}"
            elif classes:
                return f"{tagName}.{classes[0]}{attr_selector}"
            else:
                return f"{tagName}{attr_selector}"
        elif tagName:
            return f"{tagName}{attr_selector}"
        else:
            return attr_selector

    def update_state_for_step(self, step_text: str, module: str):
        """
        Update sequential context state for a step (without selector search).
        Call this to track state transitions even when L1 is skipped.

        Args:
            step_text: Step description
            module: Module context
        """
        if self.use_sequential_context and self.context_tracker:
            self.context_tracker.update_from_step(step_text, module or "", None)

    def reset_state(self):
        """Reset sequential context state for new test"""
        if self.context_tracker:
            self.context_tracker.reset()
            self.logger.info("Sequential context state reset")

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        if not self.context_tracker:
            return {"enabled": False}

        state = self.context_tracker.state
        return {
            "enabled": True,
            "current_module": state.current_module,
            "current_section": state.current_section,
            "visible_modules": state.visible_modules,
            "edit_mode": state.edit_mode,
            "dialog_open": state.dialog_open,
            "step_count": state.step_count,
        }
