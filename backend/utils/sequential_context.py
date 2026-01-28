"""
Sequential Context Tracker for Test Execution

Tracks execution state across test steps to improve L1 selector matching.
Maintains context about current module, UI state, and navigation path.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TestExecutionState:
    """
    Maintains execution state across test steps.
    Updated after each step to track context changes.
    """

    # Navigation state
    current_module: Optional[str] = None
    current_section: Optional[str] = None
    navigation_path: List[str] = field(default_factory=list)

    # UI state
    visible_modules: List[str] = field(default_factory=list)
    edit_mode: bool = False
    dialog_open: bool = False
    dropdown_open: bool = False
    expanded_sections: List[str] = field(default_factory=list)

    # History tracking
    previous_modules: List[str] = field(default_factory=list)
    previous_selectors: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0

    # Module dependencies (can be loaded from config)
    module_dependencies: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize module dependencies"""
        # Define which modules depend on which shared components
        self.module_dependencies = {
            "parts": ["entity-attribute"],
            "teststep": ["entity-attribute"],
            "attributes": ["entity-attribute"],
            "project": ["entity-attribute"],
            "task": ["entity-attribute"],
        }


class SequentialContextTracker:
    """
    Tracks and updates execution state based on test steps.
    Provides context-aware module search scope for L1 matching.
    """

    def __init__(self):
        self.logger = logging.getLogger("TA_AI_Project")
        self.state = TestExecutionState()

    def update_from_step(
        self,
        step_text: str,
        jira_module: str,
        selected_selector: Optional[Dict[str, Any]] = None
    ):
        """
        Update state based on executed step.

        Args:
            step_text: The test step description
            jira_module: Module from Jira ticket
            selected_selector: The selector that was used (if found)
        """
        self.state.step_count += 1
        step_lower = step_text.lower()

        # Detect action type
        action = self._detect_action(step_lower)

        self.logger.debug(f"Step {self.state.step_count}: Action={action}, Module={jira_module}")

        # Update state based on action
        if action == "login":
            self._handle_login()

        elif action == "navigate":
            target_module = self._extract_module_name(step_lower, jira_module)
            self._handle_navigation(target_module)

        elif action == "open_detail":
            self._handle_open_detail(jira_module)

        elif action == "expand":
            section = self._extract_section_name(step_lower)
            self._handle_expand_section(section, jira_module)

        elif action == "edit":
            self._handle_edit_mode()

        elif action == "open_dialog":
            self._handle_dialog_open()

        elif action == "open_dropdown":
            self._handle_dropdown_open()

        elif action == "save":
            self._handle_save()

        elif action == "close":
            self._handle_close()

        # Track selector usage
        if selected_selector:
            self.state.previous_selectors.append(selected_selector)
            selector_module = selected_selector.get('module', '')
            if selector_module and selector_module != self.state.current_module:
                self.logger.info(f"Module switch: {self.state.current_module} → {selector_module}")
                # Don't override current_module unless it's a major navigation
                # (expand/edit actions already handle this)

        self.logger.debug(f"State after step: module={self.state.current_module}, "
                         f"visible={self.state.visible_modules}, edit={self.state.edit_mode}")

    def get_search_scope(self, jira_module: str) -> List[str]:
        """
        Get the list of modules to search based on current state.

        Args:
            jira_module: Module from Jira ticket

        Returns:
            List of module names to search in
        """
        if not self.state.visible_modules:
            # No state yet, use Jira module
            return [jira_module]

        # Use visible modules from state
        scope = list(self.state.visible_modules)

        # Always include Jira module as fallback
        if jira_module and jira_module not in scope:
            scope.append(jira_module)

        self.logger.debug(f"Search scope: {scope}")
        return scope

    def get_state_score_boost(self, selector: Dict[str, Any]) -> int:
        """
        Calculate score boost based on how well selector matches current state.

        Args:
            selector: Selector object from JSON

        Returns:
            Score boost (0-30)
        """
        score = 0
        selector_module = selector.get('module', '').lower()
        selector_context = selector.get('context', [])

        # Boost for current module
        if self.state.current_module and selector_module == self.state.current_module.lower():
            score += 20
            self.logger.debug(f"Current module match: +20 (module={selector_module})")

        # Boost for visible modules
        elif selector_module in [m.lower() for m in self.state.visible_modules]:
            score += 15
            self.logger.debug(f"Visible module match: +15 (module={selector_module})")

        # Boost for recent modules
        elif selector_module in [m.lower() for m in self.state.previous_modules[-3:]]:
            score += 10
            self.logger.debug(f"Recent module match: +10 (module={selector_module})")

        # Boost based on UI state
        if self.state.edit_mode:
            if 'edit' in selector_context or 'input' in selector_context:
                score += 8
                self.logger.debug(f"Edit mode match: +8")

        if self.state.dialog_open:
            if 'dialog' in selector_context:
                score += 8
                self.logger.debug(f"Dialog context match: +8")

        if self.state.dropdown_open:
            if 'dropdown' in selector_context or 'menu-item' in selector_context:
                score += 8
                self.logger.debug(f"Dropdown context match: +8")

        # Boost for current section match
        if self.state.current_section:
            section_lower = self.state.current_section.lower()
            selector_value = selector.get('value', '').lower()
            if section_lower in selector_value or section_lower in selector_module:
                score += 10
                self.logger.debug(f"Section match: +10 (section={self.state.current_section})")

        return score

    # ============================================
    # Action Detection
    # ============================================

    def _detect_action(self, step_lower: str) -> str:
        """Detect the action type from step text"""

        if 'login' in step_lower:
            return "login"

        if any(kw in step_lower for kw in ['navigate', 'go to', 'open page']):
            return "navigate"

        # Detect "click on [item] named as X" → opens detail view
        if 'named as' in step_lower and 'click' in step_lower:
            return "open_detail"

        if any(kw in step_lower for kw in ['expand', 'open accordion', 'open panel']):
            return "expand"

        if 'edit' in step_lower and ('button' in step_lower or 'click' in step_lower):
            return "edit"

        if 'save' in step_lower:
            return "save"

        if any(kw in step_lower for kw in ['close', 'cancel']):
            return "close"

        # Dropdown detection
        if '... +' in step_lower or ('click' in step_lower and ('dropdown' in step_lower or 'menu' in step_lower)):
            return "open_dropdown"

        # Dialog detection (create/add buttons often open dialogs)
        if 'create' in step_lower or 'add new' in step_lower:
            return "open_dialog"

        return "interact"

    # ============================================
    # State Update Handlers
    # ============================================

    def _handle_login(self):
        """Handle login action"""
        self.state.current_module = "login"
        self.state.visible_modules = ["login"]
        self.state.navigation_path.append("login")
        self.logger.info("State: Logged in")

    def _handle_navigation(self, target_module: str):
        """Handle navigation to a module"""
        self.state.current_module = target_module
        self.state.visible_modules = [target_module]
        self.state.navigation_path.append(target_module)

        # Reset UI state on navigation
        self.state.edit_mode = False
        self.state.dialog_open = False
        self.state.dropdown_open = False
        self.state.expanded_sections = []

        self.logger.info(f"State: Navigated to {target_module}")

    def _handle_open_detail(self, base_module: str):
        """Handle opening detail view (clicking on a row)"""
        # When clicking on a row, we transition to detail view
        self.state.current_module = "DetailView"
        self.state.visible_modules = [base_module, "DetailView"]
        self.state.navigation_path.append("DetailView")

        # Reset UI state
        self.state.edit_mode = False
        self.state.dialog_open = False
        self.state.dropdown_open = False
        self.state.expanded_sections = []

        self.logger.info(f"State: Opened detail view (base={base_module}, visible={self.state.visible_modules})")

    def _handle_expand_section(self, section: str, base_module: str):
        """Handle accordion/section expansion"""
        self.state.current_section = section
        self.state.expanded_sections.append(section)

        # Map section to module
        section_lower = section.lower()
        section_module = section_lower  # Default: section name = module name

        # Update current module to the section's module
        self.state.current_module = section_module

        # Build visible modules
        visible = [base_module, section_module]

        # Add dependent modules
        deps = self.state.module_dependencies.get(section_module, [])
        visible.extend(deps)

        self.state.visible_modules = list(set(visible))  # Remove duplicates

        self.logger.info(f"State: Expanded {section} section, visible_modules={self.state.visible_modules}")

    def _handle_edit_mode(self):
        """Handle entering edit mode"""
        self.state.edit_mode = True

        # In edit mode, prioritize entity-attribute (used for dropdowns/inputs)
        if self.state.current_module:
            deps = self.state.module_dependencies.get(self.state.current_module, [])
            if deps:
                # Make dependencies primary (they contain the input fields)
                self.state.visible_modules = [self.state.current_module] + deps
                self.logger.info(f"State: Edit mode active, visible_modules={self.state.visible_modules}")

    def _handle_dialog_open(self):
        """Handle dialog/modal opening"""
        self.state.dialog_open = True
        # Dialogs often use create-new module
        if "create-new" not in self.state.visible_modules:
            self.state.visible_modules.append("create-new")
        self.logger.info("State: Dialog opened")

    def _handle_dropdown_open(self):
        """Handle dropdown menu opening"""
        self.state.dropdown_open = True
        # Dropdown items might be in create-new module
        if "create-new" not in self.state.visible_modules:
            self.state.visible_modules.append("create-new")
        self.logger.info("State: Dropdown opened")

    def _handle_save(self):
        """Handle save action"""
        # Save typically closes edit mode
        self.state.edit_mode = False
        self.state.dialog_open = False
        self.logger.info("State: Saved (edit mode closed)")

    def _handle_close(self):
        """Handle close/cancel action"""
        self.state.edit_mode = False
        self.state.dialog_open = False
        self.state.dropdown_open = False
        self.logger.info("State: Closed (reset UI state)")

    # ============================================
    # Extraction Helpers
    # ============================================

    def _extract_module_name(self, step_lower: str, default: str) -> str:
        """Extract module name from step text"""
        # Common module names
        modules = ['teststep', 'project', 'task', 'parts', 'attributes',
                   'entity-attribute', 'create-new']

        for module in modules:
            if module in step_lower:
                return module

        return default

    def _extract_section_name(self, step_lower: str) -> str:
        """Extract section name from step text (for accordion expansion)"""
        # Common section names
        sections = ['parts', 'attributes', 'measurements', 'relations',
                    'requirements', 'attachments']

        for section in sections:
            if section in step_lower:
                return section.capitalize()

        # Try to extract from patterns like "expand XYZ accordion"
        if 'accordion' in step_lower or 'panel' in step_lower:
            words = step_lower.split()
            for i, word in enumerate(words):
                if word in ['accordion', 'panel'] and i > 0:
                    return words[i-1].capitalize()

        return "Unknown"

    def reset(self):
        """Reset state for new test"""
        self.state = TestExecutionState()
        self.logger.info("State: Reset for new test")
