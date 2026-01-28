"""
Module Mapper Utility

Maps Jira module names to web application UI element names.

Example:
  Jira Component: "Teststep"
  Web App shows: "Runs"

  Mapping: teststep -> Runs
"""

import logging
from typing import Optional, List


class ModuleMapper:
    """
    Maps Jira module names to web application UI names.
    """

    def __init__(self, config: dict):
        """
        Initialize module mapper.

        Args:
            config: Configuration dict containing module_name and alternative mappings
        """
        self.logger = logging.getLogger("TA_AI_Project")
        self.config = config

        # Load mappings from config
        self.module_names = config.get('module_name', [])
        self.alternatives = config.get('alternative', [])

        # Create mapping dict
        self.mapping = {}
        if len(self.module_names) == len(self.alternatives):
            for i, module in enumerate(self.module_names):
                self.mapping[module.lower()] = self.alternatives[i]

            self.logger.info(f"Module mapping loaded: {self.mapping}")
        else:
            self.logger.warning("Module mapping configuration mismatch!")

    def get_web_name(self, jira_module: str) -> str:
        """
        Get web application name for a Jira module.

        Args:
            jira_module: Module name from Jira (e.g., "Teststep")

        Returns:
            Web application name (e.g., "Runs")
        """
        # Normalize to lowercase for comparison
        module_lower = jira_module.lower()

        # Check mapping
        if module_lower in self.mapping:
            web_name = self.mapping[module_lower]
            self.logger.info(f"Module mapping: '{jira_module}' -> '{web_name}'")
            return web_name

        # No mapping found - return original
        self.logger.warning(f"No mapping found for module '{jira_module}', using original")
        return jira_module

    def get_all_web_names(self, jira_module: str) -> List[str]:
        """
        Get all possible web names (mapped + original).

        Args:
            jira_module: Module name from Jira

        Returns:
            List of possible names to try
        """
        web_name = self.get_web_name(jira_module)

        # Return both mapped name and original (in case mapping is wrong)
        names = [web_name]
        if web_name.lower() != jira_module.lower():
            names.append(jira_module)

        return names
