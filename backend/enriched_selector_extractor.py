"""
Enriched Selector Extractor - Complete in One File
==================================================

Extracts and enriches web element selectors from codebase.

Phases:
  Phase 1: HTMLParser - Extract static selectors from HTML
  Phase 2: DynamicExtractor - Extract dynamic values from TypeScript
  Phase 3: ContextEnricher - Add semantic context and keywords
  Phase 4: PriorityCalculator - Calculate priority scores

Author: AI Assistant
Date: 2025-11-03
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys


# ============================================================
# PHASE 1: HTML PARSER
# ============================================================
class HTMLParser:
    """
    Extracts basic selectors from HTML files.

    Features:
    - Finds all data-* attributes
    - Detects element types
    - Identifies module from folder structure
    - Extracts parent context
    """

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.selectors = []
        self.selector_id_counter = 0

    def extract_static_selectors(self) -> List[Dict[str, Any]]:
        """
        Extract all static selectors from HTML files in codebase.

        Returns:
            List of selector dictionaries
        """
        print(f"\n{'='*60}")
        print("PHASE 1: Extracting Static Selectors from HTML")
        print(f"{'='*60}")
        print(f"Scanning: {self.codebase_path}")

        # Find all HTML files
        html_files = list(self.codebase_path.rglob("*.html"))
        print(f"Found {len(html_files)} HTML files")

        # Process each HTML file
        for html_file in html_files:
            try:
                self._process_html_file(html_file)
            except Exception as e:
                print(f"  [WARN] Error processing {html_file.name}: {e}")

        print(f"\n[SUCCESS] Phase 1 Complete: Extracted {len(self.selectors)} selectors")
        return self.selectors

    def _process_html_file(self, html_file: Path):
        """Process a single HTML file and extract selectors."""
        # Read HTML content
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"  [WARN] Could not read {html_file.name}: {e}")
            return

        # Get module name from folder structure
        module_name = self._extract_module_name(html_file)

        # Get relative file path
        try:
            relative_path = html_file.relative_to(self.codebase_path)
        except ValueError:
            relative_path = html_file

        selectors_found = 0

        # Pattern 1: Static selectors - data-attrname="value"
        static_pattern = r'data-([a-zA-Z0-9_-]+)\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(static_pattern, html_content):
            attr_name = match.group(1)
            value = match.group(2)

            # Create selector object
            selector = {
                'id': f"selector_{self.selector_id_counter:04d}",
                'attr': f"data-{attr_name}",
                'value': value,
                'module': module_name,
                'filePath': str(relative_path),
                'elementType': self._extract_element_type(html_content, match.start()),
                'isDynamic': False,
                'parentElement': None,
                'htmlSnippet': self._get_snippet_around_match(html_content, match.start(), match.end()),
                'extractedDate': datetime.now().isoformat()
            }

            self.selectors.append(selector)
            self.selector_id_counter += 1
            selectors_found += 1

        # Pattern 2: Dynamic selectors - [attr.data-attrname]="variableName"
        dynamic_pattern = r'\[attr\.data-([a-zA-Z0-9_-]+)\]\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(dynamic_pattern, html_content):
            attr_name = match.group(1)
            variable_name = match.group(2)

            # Create selector object
            selector = {
                'id': f"selector_{self.selector_id_counter:04d}",
                'attr': f"data-{attr_name}",
                'value': f"{{{{{variable_name}}}}}",  # Mark as {{variableName}}
                'module': module_name,
                'filePath': str(relative_path),
                'elementType': self._extract_element_type(html_content, match.start()),
                'isDynamic': True,
                'parentElement': None,
                'htmlSnippet': self._get_snippet_around_match(html_content, match.start(), match.end()),
                'extractedDate': datetime.now().isoformat()
            }

            self.selectors.append(selector)
            self.selector_id_counter += 1
            selectors_found += 1

        if selectors_found > 0:
            print(f"  [OK] {html_file.name}: {selectors_found} selectors")

    def _extract_module_name(self, file_path: Path) -> str:
        """
        Extract module name from folder structure.

        Example: src/app/create-new/create-new.component.html -> CreateNew
        """
        # Get path parts after 'app' folder
        parts = file_path.parts

        # Find 'app' in path
        try:
            app_index = parts.index('app')
            # Next folder is the module
            if app_index + 1 < len(parts):
                module_folder = parts[app_index + 1]
                # Convert kebab-case to PascalCase
                return self._kebab_to_pascal(module_folder)
        except (ValueError, IndexError):
            pass

        # Check for 'libs' folder
        try:
            libs_index = parts.index('libs')
            if libs_index + 1 < len(parts):
                module_folder = parts[libs_index + 1]
                return self._kebab_to_pascal(module_folder)
        except (ValueError, IndexError):
            pass

        # Fallback: use parent folder name
        return self._kebab_to_pascal(file_path.parent.name)

    def _kebab_to_pascal(self, kebab_str: str) -> str:
        """
        Convert kebab-case to PascalCase.

        Example: create-new -> CreateNew
        """
        words = kebab_str.split('-')
        return ''.join(word.capitalize() for word in words)

    def _extract_element_type(self, html_content: str, match_pos: int) -> str:
        """
        Extract element type from HTML around the match position.
        Looks backwards from match position to find opening tag.
        """
        # Look backwards to find the opening tag
        start_pos = max(0, match_pos - 500)  # Look back up to 500 chars
        snippet = html_content[start_pos:match_pos]

        # Find the last opening tag before our match
        tag_pattern = r'<(\w+[-\w]*)'
        matches = list(re.finditer(tag_pattern, snippet))

        if matches:
            return matches[-1].group(1)
        return 'unknown'

    def _get_snippet_around_match(self, html_content: str, match_start: int, match_end: int, context: int = 100) -> str:
        """Get a snippet of HTML around the match position."""
        start = max(0, match_start - context)
        end = min(len(html_content), match_end + context)
        snippet = html_content[start:end]

        # Truncate if too long
        if len(snippet) > 250:
            snippet = snippet[:250] + "..."

        return snippet.strip()


# ============================================================
# PHASE 2: DYNAMIC EXTRACTOR
# ============================================================
class DynamicExtractor:
    """
    Extracts possible values for dynamic selectors from TypeScript files.

    Features:
    - Finds corresponding .ts file for each HTML file
    - Parses TypeScript to find variable declarations
    - Extracts possible values for dynamic variables
    - Handles array declarations and assignments
    """

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.ts_cache = {}  # Cache TypeScript file contents

    def add_dynamic_values(self, selectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add possible values for dynamic selectors.
        """
        print(f"\n{'='*60}")
        print("PHASE 2: Dynamic Value Extraction")
        print(f"{'='*60}")

        dynamic_selectors = [s for s in selectors if s.get('isDynamic')]
        print(f"Processing {len(dynamic_selectors)} dynamic selectors...")

        # Group selectors by file path for efficiency
        selectors_by_file = {}
        for selector in dynamic_selectors:
            file_path = selector.get('filePath', '')
            if file_path not in selectors_by_file:
                selectors_by_file[file_path] = []
            selectors_by_file[file_path].append(selector)

        # Process each file
        processed_count = 0
        for html_path, file_selectors in selectors_by_file.items():
            ts_path = self._find_typescript_file(html_path)
            if ts_path and ts_path.exists():
                ts_content = self._read_typescript_file(ts_path)
                if ts_content:
                    for selector in file_selectors:
                        self._extract_possible_values(selector, ts_content)
                        processed_count += 1

        print(f"[SUCCESS] Processed {processed_count} dynamic selectors")
        values_found = sum(1 for s in dynamic_selectors if s.get('possibleValues'))
        print(f"[INFO] Found values for {values_found}/{len(dynamic_selectors)} dynamic selectors")

        return selectors

    def _find_typescript_file(self, html_path: str) -> Optional[Path]:
        """
        Find corresponding TypeScript file for an HTML file.

        Example: add-existing.component.html -> add-existing.component.ts
        """
        html_path_obj = Path(html_path)

        # Replace .html with .ts
        ts_filename = html_path_obj.stem + '.ts'
        ts_path = self.codebase_path / html_path_obj.parent / ts_filename

        return ts_path if ts_path.exists() else None

    def _read_typescript_file(self, ts_path: Path) -> Optional[str]:
        """Read TypeScript file content."""
        if ts_path in self.ts_cache:
            return self.ts_cache[ts_path]

        try:
            with open(ts_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.ts_cache[ts_path] = content
                return content
        except Exception as e:
            print(f"  [WARN] Could not read {ts_path.name}: {e}")
            return None

    def _extract_possible_values(self, selector: Dict[str, Any], ts_content: str):
        """
        Extract possible values for a dynamic selector from TypeScript content.
        """
        # Get variable name from selector value
        value = selector.get('value', '')
        variable_name = value.replace('{{', '').replace('}}', '').strip()

        if not variable_name:
            return

        # Try different extraction methods
        possible_values = []

        # Method 1: Array declaration (e.g., options: string[] = ['val1', 'val2'])
        array_values = self._extract_from_array_declaration(variable_name, ts_content)
        if array_values:
            possible_values.extend(array_values)

        # Method 2: String assignments (e.g., this.variable = 'value')
        assignment_values = self._extract_from_assignments(variable_name, ts_content)
        if assignment_values:
            possible_values.extend(assignment_values)

        # Method 3: Object property assignments (e.g., {key: 'value'})
        object_values = self._extract_from_object_properties(variable_name, ts_content)
        if object_values:
            possible_values.extend(object_values)

        # Remove duplicates and add to selector
        if possible_values:
            unique_values = list(dict.fromkeys(possible_values))  # Preserve order
            selector['possibleValues'] = unique_values
            selector['dynamicValueSource'] = 'TypeScript analysis'

    def _extract_from_array_declaration(self, variable_name: str, ts_content: str) -> List[str]:
        """
        Extract values from array declaration.

        Example:
          options: string[] = ['value1', 'value2', 'value3'];
        """
        values = []

        # Pattern: variableName: type[] = [...]
        pattern = rf'{variable_name}\s*:\s*\w+\[\]\s*=\s*\[(.*?)\]'
        matches = re.findall(pattern, ts_content, re.DOTALL)

        for match in matches:
            # Extract quoted strings
            string_pattern = r"['\"]([^'\"]+)['\"]"
            strings = re.findall(string_pattern, match)
            values.extend(strings)

        return values

    def _extract_from_assignments(self, variable_name: str, ts_content: str) -> List[str]:
        """
        Extract values from variable assignments.

        Examples:
          this.variableName = 'value1';
          this.variableName = "value2";
        """
        values = []

        # Pattern: this.variableName = 'value' or "value"
        pattern = rf'this\.{variable_name}\s*=\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, ts_content)
        values.extend(matches)

        # Pattern: variableName = 'value' (without this.)
        pattern2 = rf'\b{variable_name}\s*=\s*["\']([^"\']+)["\']'
        matches2 = re.findall(pattern2, ts_content)
        values.extend(matches2)

        return values

    def _extract_from_object_properties(self, variable_name: str, ts_content: str) -> List[str]:
        """
        Extract values from object property patterns.

        Example:
          const obj = {
            variableName: 'value1'
          };
        """
        values = []

        # Pattern: variableName: 'value'
        pattern = rf'{variable_name}\s*:\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, ts_content)
        values.extend(matches)

        return values


# ============================================================
# PHASE 3: CONTEXT ENRICHER (PLACEHOLDER - Will implement later)
# ============================================================
class ContextEnricher:
    """
    Adds semantic context and keywords to selectors.

    TODO: Implement in Phase 3
    """

    def enrich_context(self, selectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add context keywords and semantic information.

        TODO: Will implement in Phase 3
        """
        print(f"\n{'='*60}")
        print("PHASE 3: Context Enrichment (PLACEHOLDER)")
        print(f"{'='*60}")
        print("[INFO] Phase 3 not yet implemented - will add later")

        return selectors


# ============================================================
# PHASE 4: PRIORITY CALCULATOR (PLACEHOLDER - Will implement later)
# ============================================================
class PriorityCalculator:
    """
    Calculates priority scores for selectors.

    TODO: Implement in Phase 4
    """

    def calculate_priorities(self, selectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate priority score (0-10) for each selector.

        TODO: Will implement in Phase 4
        """
        print(f"\n{'='*60}")
        print("PHASE 4: Priority Calculation (PLACEHOLDER)")
        print(f"{'='*60}")
        print("[INFO] Phase 4 not yet implemented - will add later")

        return selectors


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
class EnrichedSelectorExtractor:
    """
    Main orchestrator that runs all phases and manages checkpoints.
    """

    def __init__(self, codebase_path: str, output_dir: str = "Selectors_Folder"):
        self.codebase_path = Path(codebase_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Verify codebase path exists
        if not self.codebase_path.exists():
            raise FileNotFoundError(f"Codebase path not found: {self.codebase_path}")

        print(f"\n{'='*60}")
        print("Enriched Selector Extractor")
        print(f"{'='*60}")
        print(f"Codebase: {self.codebase_path}")
        print(f"Output: {self.output_dir}")

    def run_all_phases(self) -> List[Dict[str, Any]]:
        """
        Run all extraction phases with checkpoints.
        """
        # Phase 1: Basic extraction
        html_parser = HTMLParser(str(self.codebase_path))
        selectors = html_parser.extract_static_selectors()
        self.save_checkpoint("phase1_basic", selectors)

        # Phase 2: Dynamic extraction (placeholder for now)
        dynamic_extractor = DynamicExtractor(str(self.codebase_path))
        selectors = dynamic_extractor.add_dynamic_values(selectors)
        self.save_checkpoint("phase2_dynamic", selectors)

        # Phase 3: Context enrichment (placeholder for now)
        context_enricher = ContextEnricher()
        selectors = context_enricher.enrich_context(selectors)
        self.save_checkpoint("phase3_enriched", selectors)

        # Phase 4: Priority calculation (placeholder for now)
        priority_calculator = PriorityCalculator()
        selectors = priority_calculator.calculate_priorities(selectors)
        self.save_checkpoint("phase4_final", selectors, final=True)

        return selectors

    def save_checkpoint(self, phase: str, selectors: List[Dict[str, Any]], final: bool = False):
        """
        Save checkpoint after each phase.
        """
        if final:
            filename = "selectors_enriched_all_modules.json"
        else:
            filename = f"{phase}_selectors.json"

        output_path = self.output_dir / filename

        # Create output with metadata
        output = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'codebase_path': str(self.codebase_path),
                'total_selectors': len(selectors),
                'static_selectors': sum(1 for s in selectors if not s.get('isDynamic', False)),
                'dynamic_selectors': sum(1 for s in selectors if s.get('isDynamic', False)),
                'phase': phase
            },
            'selectors': selectors
        }

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVE] Checkpoint saved: {output_path}")
        print(f"   Selectors: {len(selectors)}")
        print(f"   Static: {output['metadata']['static_selectors']}")
        print(f"   Dynamic: {output['metadata']['dynamic_selectors']}")

    def generate_statistics(self, selectors: List[Dict[str, Any]]):
        """
        Generate statistics report.
        """
        print(f"\n{'='*60}")
        print("EXTRACTION STATISTICS")
        print(f"{'='*60}")

        total = len(selectors)
        static = sum(1 for s in selectors if not s.get('isDynamic', False))
        dynamic = sum(1 for s in selectors if s.get('isDynamic', False))

        # Module distribution
        modules = {}
        for selector in selectors:
            module = selector.get('module', 'Unknown')
            modules[module] = modules.get(module, 0) + 1

        print(f"Total selectors: {total}")
        print(f"  Static: {static} ({static/total*100:.1f}%)")
        print(f"  Dynamic: {dynamic} ({dynamic/total*100:.1f}%)")
        print(f"\nModules found: {len(modules)}")
        print("Top 10 modules:")
        for module, count in sorted(modules.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {module}: {count}")

        # Element type distribution
        element_types = {}
        for selector in selectors:
            elem_type = selector.get('elementType', 'unknown')
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        print(f"\nElement types:")
        for elem_type, count in sorted(element_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem_type}: {count}")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """
    Main entry point.
    """
    # Configuration
    codebase_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client/src"
    output_dir = "Selectors_Folder"

    try:
        # Create extractor
        extractor = EnrichedSelectorExtractor(codebase_path, output_dir)

        # Run all phases
        selectors = extractor.run_all_phases()

        # Generate statistics
        extractor.generate_statistics(selectors)

        print(f"\n{'='*60}")
        print("[SUCCESS] EXTRACTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total selectors extracted: {len(selectors)}")
        print(f"Output location: {output_dir}")
        print(f"\nNext steps:")
        print("  1. Review output files in Selectors_Folder/")
        print("  2. Implement Phase 2 (Dynamic extraction)")
        print("  3. Implement Phase 3 (Context enrichment)")
        print("  4. Implement Phase 4 (Priority calculation)")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please verify the codebase path is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
