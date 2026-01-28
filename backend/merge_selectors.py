"""
Merge Runtime Selectors with Source Code Selectors

Combines runtime-extracted selectors with existing source code selectors.
Runtime selectors take priority (more accurate).

Usage:
    python merge_selectors.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    """Save JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {file_path}")


def merge_selectors(source_file: str, runtime_files: List[str], output_file: str):
    """
    Merge source code selectors with runtime selectors.

    Priority order:
    1. Runtime selectors (highest priority - most accurate)
    2. Source code selectors

    Args:
        source_file: Path to source code selectors JSON
        runtime_files: List of paths to runtime selector JSON files
        output_file: Path to output merged JSON
    """
    logger.info("=" * 80)
    logger.info("   Merging Selectors")
    logger.info("=" * 80)

    # Load source code selectors
    logger.info(f"\n[1/4] Loading source code selectors...")
    source_data = load_json(source_file)

    if 'selectors' in source_data:
        # New enriched format
        source_selectors = source_data['selectors']
        source_metadata = source_data.get('metadata', {})
    else:
        # Old format
        source_selectors = source_data
        source_metadata = {}

    logger.info(f"  Loaded {len(source_selectors)} selectors from source code")

    # Load all runtime selectors
    logger.info(f"\n[2/4] Loading runtime selectors...")
    all_runtime_selectors = []

    for runtime_file in runtime_files:
        if not Path(runtime_file).exists():
            logger.warning(f"  Runtime file not found: {runtime_file}")
            continue

        runtime_data = load_json(runtime_file)
        runtime_selectors = runtime_data.get('selectors', [])
        all_runtime_selectors.extend(runtime_selectors)
        logger.info(f"  Loaded {len(runtime_selectors)} selectors from {Path(runtime_file).name}")

    logger.info(f"  Total runtime selectors: {len(all_runtime_selectors)}")

    # Merge selectors
    logger.info(f"\n[3/4] Merging selectors...")
    merged = {}

    # Add source code selectors first (lower priority)
    for sel in source_selectors:
        key = f"{sel.get('attr', '')}={sel.get('value', '')}"
        merged[key] = sel.copy()
        merged[key]['source'] = 'static_html'
        merged[key]['runtimeVerified'] = False

    logger.info(f"  Added {len(merged)} source code selectors")

    # Add/override with runtime selectors (higher priority)
    runtime_count = 0
    override_count = 0

    for sel in all_runtime_selectors:
        key = f"{sel.get('attr', '')}={sel.get('value', '')}"

        if key in merged:
            # Runtime version overrides source code version
            # Keep some source code metadata, update with runtime data
            merged[key].update({
                'runtimeVerified': True,
                'isClickable': sel.get('isClickable', False),
                'isVisible': sel.get('isVisible', False),
                'tagName': sel.get('tagName', merged[key].get('tagName', '')),
                'textContent': sel.get('textContent', ''),
                'role': sel.get('role'),
                'ariaLabel': sel.get('ariaLabel'),
                'source': 'runtime_override',
                'runtimeExtractedDate': sel.get('extractedDate', ''),
                'learned_from': sel.get('learned_from', ''),
                'step_num': sel.get('step_num'),
                'step_text': sel.get('step_text', ''),
                'priority': sel.get('priority', merged[key].get('priority', 50))
            })
            override_count += 1
        else:
            # New selector only found at runtime
            sel['source'] = 'runtime_only'
            sel['runtimeVerified'] = True
            merged[key] = sel
            runtime_count += 1

    logger.info(f"  Runtime selectors: {runtime_count} new, {override_count} overrides")

    # Filter selectors (remove unverified or low-priority ones if needed)
    logger.info(f"\n[4/4] Filtering and saving...")

    # Keep all selectors for now (can add filtering logic later)
    final_selectors = list(merged.values())

    # Sort by module, then priority
    final_selectors.sort(key=lambda x: (x.get('module', ''), -x.get('priority', 50)))

    # Create output data
    output_data = {
        'metadata': {
            'phase': 'runtime_merged',
            'mergeDate': datetime.now().isoformat(),
            'sourceFile': source_file,
            'runtimeFiles': runtime_files,
            'totalSelectors': len(final_selectors),
            'sourceSelectorCount': len(source_selectors),
            'runtimeSelectorCount': len(all_runtime_selectors),
            'runtimeVerifiedCount': sum(1 for s in final_selectors if s.get('runtimeVerified')),
            'sourceMetadata': source_metadata
        },
        'selectors': final_selectors
    }

    # Save merged selectors
    save_json(output_data, output_file)

    # Print summary
    logger.info(f"\n" + "=" * 80)
    logger.info(f"   Merge Complete!")
    logger.info(f"=" * 80)
    logger.info(f"Total selectors: {len(final_selectors)}")
    logger.info(f"  - From source code: {len(source_selectors)}")
    logger.info(f"  - Runtime verified: {output_data['metadata']['runtimeVerifiedCount']}")
    logger.info(f"  - Runtime only: {runtime_count}")
    logger.info(f"  - Runtime overrides: {override_count}")
    logger.info(f"\nOutput: {output_file}")
    logger.info(f"\nNext step: python run_test.py RBPLCD-8835")


if __name__ == "__main__":
    # Configuration
    source_file = "Selectors_Folder/selectors_enriched_all_modules.json"
    runtime_folder = Path("Selectors_Folder")

    # Find all runtime selector files
    runtime_files = list(runtime_folder.glob("runtime_selectors_*.json"))

    if not runtime_files:
        logger.error("No runtime selector files found!")
        logger.info("Please run: python extract_runtime_selectors.py TICKET_ID")
        exit(1)

    logger.info(f"Found {len(runtime_files)} runtime selector files:")
    for f in runtime_files:
        logger.info(f"  - {f.name}")

    output_file = "Selectors_Folder/selectors_merged_runtime.json"

    # Merge
    merge_selectors(
        source_file=source_file,
        runtime_files=[str(f) for f in runtime_files],
        output_file=output_file
    )

    # Update selector loader to use merged file
    logger.info(f"\n" + "=" * 80)
    logger.info("To use merged selectors, update agents/vision_executor_agent.py:")
    logger.info("  Change:")
    logger.info("    selectors_file='Selectors_Folder/selectors_enriched_all_modules.json'")
    logger.info("  To:")
    logger.info(f"    selectors_file='{output_file}'")
    logger.info("=" * 80)
