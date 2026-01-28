"""
Batch Selector Extractor - Extracts enriched selectors from ALL modules

This script processes all modules in the web application and extracts selectors
with context, priority, and usage_scenario fields.

NO JIRA DEPENDENCIES - All context extracted from HTML/TypeScript source code only.

Usage:
    python extract_all_modules.py

Output:
    - Selectors_Folder/selectors_enriched_all_modules.json (complete enriched selectors)
    - Selectors_Folder/extraction_stats.txt (extraction statistics)
"""

import json
import sys
from pathlib import Path
from extract_selectors_with_context import SelectorContextExtractor

def main():
    print("=" * 80)
    print("BATCH SELECTOR EXTRACTION - ALL MODULES")
    print("=" * 80)
    print()

    # Base path to web application
    base_path = "C:/Projects/AI_Chat/PLCD/cri-webapp/client"
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"ERROR: Base path does not exist: {base_path}")
        sys.exit(1)

    # Initialize extractor
    print(f"Base path: {base_path}")
    extractor = SelectorContextExtractor(base_path)

    # Get all app modules
    app_folder = base_path_obj / "src" / "app"
    if not app_folder.exists():
        print(f"ERROR: App folder not found: {app_folder}")
        sys.exit(1)

    # Find all module directories
    module_folders = [d for d in app_folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
    module_folders.sort()

    print(f"Found {len(module_folders)} modules to process")
    print()

    # Track statistics
    stats = {
        'total_modules': len(module_folders),
        'processed_modules': 0,
        'failed_modules': 0,
        'total_selectors': 0,
        'module_details': []
    }

    all_selectors = []

    # Process each module
    for i, module_folder in enumerate(module_folders, 1):
        module_name = module_folder.name
        print(f"[{i}/{len(module_folders)}] Processing module: {module_name}")

        try:
            # Extract selectors from this module
            module_path = module_folder.relative_to(base_path_obj)
            selectors = extractor.extract_from_folder(str(module_path))

            # Update statistics
            stats['processed_modules'] += 1
            stats['total_selectors'] += len(selectors)
            stats['module_details'].append({
                'module': module_name,
                'selector_count': len(selectors),
                'status': 'success'
            })

            all_selectors.extend(selectors)
            print(f"  -> Extracted {len(selectors)} selectors")

        except Exception as e:
            print(f"  -> ERROR: {str(e)}")
            stats['failed_modules'] += 1
            stats['module_details'].append({
                'module': module_name,
                'selector_count': 0,
                'status': 'failed',
                'error': str(e)
            })

        print()

    # Save combined enriched selectors
    output_file = Path("Selectors_Folder/selectors_enriched_all_modules.json")
    output_file.parent.mkdir(exist_ok=True)

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_selectors, f, indent=2, ensure_ascii=False)

    print(f"Saved enriched selectors to: {output_file}")
    print(f"Total selectors: {len(all_selectors)}")
    print()

    # Save statistics
    stats_file = Path("Selectors_Folder/extraction_stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH EXTRACTION STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total modules found: {stats['total_modules']}\n")
        f.write(f"Successfully processed: {stats['processed_modules']}\n")
        f.write(f"Failed: {stats['failed_modules']}\n")
        f.write(f"Total selectors extracted: {stats['total_selectors']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("MODULE DETAILS\n")
        f.write("=" * 80 + "\n\n")

        for detail in stats['module_details']:
            f.write(f"Module: {detail['module']}\n")
            f.write(f"  Status: {detail['status']}\n")
            f.write(f"  Selectors: {detail['selector_count']}\n")
            if detail['status'] == 'failed':
                f.write(f"  Error: {detail.get('error', 'Unknown error')}\n")
            f.write("\n")

        # Priority distribution
        f.write("=" * 80 + "\n")
        f.write("PRIORITY DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")

        priority_counts = {}
        for selector in all_selectors:
            priority = selector.get('priority', 5)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        for priority in sorted(priority_counts.keys(), reverse=True):
            count = priority_counts[priority]
            percentage = (count / len(all_selectors) * 100) if all_selectors else 0
            f.write(f"Priority {priority}: {count} selectors ({percentage:.1f}%)\n")

        f.write("\n")

        # Context keyword frequency
        f.write("=" * 80 + "\n")
        f.write("TOP 20 CONTEXT KEYWORDS\n")
        f.write("=" * 80 + "\n\n")

        context_counts = {}
        for selector in all_selectors:
            for keyword in selector.get('context', []):
                context_counts[keyword] = context_counts.get(keyword, 0) + 1

        sorted_keywords = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for keyword, count in sorted_keywords:
            percentage = (count / len(all_selectors) * 100) if all_selectors else 0
            f.write(f"{keyword}: {count} ({percentage:.1f}%)\n")

    print(f"Saved statistics to: {stats_file}")
    print()

    # Print summary
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Modules processed: {stats['processed_modules']}/{stats['total_modules']}")
    print(f"Total selectors: {stats['total_selectors']}")
    print(f"Failed modules: {stats['failed_modules']}")
    print()

    if stats['failed_modules'] > 0:
        print("WARNING: Some modules failed to process. Check extraction_stats.txt for details.")
        print()

    # Top 5 modules by selector count
    top_modules = sorted(stats['module_details'], key=lambda x: x['selector_count'], reverse=True)[:5]
    print("Top 5 modules by selector count:")
    for detail in top_modules:
        print(f"  {detail['module']}: {detail['selector_count']} selectors")
    print()

    # Compare with current selectors.json
    current_selectors_file = Path("Selectors_Folder/selectors.json")
    if current_selectors_file.exists():
        with open(current_selectors_file, 'r', encoding='utf-8') as f:
            current_selectors = json.load(f)

        print("=" * 80)
        print("COMPARISON WITH CURRENT selectors.json")
        print("=" * 80)
        print(f"Current selectors.json: {len(current_selectors)} selectors")
        print(f"New enriched version: {len(all_selectors)} selectors")

        diff = len(all_selectors) - len(current_selectors)
        if diff > 0:
            print(f"Difference: +{diff} selectors (enriched has more)")
        elif diff < 0:
            print(f"Difference: {diff} selectors (enriched has fewer)")
        else:
            print("Difference: Same count")
        print()

        # Check for new fields
        if all_selectors:
            sample_enriched = all_selectors[0]
            sample_current = current_selectors[0] if current_selectors else {}

            new_fields = set(sample_enriched.keys()) - set(sample_current.keys())
            if new_fields:
                print(f"New fields added: {', '.join(sorted(new_fields))}")
                print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review extraction_stats.txt for detailed statistics")
    print("2. Inspect selectors_enriched_all_modules.json")
    print("3. Compare with current selectors.json")
    print("4. If satisfied, backup current selectors.json:")
    print("   copy Selectors_Folder\\selectors.json Selectors_Folder\\selectors_v1.0_backup.json")
    print("5. Replace with enriched version:")
    print("   copy Selectors_Folder\\selectors_enriched_all_modules.json Selectors_Folder\\selectors.json")
    print("6. Test with RBPLCD-8862 to verify improvement")
    print()
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
