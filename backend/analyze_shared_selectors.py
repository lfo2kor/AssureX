"""
Analyze Shared Selectors - Show how selectors appear across multiple modules

This script analyzes the enriched selectors JSON to understand:
1. Which selectors appear in multiple modules
2. How context differs across modules
3. Whether we should merge or keep separate
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_shared_selectors():
    # Load enriched selectors
    enriched_file = Path("Selectors_Folder/selectors_enriched_all_modules.json")
    with open(enriched_file, 'r', encoding='utf-8') as f:
        selectors = json.load(f)

    print("=" * 100)
    print("SHARED SELECTOR ANALYSIS")
    print("=" * 100)
    print()

    # Group by attr+value
    grouped = defaultdict(list)
    for selector in selectors:
        key = f"{selector['attr']}={selector['value']}"
        grouped[key].append(selector)

    # Find duplicates
    shared_selectors = {k: v for k, v in grouped.items() if len(v) > 1}
    unique_selectors = {k: v for k, v in grouped.items() if len(v) == 1}

    print(f"Total selectors in enriched JSON: {len(selectors)}")
    print(f"Unique attr+value combinations: {len(grouped)}")
    print(f"Shared selectors (multiple occurrences): {len(shared_selectors)}")
    print(f"Unique selectors (single occurrence): {len(unique_selectors)}")
    print()

    # Calculate duplication
    total_duplicate_entries = sum(len(v) - 1 for v in shared_selectors.values())
    print(f"Total duplicate entries: {total_duplicate_entries}")
    print(f"If we merge duplicates, JSON would have: {len(grouped)} entries (vs current {len(selectors)})")
    print()

    # Analyze shared selectors
    print("=" * 100)
    print("DETAILED ANALYSIS OF SHARED SELECTORS")
    print("=" * 100)
    print()

    for i, (key, occurrences) in enumerate(list(shared_selectors.items())[:10], 1):
        print(f"\n{i}. Selector: {key}")
        print(f"   Appears in {len(occurrences)} locations")
        print()

        for j, occ in enumerate(occurrences, 1):
            print(f"   Occurrence {j}:")
            print(f"     Module: {occ['module']}")
            print(f"     File: {occ['filePath']}")
            print(f"     Line: {occ['lineNumber']}")
            print(f"     Element: {occ['elementType']}")
            print(f"     Context: {occ['context']}")
            print(f"     Priority: {occ['priority']}")
            print()

        # Check if context is identical
        contexts = [set(occ['context']) for occ in occurrences]
        if all(ctx == contexts[0] for ctx in contexts):
            print(f"   [OK] IDENTICAL CONTEXT - Safe to merge")
        else:
            print(f"   [WARNING] DIFFERENT CONTEXT - Module-specific behavior detected")
            print(f"      Context differences:")
            for j, occ in enumerate(occurrences, 1):
                unique_to_this = set(occ['context']) - set.union(*[set(other['context']) for k, other in enumerate(occurrences) if k != j-1])
                if unique_to_this:
                    print(f"        {occ['module']}: {unique_to_this}")

        print("-" * 100)

    # Summary statistics
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    identical_context_count = 0
    different_context_count = 0

    for key, occurrences in shared_selectors.items():
        contexts = [set(occ['context']) for occ in occurrences]
        if all(ctx == contexts[0] for ctx in contexts):
            identical_context_count += 1
        else:
            different_context_count += 1

    print(f"Shared selectors with IDENTICAL context: {identical_context_count}")
    print(f"  -> Safe to merge (no information loss)")
    print()
    print(f"Shared selectors with DIFFERENT context: {different_context_count}")
    print(f"  -> Merging would lose module-specific context")
    print()

    # Recommendation
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print()

    if different_context_count > identical_context_count:
        print("[RECOMMENDED] KEEP SEPARATE ENTRIES (Current approach)")
        print()
        print("Reason:")
        print(f"  - {different_context_count} selectors have module-specific context")
        print("  - Merging would lose important behavioral differences")
        print("  - Better matching accuracy with module-specific context")
        print()
        print("Implementation:")
        print("  1. Keep current extraction (separate entries)")
        print("  2. Update selector_loader.py to handle module hierarchy:")
        print("     - If test says 'Teststep module', search in:")
        print("       a) teststeps module")
        print("       b) Modules used BY teststeps (parts, entity-attribute)")
        print("  3. Add 'shared_selector' flag for cross-module fallback")
    else:
        print("[RECOMMENDED] MERGE DUPLICATES")
        print()
        print("Reason:")
        print(f"  - {identical_context_count} selectors have identical context")
        print("  - Merging reduces duplication without losing information")
        print("  - Simpler matching logic")
        print()
        print("Implementation:")
        print("  1. Create merge_duplicates.py script")
        print("  2. Combine entries with same attr+value")
        print("  3. Store all module locations in 'occurrences' array")
        print("  4. Try each location during matching")

    # Save detailed report
    report_file = Path("Selectors_Folder/shared_selectors_report.json")
    report = {
        "summary": {
            "total_selectors": len(selectors),
            "unique_combinations": len(grouped),
            "shared_selectors": len(shared_selectors),
            "unique_selectors": len(unique_selectors),
            "identical_context": identical_context_count,
            "different_context": different_context_count
        },
        "shared_selectors": {}
    }

    for key, occurrences in shared_selectors.items():
        contexts = [set(occ['context']) for occ in occurrences]
        identical = all(ctx == contexts[0] for ctx in contexts)

        report["shared_selectors"][key] = {
            "count": len(occurrences),
            "identical_context": identical,
            "occurrences": [
                {
                    "module": occ['module'],
                    "filePath": occ['filePath'],
                    "lineNumber": occ['lineNumber'],
                    "context": occ['context'],
                    "priority": occ['priority']
                }
                for occ in occurrences
            ]
        }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Detailed report saved to: {report_file}")
    print()


if __name__ == "__main__":
    analyze_shared_selectors()
