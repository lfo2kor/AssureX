"""
Cleanup script to remove old test artifacts
Removes: HTML reports, videos, screenshots, and generated scripts
"""
import os
import glob

def cleanup_files():
    """Remove all old test artifacts"""

    print("="*70)
    print("Cleaning up old test files...")
    print("="*70)
    print()

    # Define folders and file patterns to clean
    cleanup_targets = [
        ("Reports/*.html", "HTML reports"),
        ("Videos/*.webm", "Video recordings"),
        ("Screenshots/*.png", "Screenshots"),
        ("Generated_Scripts/*.py", "Generated Playwright scripts"),
    ]

    total_deleted = 0

    for pattern, description in cleanup_targets:
        files = glob.glob(pattern)
        count = len(files)

        if count > 0:
            print(f"Deleting {count} {description}...")
            for file in files:
                try:
                    os.remove(file)
                    print(f"  Deleted: {file}")
                except Exception as e:
                    print(f"  ERROR deleting {file}: {e}")
            total_deleted += count
        else:
            print(f"No {description} found")

    print()
    print("="*70)
    print(f"Cleanup complete! Removed {total_deleted} files")
    print("="*70)

if __name__ == "__main__":
    cleanup_files()
