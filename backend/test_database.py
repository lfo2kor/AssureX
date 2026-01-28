"""
Test script for database functionality.
This script tests all database operations to ensure they work correctly.
"""
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from database.db import (
    create_tables,
    add_project,
    add_tester,
    authenticate_tester,
    get_project_by_name,
    get_project_by_id,
    get_testers_by_project,
    project_exists,
    tester_exists
)

def main():
    print("="*80)
    print("Testing Database Functionality")
    print("="*80)
    print()

    # Test 1: Create tables
    print("Test 1: Creating database tables...")
    try:
        create_tables()
        print("✓ Tables created successfully")
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return
    print()

    # Test 2: Add a project
    print("Test 2: Adding a project...")
    try:
        project_id = add_project("test_project", "Projects")
        print(f"✓ Project created with ID: {project_id}")
    except Exception as e:
        print(f"✗ Failed to add project: {e}")
        return
    print()

    # Test 3: Check if project exists
    print("Test 3: Checking if project exists...")
    try:
        exists = project_exists("test_project")
        print(f"✓ Project exists: {exists}")
    except Exception as e:
        print(f"✗ Failed to check project existence: {e}")
    print()

    # Test 4: Get project by name
    print("Test 4: Getting project by name...")
    try:
        project = get_project_by_name("test_project")
        print(f"✓ Project retrieved: {project}")
    except Exception as e:
        print(f"✗ Failed to get project: {e}")
    print()

    # Test 5: Get project by ID
    print("Test 5: Getting project by ID...")
    try:
        project = get_project_by_id(project_id)
        print(f"✓ Project retrieved: {project}")
    except Exception as e:
        print(f"✗ Failed to get project: {e}")
    print()

    # Test 6: Add a tester
    print("Test 6: Adding a tester...")
    try:
        tester_id = add_tester("testuser", "password123", project_id)
        print(f"✓ Tester created with ID: {tester_id}")
    except Exception as e:
        print(f"✗ Failed to add tester: {e}")
        return
    print()

    # Test 7: Check if tester exists
    print("Test 7: Checking if tester exists...")
    try:
        exists = tester_exists("testuser")
        print(f"✓ Tester exists: {exists}")
    except Exception as e:
        print(f"✗ Failed to check tester existence: {e}")
    print()

    # Test 8: Authenticate tester (valid credentials)
    print("Test 8: Authenticating tester (valid credentials)...")
    try:
        result = authenticate_tester("testuser", "password123")
        if result:
            print(f"✓ Authentication successful!")
            print(f"  Tester info: {result}")
        else:
            print("✗ Authentication failed")
    except Exception as e:
        print(f"✗ Authentication error: {e}")
    print()

    # Test 9: Authenticate tester (invalid credentials)
    print("Test 9: Authenticating tester (invalid credentials)...")
    try:
        result = authenticate_tester("testuser", "wrongpassword")
        if result:
            print("✗ Authentication should have failed but succeeded")
        else:
            print("✓ Authentication correctly rejected invalid credentials")
    except Exception as e:
        print(f"✗ Authentication error: {e}")
    print()

    # Test 10: Get testers by project
    print("Test 10: Getting testers by project...")
    try:
        testers = get_testers_by_project(project_id)
        print(f"✓ Found {len(testers)} tester(s):")
        for tester in testers:
            print(f"  - {tester}")
    except Exception as e:
        print(f"✗ Failed to get testers: {e}")
    print()

    # Test 11: Add another tester
    print("Test 11: Adding another tester...")
    try:
        tester_id_2 = add_tester("testuser2", "secure456", project_id)
        print(f"✓ Second tester created with ID: {tester_id_2}")
    except Exception as e:
        print(f"✗ Failed to add second tester: {e}")
    print()

    # Test 12: Get all testers for project again
    print("Test 12: Getting all testers by project...")
    try:
        testers = get_testers_by_project(project_id)
        print(f"✓ Found {len(testers)} tester(s):")
        for tester in testers:
            print(f"  - Username: {tester['username']}, ID: {tester['id']}")
    except Exception as e:
        print(f"✗ Failed to get testers: {e}")
    print()

    # Test 13: Try to add duplicate project (should fail)
    print("Test 13: Trying to add duplicate project (should fail)...")
    try:
        add_project("test_project", "Projects")
        print("✗ Should have raised an error for duplicate project")
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate project: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    print()

    # Test 14: Try to add duplicate username (should fail)
    print("Test 14: Trying to add duplicate username (should fail)...")
    try:
        add_tester("testuser", "newpassword", project_id)
        print("✗ Should have raised an error for duplicate username")
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate username: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    print()

    print("="*80)
    print("All tests completed!")
    print("="*80)

if __name__ == "__main__":
    main()
