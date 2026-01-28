"""
Test script to verify all imports for Streamlit app work correctly.
"""
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Testing imports for Streamlit application...")
print("=" * 80)

# Test 1: Streamlit
print("\n1. Testing streamlit import...")
try:
    import streamlit as st
    print(f"   ✓ Streamlit version: {st.__version__}")
except ImportError as e:
    print(f"   ✗ Failed to import streamlit: {e}")
    sys.exit(1)

# Test 2: YAML
print("\n2. Testing yaml import...")
try:
    import yaml
    print(f"   ✓ PyYAML imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import yaml: {e}")
    sys.exit(1)

# Test 3: Project Manager Agent
print("\n3. Testing project_manager_agent import...")
try:
    from agents.project_manager_agent import project_manager_agent
    print(f"   ✓ Project Manager Agent imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import project_manager_agent: {e}")
    sys.exit(1)

# Test 4: Web module
print("\n4. Testing web module import...")
try:
    from web.admin_upload import show_admin_upload_page
    print(f"   ✓ Admin upload page imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import admin_upload: {e}")
    sys.exit(1)

# Test 5: Database
print("\n5. Testing database import...")
try:
    from database.db import create_tables, add_project, add_tester
    print(f"   ✓ Database functions imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import database: {e}")
    sys.exit(1)

# Test 6: Check sample config exists
print("\n6. Checking sample config file...")
import os
if os.path.exists("sample_config.yaml"):
    print(f"   ✓ sample_config.yaml found")
else:
    print(f"   ✗ sample_config.yaml not found")

print("\n" + "=" * 80)
print("✓ All imports successful! Streamlit app is ready to run.")
print("\nTo start the application, run:")
print("  streamlit run web_app.py")
print("=" * 80)
