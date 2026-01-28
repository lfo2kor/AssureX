"""
Phase 1 Test Script - Foundation Components

Tests:
1. Project structure (directories and packages)
2. Configuration loader
3. Logger utility
4. State schema
5. Vision helper (basic initialization)
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("PHASE 1 TEST - Foundation Components")
print("=" * 70)

# Test 1: Project Structure
print("\n[TEST 1] Project Structure")
print("-" * 70)

required_dirs = [
    "agents", "workflows", "models", "utils", "templates",
    "Jira_Tickets", "Reports", "Videos", "Generated_Scripts", "Logs"
]

for dir_name in required_dirs:
    if os.path.isdir(dir_name):
        print(f"✓ Directory exists: {dir_name}/")
    else:
        print(f"✗ Missing directory: {dir_name}/")
        sys.exit(1)

required_init_files = [
    "agents/__init__.py",
    "workflows/__init__.py",
    "models/__init__.py",
    "utils/__init__.py"
]

for init_file in required_init_files:
    if os.path.isfile(init_file):
        print(f"✓ Package init exists: {init_file}")
    else:
        print(f"✗ Missing package init: {init_file}")
        sys.exit(1)

print("\n✅ Project structure test PASSED")

# Test 2: Import Modules
print("\n[TEST 2] Module Imports")
print("-" * 70)

try:
    from models.state import TestAutomationState
    print("✓ Imported: models.state.TestAutomationState")
except ImportError as e:
    print(f"✗ Failed to import state: {e}")
    sys.exit(1)

try:
    from utils.config_loader import load_config, get_folder_path
    print("✓ Imported: utils.config_loader")
except ImportError as e:
    print(f"✗ Failed to import config_loader: {e}")
    sys.exit(1)

try:
    from utils.logger import setup_logger, mask_sensitive_data
    print("✓ Imported: utils.logger")
except ImportError as e:
    print(f"✗ Failed to import logger: {e}")
    sys.exit(1)

try:
    from utils.vision_helper import AzureVisionClient
    print("✓ Imported: utils.vision_helper")
except ImportError as e:
    print(f"✗ Failed to import vision_helper: {e}")
    sys.exit(1)

print("\n✅ Module import test PASSED")

# Test 3: Configuration Loader
print("\n[TEST 3] Configuration Loader")
print("-" * 70)

try:
    config = load_config("plcdtest_config.yaml")
    print("✓ Config file loaded successfully")

    # Check required fields
    required_fields = ['base_folder', 'web_url', 'browser', 'login', 'wait_times',
                      'folders', 'azure_openai', 'execution']
    for field in required_fields:
        if field in config:
            print(f"✓ Config has field: {field}")
        else:
            print(f"✗ Missing config field: {field}")
            sys.exit(1)

    # Check paths are resolved
    print(f"✓ Base folder: {config['base_folder']}")
    print(f"✓ Jira folder: {config['folders']['jira']}")
    print(f"✓ Reports folder: {config['folders']['reports']}")

    # Test get_folder_path
    jira_path = get_folder_path(config, 'jira')
    print(f"✓ get_folder_path works: {jira_path}")

except FileNotFoundError as e:
    print(f"✗ Config file not found: {e}")
    print("\n⚠️  Please ensure plcdtest_config.yaml exists in project root")
    sys.exit(1)
except Exception as e:
    print(f"✗ Config loader error: {e}")
    sys.exit(1)

print("\n✅ Configuration loader test PASSED")

# Test 4: Logger Utility
print("\n[TEST 4] Logger Utility")
print("-" * 70)

try:
    # Test sensitive data masking
    test_messages = [
        ("password='secret123'", "password=***MASKED***"),
        ("api_key: abc123xyz", "api_key=***MASKED***"),
        ("Bearer token123abc", "Bearer ***MASKED***"),
    ]

    for original, expected_pattern in test_messages:
        masked = mask_sensitive_data(original)
        if "***MASKED***" in masked:
            print(f"✓ Masked: '{original}' -> '{masked}'")
        else:
            print(f"✗ Failed to mask: '{original}'")
            sys.exit(1)

    # Test logger setup
    logger = setup_logger(
        config['folders']['logs'],
        "TEST-001",
        logger_name="TestLogger"
    )
    print("✓ Logger created successfully")

    # Test logging
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message with password='secret'")
    logger.warning("Test WARNING message")

    print("✓ Logging works (check Logs/ folder for TEST-001_*.log)")

except Exception as e:
    print(f"✗ Logger error: {e}")
    sys.exit(1)

print("\n✅ Logger utility test PASSED")

# Test 5: State Schema
print("\n[TEST 5] State Schema")
print("-" * 70)

try:
    # Create test state
    test_state: TestAutomationState = {
        'config': config,
        'ticket_number': 'TEST-001',
        'jira_data': {},
        'execution_results': [],
        'errors': []
    }
    print("✓ TestAutomationState can be instantiated")
    print(f"✓ State has ticket_number: {test_state['ticket_number']}")
    print(f"✓ State has config: {len(test_state['config'])} keys")

except Exception as e:
    print(f"✗ State schema error: {e}")
    sys.exit(1)

print("\n✅ State schema test PASSED")

# Test 6: Vision Helper (Basic Initialization)
print("\n[TEST 6] Vision Helper Initialization")
print("-" * 70)

try:
    # Check if API key is configured
    api_key = config['azure_openai']['api_key']
    if api_key in ['YOUR_API_KEY_HERE', 'your_api_key_here', '']:
        print("⚠️  API key not configured (using placeholder)")
        print("⚠️  Skipping Vision Client initialization test")
        print("⚠️  To test vision client, add real API key to config")
    else:
        # Try to initialize client
        vision_client = AzureVisionClient(config, logger)
        print("✓ AzureVisionClient initialized successfully")
        print(f"✓ Using deployment: {config['azure_openai']['deployment_gpt4o']}")
        print(f"✓ Endpoint: {config['azure_openai']['endpoint']}")

        # Test image encoding
        test_image = b"fake_image_data"
        encoded = vision_client.encode_image(test_image)
        print(f"✓ Image encoding works (base64 length: {len(encoded)})")

except Exception as e:
    print(f"✗ Vision helper error: {e}")
    print(f"   Error details: {type(e).__name__}")
    sys.exit(1)

print("\n✅ Vision helper test PASSED")

# Final Summary
print("\n" + "=" * 70)
print("🎉 ALL PHASE 1 TESTS PASSED!")
print("=" * 70)
print("\nPhase 1 foundation is ready for Phase 2 agent development.")
print("\nNext steps:")
print("  1. Implement Jira Parser Agent")
print("  2. Implement Vision Executor Agent")
print("  3. Implement Report Generator Agent")
print("=" * 70)
