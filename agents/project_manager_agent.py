"""
Project Manager Agent for validating and setting up new projects.

This agent validates config.yaml structure, checks system requirements,
creates project folder structure, and registers projects in the database.
"""

import os
import json
import logging
import shutil
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from database.db import add_project, add_tester, project_exists

# Setup logging
logger = logging.getLogger(__name__)


def project_manager_agent(state: dict) -> dict:
    """
    Main agent function that validates config and sets up new project.

    This function orchestrates the entire project setup process including
    validation, folder creation, and database registration.

    Args:
        state: Dictionary containing:
            - uploaded_config: Dict (parsed config.yaml content)
            - app_root: str (application root path)

    Returns:
        dict: Updated state with validation results and project info

    Example:
        >>> state = {
        ...     'uploaded_config': {...},
        ...     'app_root': 'C:/Projects/AI_Chat/PLCD/TA_AI_Project'
        ... }
        >>> result = project_manager_agent(state)
        >>> print(result['validation_status'])
        'success'
    """
    logger.info("Project Manager Agent started")

    # Initialize response structure
    validation_steps = []
    validation_errors = []
    validation_warnings = []
    project_created = False
    project_id = None
    testers_created = 0
    project_path = None
    created_folders = []
    created_files = []

    try:
        # Extract inputs
        uploaded_config = state.get('uploaded_config')
        app_root = state.get('app_root')

        if not uploaded_config:
            raise ValueError("Missing 'uploaded_config' in state")
        if not app_root:
            raise ValueError("Missing 'app_root' in state")

        # Step 1: Validate config structure
        logger.info("Step 1: Validating config structure")
        config_valid, config_errors = validate_config_structure(uploaded_config)
        if config_valid:
            validation_steps.append({
                "step": "validate_structure",
                "status": "passed",
                "message": "All required fields present and valid"
            })
        else:
            validation_steps.append({
                "step": "validate_structure",
                "status": "failed",
                "message": f"Config validation failed: {', '.join(config_errors)}"
            })
            validation_errors.extend(config_errors)
            raise ValueError("Config structure validation failed")

        # Step 2: Validate base folder path
        logger.info("Step 2: Validating base folder path")
        base_folder = uploaded_config.get('base_folder')
        project_name = uploaded_config.get('project', {}).get('name')

        base_path_valid, base_path_msg, full_base_path = validate_base_folder(
            app_root, base_folder
        )

        if base_path_valid:
            validation_steps.append({
                "step": "check_base_folder",
                "status": "passed",
                "message": base_path_msg
            })
        else:
            validation_steps.append({
                "step": "check_base_folder",
                "status": "failed",
                "message": base_path_msg
            })
            validation_errors.append(base_path_msg)
            raise ValueError("Base folder validation failed")

        # Step 3: Check project name uniqueness
        logger.info("Step 3: Checking project name uniqueness")
        if project_exists(project_name):
            validation_steps.append({
                "step": "check_project_uniqueness",
                "status": "failed",
                "message": f"Project '{project_name}' already exists in database"
            })
            validation_errors.append(f"Project '{project_name}' already exists")
            raise ValueError("Project name already exists")
        else:
            validation_steps.append({
                "step": "check_project_uniqueness",
                "status": "passed",
                "message": f"Project name '{project_name}' is unique"
            })

        # Step 4: Validate web application access
        logger.info("Step 4: Validating web application access")
        web_url = uploaded_config.get('web_application', {}).get('url')
        url_accessible, url_msg = validate_web_url(web_url)

        if url_accessible:
            validation_steps.append({
                "step": "check_web_url",
                "status": "passed",
                "message": url_msg
            })
        else:
            validation_steps.append({
                "step": "check_web_url",
                "status": "warning",
                "message": url_msg
            })
            validation_warnings.append(url_msg)

        # Step 5: Validate login credentials (TODO - placeholder for now)
        logger.info("Step 5: Validating login credentials (skipped)")
        validation_steps.append({
            "step": "validate_login",
            "status": "skipped",
            "message": "Login validation not yet implemented (TODO)"
        })
        # TODO: Implement Playwright-based login validation
        # This would involve launching headless browser, navigating to URL,
        # and attempting login with test credentials

        # Step 6: Create project folder structure
        logger.info("Step 6: Creating project folder structure")
        project_path = str(Path(full_base_path) / project_name)

        folders_created, folder_msg, created_folders = create_project_folders(
            project_path
        )

        if folders_created:
            validation_steps.append({
                "step": "create_folders",
                "status": "passed",
                "message": folder_msg
            })
        else:
            validation_steps.append({
                "step": "create_folders",
                "status": "failed",
                "message": folder_msg
            })
            validation_errors.append(folder_msg)
            raise ValueError("Failed to create project folders")

        # Create initial files
        logger.info("Creating initial project files")
        files_created, files_msg, created_files = create_initial_files(project_path)

        if files_created:
            validation_steps.append({
                "step": "create_initial_files",
                "status": "passed",
                "message": files_msg
            })
        else:
            validation_steps.append({
                "step": "create_initial_files",
                "status": "failed",
                "message": files_msg
            })
            validation_errors.append(files_msg)
            raise ValueError("Failed to create initial files")

        # Step 7: Save config file
        logger.info("Step 7: Saving config.yaml")
        config_saved, config_msg = save_config_file(project_path, uploaded_config)

        if config_saved:
            validation_steps.append({
                "step": "save_config",
                "status": "passed",
                "message": config_msg
            })
            created_files.append(str(Path(project_path) / "config.yaml"))
        else:
            validation_steps.append({
                "step": "save_config",
                "status": "failed",
                "message": config_msg
            })
            validation_errors.append(config_msg)
            raise ValueError("Failed to save config file")

        # Step 8: Create database records
        logger.info("Step 8: Creating database records")
        db_created, db_msg, project_id, testers_created = create_database_records(
            project_name, base_folder, uploaded_config.get('testers', [])
        )

        if db_created:
            project_created = True
            validation_steps.append({
                "step": "create_database_records",
                "status": "passed",
                "message": db_msg
            })
        else:
            validation_steps.append({
                "step": "create_database_records",
                "status": "failed",
                "message": db_msg
            })
            validation_errors.append(db_msg)
            raise ValueError("Failed to create database records")

        # Success!
        logger.info("Project setup completed successfully")
        state.update({
            "validation_status": "success",
            "validation_steps": validation_steps,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "project_created": project_created,
            "project_id": project_id,
            "testers_created": testers_created,
            "project_path": project_path,
            "project_name": project_name
        })

    except Exception as e:
        logger.error(f"Project setup failed: {e}", exc_info=True)

        # Rollback changes
        logger.info("Rolling back changes due to error")
        rollback_changes(created_folders, created_files, project_id)

        validation_steps.append({
            "step": "rollback",
            "status": "executed",
            "message": "Rolled back all changes due to error"
        })

        state.update({
            "validation_status": "failed",
            "validation_steps": validation_steps,
            "validation_errors": validation_errors if validation_errors else [str(e)],
            "validation_warnings": validation_warnings,
            "project_created": False,
            "project_id": None,
            "testers_created": 0,
            "project_path": None
        })

    return state


def validate_config_structure(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that config.yaml has all required fields with correct types.

    Args:
        config: Parsed config dictionary

    Returns:
        Tuple of (is_valid: bool, errors: List[str])

    Example:
        >>> config = {'base_folder': 'Projects', 'project': {'name': 'test'}}
        >>> is_valid, errors = validate_config_structure(config)
    """
    errors = []

    # Required top-level fields
    required_fields = [
        'base_folder',
        'project',
        'testers',
        'web_application',
        'wait_times',
        'folders',
        'azure_openai',
        'module_mapping',
        'execution'
    ]

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: '{field}'")

    # Validate base_folder
    if 'base_folder' in config:
        if not isinstance(config['base_folder'], str) or not config['base_folder'].strip():
            errors.append("'base_folder' must be a non-empty string")

    # Validate project.name
    if 'project' in config:
        if not isinstance(config['project'], dict):
            errors.append("'project' must be a dictionary")
        elif 'name' not in config['project']:
            errors.append("Missing required field: 'project.name'")
        elif not isinstance(config['project']['name'], str) or not config['project']['name'].strip():
            errors.append("'project.name' must be a non-empty string")

    # Validate testers
    if 'testers' in config:
        if not isinstance(config['testers'], list):
            errors.append("'testers' must be a list")
        elif len(config['testers']) == 0:
            errors.append("'testers' must contain at least one tester")
        else:
            for i, tester in enumerate(config['testers']):
                if not isinstance(tester, dict):
                    errors.append(f"Tester {i} must be a dictionary")
                    continue
                # Check username
                if 'username' not in tester:
                    errors.append(f"Tester {i} missing 'username'")
                elif not isinstance(tester['username'], str) or len(tester['username'].strip()) < 3:
                    errors.append(f"Tester {i} 'username' must be at least 3 characters")
                # Check password
                if 'password' not in tester:
                    errors.append(f"Tester {i} missing 'password'")
                elif not isinstance(tester['password'], str) or len(tester['password']) < 6:
                    errors.append(f"Tester {i} 'password' must be at least 6 characters")

    # Validate web_application
    if 'web_application' in config:
        if not isinstance(config['web_application'], dict):
            errors.append("'web_application' must be a dictionary")
        else:
            # Required fields
            if 'url' not in config['web_application']:
                errors.append("Missing required field: 'web_application.url'")
            elif not isinstance(config['web_application']['url'], str):
                errors.append("'web_application.url' must be a string")

            if 'browser' not in config['web_application']:
                errors.append("Missing required field: 'web_application.browser'")
            elif not isinstance(config['web_application']['browser'], str):
                errors.append("'web_application.browser' must be a string")

            if 'environment' not in config['web_application']:
                errors.append("Missing required field: 'web_application.environment'")
            elif not isinstance(config['web_application']['environment'], str):
                errors.append("'web_application.environment' must be a string")

            # Validate test_credentials
            if 'test_credentials' not in config['web_application']:
                errors.append("Missing required field: 'web_application.test_credentials'")
            elif not isinstance(config['web_application']['test_credentials'], dict):
                errors.append("'web_application.test_credentials' must be a dictionary")
            else:
                test_creds = config['web_application']['test_credentials']
                if 'username' not in test_creds:
                    errors.append("Missing required field: 'web_application.test_credentials.username'")
                if 'password' not in test_creds:
                    errors.append("Missing required field: 'web_application.test_credentials.password'")

    # Validate wait_times
    if 'wait_times' in config:
        if not isinstance(config['wait_times'], dict):
            errors.append("'wait_times' must be a dictionary")
        else:
            required_wait_times = [
                'after_login',
                'after_navigation',
                'after_click',
                'after_type',
                'after_dropdown',
                'page_load'
            ]
            for wait_field in required_wait_times:
                if wait_field not in config['wait_times']:
                    errors.append(f"Missing required field: 'wait_times.{wait_field}'")

    # Validate folders
    if 'folders' in config:
        if not isinstance(config['folders'], dict):
            errors.append("'folders' must be a dictionary")
        else:
            required_folders = ['jira', 'reports', 'videos', 'scripts', 'logs', 'selectors']
            for folder_field in required_folders:
                if folder_field not in config['folders']:
                    errors.append(f"Missing required field: 'folders.{folder_field}'")

    # Validate azure_openai
    if 'azure_openai' in config:
        if not isinstance(config['azure_openai'], dict):
            errors.append("'azure_openai' must be a dictionary")
        else:
            required_azure_fields = ['api_key', 'endpoint', 'api_version', 'deployment_gpt4o']
            for azure_field in required_azure_fields:
                if azure_field not in config['azure_openai']:
                    errors.append(f"Missing required field: 'azure_openai.{azure_field}'")

    # Validate module_mapping
    if 'module_mapping' in config:
        if not isinstance(config['module_mapping'], list):
            errors.append("'module_mapping' must be a list")
        else:
            for i, mapping in enumerate(config['module_mapping']):
                if not isinstance(mapping, dict):
                    errors.append(f"module_mapping[{i}] must be a dictionary")
                    continue
                if 'jira_name' not in mapping:
                    errors.append(f"module_mapping[{i}] missing 'jira_name'")
                if 'web_app_name' not in mapping:
                    errors.append(f"module_mapping[{i}] missing 'web_app_name'")

    # Validate execution
    if 'execution' in config:
        if not isinstance(config['execution'], dict):
            errors.append("'execution' must be a dictionary")
        else:
            required_execution_fields = [
                'max_retries',
                'screenshot_on_every_step',
                'record_video',
                'generate_script',
                'headless'
            ]
            for exec_field in required_execution_fields:
                if exec_field not in config['execution']:
                    errors.append(f"Missing required field: 'execution.{exec_field}'")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_base_folder(app_root: str, base_folder: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that base folder exists and is writable.

    Args:
        app_root: Application root path
        base_folder: Relative base folder path

    Returns:
        Tuple of (is_valid: bool, message: str, full_path: Optional[str])

    Example:
        >>> valid, msg, path = validate_base_folder('/app', 'Projects')
        >>> print(valid, msg)
        True Path exists and is writable
    """
    try:
        full_path = Path(app_root) / base_folder

        # Check if path exists
        if not full_path.exists():
            # Try to create it
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                message = f"Created base folder: {full_path}"
                logger.info(message)
            except Exception as e:
                return False, f"Cannot create base folder {full_path}: {e}", None

        # Check if writable
        if not os.access(full_path, os.W_OK):
            return False, f"Base folder {full_path} is not writable", None

        return True, f"Base folder validated: {full_path}", str(full_path)

    except Exception as e:
        return False, f"Base folder validation error: {e}", None


def validate_web_url(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Validate that web application URL is accessible.

    Args:
        url: Web application URL to check
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_accessible: bool, message: str)

    Example:
        >>> accessible, msg = validate_web_url('https://example.com')
    """
    if not url:
        return False, "Web application URL is empty"

    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return True, f"Web URL accessible: {url} (Status: 200)"
        else:
            return False, f"Web URL returned status {response.status_code}: {url}"

    except requests.Timeout:
        return False, f"Web URL timed out after {timeout}s: {url}"
    except requests.ConnectionError:
        return False, f"Cannot connect to web URL: {url}"
    except Exception as e:
        return False, f"Error checking web URL: {e}"


def create_project_folders(project_path: str) -> Tuple[bool, str, List[str]]:
    """
    Create project folder structure.

    Args:
        project_path: Full path to project root folder

    Returns:
        Tuple of (success: bool, message: str, created_folders: List[str])

    Example:
        >>> success, msg, folders = create_project_folders('/path/to/project')
    """
    created_folders = []

    try:
        # Define folder structure
        folders = [
            "",  # Root project folder
            "Jira_Tickets",
            "Selectors_Folder",
            "Reports",
            "Videos",
            "Generated_Scripts",
            "Logs"
        ]

        base_path = Path(project_path)

        for folder in folders:
            folder_path = base_path / folder if folder else base_path

            if folder_path.exists():
                logger.warning(f"Folder already exists: {folder_path}")
            else:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(str(folder_path))
                logger.info(f"Created folder: {folder_path}")

        message = f"Created {len(created_folders)} project folders"
        return True, message, created_folders

    except Exception as e:
        logger.error(f"Failed to create folders: {e}")
        return False, f"Failed to create project folders: {e}", created_folders


def create_initial_files(project_path: str) -> Tuple[bool, str, List[str]]:
    """
    Create initial empty files for the project.

    Args:
        project_path: Full path to project root folder

    Returns:
        Tuple of (success: bool, message: str, created_files: List[str])

    Example:
        >>> success, msg, files = create_initial_files('/path/to/project')
    """
    created_files = []

    try:
        base_path = Path(project_path)

        # Create selectors.json
        selectors_file = base_path / "Selectors_Folder" / "selectors.json"
        with open(selectors_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        created_files.append(str(selectors_file))
        logger.info(f"Created file: {selectors_file}")

        # Create feedback_log.json
        feedback_file = base_path / "feedback_log.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        created_files.append(str(feedback_file))
        logger.info(f"Created file: {feedback_file}")

        message = f"Created {len(created_files)} initial files"
        return True, message, created_files

    except Exception as e:
        logger.error(f"Failed to create initial files: {e}")
        return False, f"Failed to create initial files: {e}", created_files


def save_config_file(project_path: str, config: Dict) -> Tuple[bool, str]:
    """
    Save config.yaml to project folder.

    Args:
        project_path: Full path to project root folder
        config: Config dictionary to save

    Returns:
        Tuple of (success: bool, message: str)

    Example:
        >>> success, msg = save_config_file('/path/to/project', config_dict)
    """
    try:
        config_file = Path(project_path) / "config.yaml"

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved config file: {config_file}")
        return True, f"Config saved to: {config_file}"

    except Exception as e:
        logger.error(f"Failed to save config file: {e}")
        return False, f"Failed to save config file: {e}"


def create_database_records(
    project_name: str,
    base_folder: str,
    testers: List[Dict]
) -> Tuple[bool, str, Optional[int], int]:
    """
    Create project and tester records in database.

    Args:
        project_name: Name of the project
        base_folder: Base folder path
        testers: List of tester dictionaries with username and password

    Returns:
        Tuple of (success: bool, message: str, project_id: Optional[int], testers_created: int)

    Example:
        >>> success, msg, proj_id, tester_count = create_database_records(
        ...     'test_proj', 'Projects', [{'username': 'user1', 'password': 'pass1'}]
        ... )
    """
    project_id = None
    testers_created = 0

    try:
        # Create project record
        project_id = add_project(project_name, base_folder)
        logger.info(f"Created project in database: {project_name} (ID: {project_id})")

        # Create tester records
        for tester in testers:
            username = tester.get('username')
            password = tester.get('password')

            if not username or not password:
                logger.warning(f"Skipping tester with missing credentials: {tester}")
                continue

            try:
                tester_id = add_tester(username, password, project_id)
                testers_created += 1
                logger.info(f"Created tester in database: {username} (ID: {tester_id})")
            except ValueError as e:
                # Tester might already exist - log warning but continue
                logger.warning(f"Could not create tester {username}: {e}")
            except Exception as e:
                logger.error(f"Error creating tester {username}: {e}")
                raise

        message = f"Created project (ID: {project_id}) and {testers_created} tester(s)"
        return True, message, project_id, testers_created

    except Exception as e:
        logger.error(f"Failed to create database records: {e}")
        return False, f"Database error: {e}", project_id, testers_created


def rollback_changes(
    created_folders: List[str],
    created_files: List[str],
    project_id: Optional[int]
) -> None:
    """
    Rollback all changes made during project setup.

    This function removes created folders, files, and database records
    if project setup fails.

    Args:
        created_folders: List of folder paths that were created
        created_files: List of file paths that were created
        project_id: Project ID in database (if created)

    Example:
        >>> rollback_changes(['/path/folder1'], ['/path/file1'], 1)
    """
    logger.info("Starting rollback of changes")

    # Delete created files
    for file_path in created_files:
        try:
            if Path(file_path).exists():
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")

    # Delete created folders (in reverse order to delete children first)
    for folder_path in reversed(created_folders):
        try:
            folder = Path(folder_path)
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder)
                logger.info(f"Deleted folder: {folder_path}")
        except Exception as e:
            logger.error(f"Failed to delete folder {folder_path}: {e}")

    # TODO: Delete database records
    # Note: database.db.py doesn't currently have delete functions
    # This should be implemented for complete rollback capability
    if project_id:
        logger.warning(f"Database rollback not implemented - project_id {project_id} remains in database")
        logger.warning("Manual cleanup may be required in database")

    logger.info("Rollback completed")
