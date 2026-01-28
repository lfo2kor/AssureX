"""
Admin Upload Page for Test Automation Portal.

This module provides the Streamlit UI for uploading config.yaml files
and setting up new testing projects.
"""

import os
import yaml
import logging
import streamlit as st
from pathlib import Path
from typing import Dict, Optional, Any

from agents.project_manager_agent import project_manager_agent

# Setup logging
logger = logging.getLogger(__name__)

# Application root path (hardcoded for now)
APP_ROOT = "C:/Projects/AI_Chat/PLCD/TA_AI_Project"


def show_admin_upload_page() -> None:
    """
    Display the admin upload page for config.yaml files.

    This page allows administrators to:
    - Upload config.yaml files
    - Preview configuration
    - Validate and create projects
    - View tester accounts

    Example:
        >>> show_admin_upload_page()
    """
    # Initialize session state
    initialize_session_state()

    # Render header section
    render_header()

    # Render file upload section
    render_file_upload()

    # If file is uploaded, show preview and validation
    if st.session_state.uploaded_file is not None:
        render_file_preview()
        render_validation_section()

    # Show results if validation has been run
    if st.session_state.show_results and st.session_state.validation_result:
        render_results()


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state variables.

    Creates default values for session state if they don't exist.
    """
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    if 'parsed_config' not in st.session_state:
        st.session_state.parsed_config = None

    if 'validation_result' not in st.session_state:
        st.session_state.validation_result = None

    if 'show_results' not in st.session_state:
        st.session_state.show_results = False


def render_header() -> None:
    """
    Render the page header section.

    Displays title, description, and divider.
    """
    st.title("📤 Upload Project Configuration")
    st.markdown(
        "Upload a **config.yaml** file to set up a new testing project. "
        "The system will validate the configuration and create the necessary "
        "folder structure and database records."
    )
    st.divider()


def render_file_upload() -> None:
    """
    Render the file upload section.

    Provides file uploader widget for config.yaml files.
    """
    uploaded_file = st.file_uploader(
        "Choose config.yaml file",
        type=['yaml', 'yml'],
        help="Upload your project configuration file",
        key="config_file_uploader"
    )

    # Handle file upload
    if uploaded_file is not None:
        # Check if it's a new file
        if (st.session_state.uploaded_file is None or
            uploaded_file.name != st.session_state.uploaded_file.name):
            # Reset state for new file
            st.session_state.uploaded_file = uploaded_file
            st.session_state.show_results = False
            st.session_state.validation_result = None

            # Parse the YAML file
            try:
                content = uploaded_file.read()
                parsed_config = yaml.safe_load(content)
                st.session_state.parsed_config = parsed_config
                logger.info(f"Successfully parsed config file: {uploaded_file.name}")
            except yaml.YAMLError as e:
                st.error(f"❌ Invalid YAML file: {e}")
                st.session_state.parsed_config = None
                st.session_state.uploaded_file = None
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.session_state.parsed_config = None
                st.session_state.uploaded_file = None
    else:
        # File removed
        if st.session_state.uploaded_file is not None:
            st.session_state.uploaded_file = None
            st.session_state.parsed_config = None
            st.session_state.show_results = False
            st.session_state.validation_result = None


def render_file_preview() -> None:
    """
    Render the configuration file preview section.

    Displays key information from the parsed config.
    """
    if st.session_state.parsed_config is None:
        return

    config = st.session_state.parsed_config

    with st.expander("📄 Configuration Preview", expanded=True):
        try:
            # Extract key information
            project_name = config.get('project', {}).get('name', 'N/A')
            base_folder = config.get('base_folder', 'N/A')
            testers = config.get('testers', [])
            web_url = config.get('web_application', {}).get('url', 'N/A')
            module_mapping = config.get('module_mapping', {})

            # Display in columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📁 Project Information**")
                st.markdown(f"- **Project Name:** `{project_name}`")
                st.markdown(f"- **Base Folder:** `{base_folder}`")
                st.markdown(f"- **Web URL:** `{web_url}`")

            with col2:
                st.markdown("**👥 Testers & Modules**")
                st.markdown(f"- **Number of Testers:** {len(testers)}")
                if testers:
                    tester_usernames = [t.get('username', 'N/A') for t in testers]
                    st.markdown(f"- **Tester Usernames:** {', '.join(tester_usernames)}")
                st.markdown(f"- **Module Mappings:** {len(module_mapping)}")

            # Show full config in expander
            with st.expander("View Full Configuration"):
                st.json(config)

        except Exception as e:
            st.error(f"Error displaying preview: {e}")
            logger.error(f"Error in preview: {e}", exc_info=True)


def render_validation_section() -> None:
    """
    Render the validation section with button and progress.

    Provides button to trigger validation and shows progress.
    """
    if st.session_state.parsed_config is None:
        return

    st.divider()

    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        validate_button = st.button(
            "🔍 Validate & Create Project",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.show_results
        )

    if validate_button:
        run_validation()


def run_validation() -> None:
    """
    Run the project validation and creation process.

    Calls the project_manager_agent and displays progress.
    """
    try:
        # Prepare state
        state = {
            'uploaded_config': st.session_state.parsed_config,
            'app_root': APP_ROOT
        }

        # Run validation with progress indicator
        with st.spinner("🔄 Validating configuration and creating project..."):
            result = project_manager_agent(state)

        # Store result and show it
        st.session_state.validation_result = result
        st.session_state.show_results = True

        # Force rerun to show results
        st.rerun()

    except Exception as e:
        st.error(f"❌ Unexpected error during validation: {e}")
        logger.error(f"Validation error: {e}", exc_info=True)


def render_results() -> None:
    """
    Render the validation results section.

    Displays success or failure message with details.
    """
    result = st.session_state.validation_result
    if result is None:
        return

    st.divider()

    validation_status = result.get('validation_status', 'unknown')

    if validation_status == 'success':
        render_success_results(result)
    else:
        render_failure_results(result)

    # Show validation steps
    render_validation_steps(result)


def render_success_results(result: Dict[str, Any]) -> None:
    """
    Render success results section.

    Args:
        result: Validation result dictionary from project_manager_agent

    Example:
        >>> render_success_results(validation_result)
    """
    st.success("✅ Project Created Successfully!")

    # Project information container
    with st.container():
        st.markdown("### 📊 Project Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Project Name", result.get('project_name', 'N/A'))
            st.metric("Project ID", result.get('project_id', 'N/A'))

        with col2:
            st.metric("Testers Created", result.get('testers_created', 0))
            project_path = result.get('project_path', 'N/A')
            st.info(f"📁 **Project Location:**\n`{project_path}`")

    # Show tester accounts
    render_tester_accounts(result)

    # Show warnings if any
    if result.get('validation_warnings'):
        st.warning("⚠️ **Warnings:**")
        for warning in result['validation_warnings']:
            st.markdown(f"- {warning}")

    # Reset button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕ Create Another Project", type="secondary", use_container_width=True):
            reset_form()


def render_failure_results(result: Dict[str, Any]) -> None:
    """
    Render failure results section.

    Args:
        result: Validation result dictionary from project_manager_agent

    Example:
        >>> render_failure_results(validation_result)
    """
    st.error("❌ Project Creation Failed")

    # Show errors
    errors = result.get('validation_errors', [])
    if errors:
        st.markdown("### 🚫 Errors:")
        for error in errors:
            st.error(f"**Error:** {error}")

    # Show warnings if any
    warnings = result.get('validation_warnings', [])
    if warnings:
        st.markdown("### ⚠️ Warnings:")
        for warning in warnings:
            st.warning(warning)

    # Try again button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Try Again", type="secondary", use_container_width=True):
            reset_form()


def render_validation_steps(result: Dict[str, Any]) -> None:
    """
    Render validation steps in an expander.

    Args:
        result: Validation result dictionary from project_manager_agent

    Example:
        >>> render_validation_steps(validation_result)
    """
    validation_steps = result.get('validation_steps', [])

    if not validation_steps:
        return

    with st.expander("🔍 Validation Steps Details", expanded=False):
        for step in validation_steps:
            step_name = step.get('step', 'Unknown')
            status = step.get('status', 'unknown')
            message = step.get('message', '')

            # Choose icon based on status
            if status == 'passed':
                icon = "✅"
                color = "green"
            elif status == 'failed':
                icon = "❌"
                color = "red"
            elif status == 'warning':
                icon = "⚠️"
                color = "orange"
            elif status == 'skipped':
                icon = "○"
                color = "gray"
            else:
                icon = "➖"
                color = "blue"

            # Display step
            st.markdown(f"{icon} **{step_name}:** {message}")


def render_tester_accounts(result: Dict[str, Any]) -> None:
    """
    Render tester accounts information.

    Args:
        result: Validation result dictionary from project_manager_agent

    Example:
        >>> render_tester_accounts(validation_result)
    """
    # Get testers from original config
    config = st.session_state.parsed_config
    testers = config.get('testers', [])

    if not testers or result.get('testers_created', 0) == 0:
        return

    with st.expander("👥 Tester Accounts", expanded=True):
        st.markdown("**Login Credentials for Testers:**")

        # Create table header
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown("**Username**")
        with col2:
            st.markdown("**Password**")
        with col3:
            st.markdown("**Login URL**")

        st.divider()

        # Display each tester
        for tester in testers:
            username = tester.get('username', 'N/A')
            password = tester.get('password', 'N/A')
            login_url = "http://localhost:8501/tester"  # TODO: Implement tester portal

            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.code(username)
            with col2:
                st.code(password)
            with col3:
                st.markdown(f"`{login_url}` (TODO)")

        st.info("ℹ️ **Note:** Passwords are shown here for initial setup. "
                "Testers should change their passwords after first login. "
                "Tester portal is not yet implemented (marked as TODO).")


def reset_form() -> None:
    """
    Reset the form to initial state.

    Clears all session state variables.
    """
    st.session_state.uploaded_file = None
    st.session_state.parsed_config = None
    st.session_state.validation_result = None
    st.session_state.show_results = False
    st.rerun()
