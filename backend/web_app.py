"""
Test Automation - Admin Portal

Main entry point for the Streamlit web application.
This application provides an admin interface for uploading config.yaml
files and setting up new testing projects.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Test Automation - Admin Portal",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import admin page
from web.admin_upload import show_admin_upload_page

# Show the admin upload page
show_admin_upload_page()
