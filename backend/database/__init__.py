"""
Database package for test automation system.

This package provides database operations for managing projects and testers.
"""

from .db import (
    create_tables,
    add_project,
    add_tester,
    authenticate_tester,
    get_project_by_name,
    get_project_by_id,
    get_testers_by_project,
    project_exists,
    tester_exists,
)

from .models import (
    ProjectCreate,
    ProjectDB,
    TesterCreate,
    TesterDB,
)

__all__ = [
    # Database functions
    'create_tables',
    'add_project',
    'add_tester',
    'authenticate_tester',
    'get_project_by_name',
    'get_project_by_id',
    'get_testers_by_project',
    'project_exists',
    'tester_exists',
    # Models
    'ProjectCreate',
    'ProjectDB',
    'TesterCreate',
    'TesterDB',
]
