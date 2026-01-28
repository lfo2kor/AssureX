"""
Pydantic models for database validation.

This module contains data models for validating project and tester data
before database operations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


class ProjectCreate(BaseModel):
    """
    Model for creating a new project.

    This model is used when creating a new project entry in the database.
    It validates the project name and base folder path.

    Attributes:
        name: Project name (3-50 characters)
        base_folder: Base folder path for the project

    Example:
        >>> project = ProjectCreate(name="plcd_test", base_folder="Projects")
        >>> print(project.name)
        'plcd_test'
    """
    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Project name (3-50 characters)"
    )
    base_folder: str = Field(
        ...,
        description="Base folder path for the project"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """
        Validate project name format.

        Args:
            v: Project name to validate

        Returns:
            Validated project name

        Raises:
            ValueError: If name contains invalid characters
        """
        # Allow alphanumeric, underscore, hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                'Project name can only contain letters, numbers, underscores, and hyphens'
            )
        return v

    @field_validator('base_folder')
    @classmethod
    def validate_base_folder(cls, v: str) -> str:
        """
        Validate base folder is not empty.

        Args:
            v: Base folder path to validate

        Returns:
            Validated base folder path

        Raises:
            ValueError: If base folder is empty
        """
        if not v.strip():
            raise ValueError('Base folder cannot be empty')
        return v.strip()


class ProjectDB(BaseModel):
    """
    Model for project data from database.

    This model represents a project record retrieved from the database,
    including its auto-generated ID and creation timestamp.

    Attributes:
        id: Project ID (auto-generated)
        name: Project name
        base_folder: Base folder path
        created_at: ISO format timestamp of creation

    Example:
        >>> project = ProjectDB(
        ...     id=1,
        ...     name="plcd_test",
        ...     base_folder="Projects",
        ...     created_at="2025-01-15T10:30:00"
        ... )
        >>> print(f"Project {project.name} created at {project.created_at}")
        'Project plcd_test created at 2025-01-15T10:30:00'
    """
    id: int = Field(..., description="Project ID (auto-generated)")
    name: str = Field(..., description="Project name")
    base_folder: str = Field(..., description="Base folder path")
    created_at: str = Field(..., description="ISO format timestamp of creation")


class TesterCreate(BaseModel):
    """
    Model for creating a new tester account.

    This model is used when creating a new tester entry in the database.
    It validates username, password strength, and project association.

    Attributes:
        username: Tester username (3-50 characters, alphanumeric and underscore)
        password: Password (minimum 6 characters)
        project_id: ID of the associated project (must be > 0)

    Example:
        >>> tester = TesterCreate(
        ...     username="john_doe",
        ...     password="secure123",
        ...     project_id=1
        ... )
        >>> print(tester.username)
        'john_doe'
    """
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Tester username (3-50 characters)"
    )
    password: str = Field(
        ...,
        min_length=6,
        description="Password (minimum 6 characters)"
    )
    project_id: int = Field(
        ...,
        gt=0,
        description="ID of the associated project"
    )

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """
        Validate username format.

        Username must contain only alphanumeric characters and underscores.

        Args:
            v: Username to validate

        Returns:
            Validated username

        Raises:
            ValueError: If username contains invalid characters
        """
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError(
                'Username can only contain letters, numbers, and underscores'
            )
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """
        Validate password strength.

        Password must be at least 6 characters long and cannot be all whitespace.

        Args:
            v: Password to validate

        Returns:
            Validated password

        Raises:
            ValueError: If password is too weak
        """
        if not v.strip():
            raise ValueError('Password cannot be empty or only whitespace')
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v


class TesterDB(BaseModel):
    """
    Model for tester data from database.

    This model represents a tester record retrieved from the database.
    Note: password_hash is intentionally excluded for security reasons.

    Attributes:
        id: Tester ID (auto-generated)
        username: Tester username
        project_id: ID of the associated project
        created_at: ISO format timestamp of creation

    Example:
        >>> tester = TesterDB(
        ...     id=1,
        ...     username="john_doe",
        ...     project_id=1,
        ...     created_at="2025-01-15T10:30:00"
        ... )
        >>> print(f"Tester {tester.username} in project {tester.project_id}")
        'Tester john_doe in project 1'
    """
    id: int = Field(..., description="Tester ID (auto-generated)")
    username: str = Field(..., description="Tester username")
    project_id: int = Field(..., description="ID of the associated project")
    created_at: str = Field(..., description="ISO format timestamp of creation")
