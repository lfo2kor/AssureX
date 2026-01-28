"""
SQLite database operations for projects and testers.

This module provides all database operations for managing projects and tester accounts
in the test automation system. It uses SQLite for storage and bcrypt for password hashing.
"""

import sqlite3
import bcrypt
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

# Setup logging
logger = logging.getLogger(__name__)

# Database file location
DB_DIR = Path(__file__).parent
DB_FILE = DB_DIR / "testers.db"


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    Provides a database connection with proper error handling and automatic cleanup.
    Uses WAL mode for better concurrent access and sets a timeout for locked database.

    Yields:
        sqlite3.Connection: Database connection object

    Raises:
        sqlite3.Error: If database connection fails
    """
    conn = None
    try:
        conn = sqlite3.connect(str(DB_FILE), timeout=10.0)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        # Enable Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def create_tables() -> None:
    """
    Create database tables if they don't exist.

    Creates the 'projects' and 'testers' tables with proper schema and constraints.
    Safe to call multiple times - will only create tables if they don't exist.

    Raises:
        sqlite3.Error: If table creation fails
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    base_folder TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Create testers table with foreign key to projects
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS testers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    project_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

            # Create index on project_id for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_testers_project_id
                ON testers(project_id)
            """)

            logger.info("Database tables created successfully")

    except sqlite3.Error as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def add_project(name: str, base_folder: str) -> int:
    """
    Add a new project to the database.

    Args:
        name: Project name (must be unique)
        base_folder: Base folder path for the project

    Returns:
        int: The ID of the newly created project

    Raises:
        ValueError: If project name already exists
        sqlite3.Error: If database operation fails
    """
    try:
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        if not base_folder or not base_folder.strip():
            raise ValueError("Base folder cannot be empty")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check if project already exists
            cursor.execute("SELECT id FROM projects WHERE name = ?", (name,))
            if cursor.fetchone():
                raise ValueError(f"Project '{name}' already exists")

            # Insert new project
            created_at = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO projects (name, base_folder, created_at) VALUES (?, ?, ?)",
                (name.strip(), base_folder.strip(), created_at)
            )

            project_id = cursor.lastrowid
            logger.info(f"Created project '{name}' with ID {project_id}")
            return project_id

    except sqlite3.IntegrityError as e:
        logger.error(f"Project '{name}' already exists: {e}")
        raise ValueError(f"Project '{name}' already exists")
    except sqlite3.Error as e:
        logger.error(f"Failed to add project '{name}': {e}")
        raise


def add_tester(username: str, password: str, project_id: int) -> int:
    """
    Add a new tester account to the database.

    The password is hashed using bcrypt before storing.

    Args:
        username: Tester username (must be unique)
        password: Plain text password (will be hashed)
        project_id: ID of the project this tester belongs to

    Returns:
        int: The ID of the newly created tester account

    Raises:
        ValueError: If username already exists or project_id is invalid
        sqlite3.Error: If database operation fails
    """
    try:
        # Validate inputs
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        if not password:
            raise ValueError("Password cannot be empty")
        if project_id <= 0:
            raise ValueError("Invalid project_id")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check if project exists
            cursor.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
            if not cursor.fetchone():
                raise ValueError(f"Project with ID {project_id} does not exist")

            # Check if username already exists
            cursor.execute("SELECT id FROM testers WHERE username = ?", (username,))
            if cursor.fetchone():
                raise ValueError(f"Username '{username}' already exists")

            # Hash the password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert new tester
            created_at = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO testers (username, password_hash, project_id, created_at) VALUES (?, ?, ?, ?)",
                (username.strip(), password_hash, project_id, created_at)
            )

            tester_id = cursor.lastrowid
            logger.info(f"Created tester '{username}' with ID {tester_id} for project {project_id}")
            return tester_id

    except sqlite3.IntegrityError as e:
        logger.error(f"Tester '{username}' already exists: {e}")
        raise ValueError(f"Username '{username}' already exists")
    except sqlite3.Error as e:
        logger.error(f"Failed to add tester '{username}': {e}")
        raise


def authenticate_tester(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a tester account.

    Verifies the username exists and the password matches the stored hash.

    Args:
        username: Tester username
        password: Plain text password to verify

    Returns:
        Dict containing tester info if authentication succeeds:
            {
                'id': int,
                'username': str,
                'project_id': int,
                'created_at': str
            }
        None if authentication fails

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if not username or not password:
            logger.warning("Authentication failed: Empty username or password")
            return None

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get tester record
            cursor.execute(
                "SELECT id, username, password_hash, project_id, created_at FROM testers WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Authentication failed: Username '{username}' not found")
                return None

            # Verify password
            stored_hash = row['password_hash']
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                logger.info(f"Authentication successful for user '{username}'")
                return {
                    'id': row['id'],
                    'username': row['username'],
                    'project_id': row['project_id'],
                    'created_at': row['created_at']
                }
            else:
                logger.warning(f"Authentication failed: Invalid password for user '{username}'")
                return None

    except sqlite3.Error as e:
        logger.error(f"Authentication error for user '{username}': {e}")
        raise


def get_project_by_name(name: str) -> Optional[Dict]:
    """
    Retrieve a project by its name.

    Args:
        name: Project name to search for

    Returns:
        Dict containing project info if found:
            {
                'id': int,
                'name': str,
                'base_folder': str,
                'created_at': str
            }
        None if project not found

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if not name:
            return None

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, base_folder, created_at FROM projects WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    except sqlite3.Error as e:
        logger.error(f"Failed to get project '{name}': {e}")
        raise


def get_project_by_id(project_id: int) -> Optional[Dict]:
    """
    Retrieve a project by its ID.

    Args:
        project_id: Project ID to search for

    Returns:
        Dict containing project info if found:
            {
                'id': int,
                'name': str,
                'base_folder': str,
                'created_at': str
            }
        None if project not found

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if project_id <= 0:
            return None

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, base_folder, created_at FROM projects WHERE id = ?",
                (project_id,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    except sqlite3.Error as e:
        logger.error(f"Failed to get project with ID {project_id}: {e}")
        raise


def get_testers_by_project(project_id: int) -> List[Dict]:
    """
    Retrieve all testers for a specific project.

    Args:
        project_id: Project ID to get testers for

    Returns:
        List of dictionaries containing tester info:
            [
                {
                    'id': int,
                    'username': str,
                    'project_id': int,
                    'created_at': str
                },
                ...
            ]
        Empty list if no testers found

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if project_id <= 0:
            return []

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, project_id, created_at FROM testers WHERE project_id = ?",
                (project_id,)
            )
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    except sqlite3.Error as e:
        logger.error(f"Failed to get testers for project {project_id}: {e}")
        raise


def project_exists(name: str) -> bool:
    """
    Check if a project with the given name exists.

    Args:
        name: Project name to check

    Returns:
        bool: True if project exists, False otherwise

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if not name:
            return False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM projects WHERE name = ?", (name,))
            return cursor.fetchone() is not None

    except sqlite3.Error as e:
        logger.error(f"Failed to check if project '{name}' exists: {e}")
        raise


def tester_exists(username: str) -> bool:
    """
    Check if a tester with the given username exists.

    Args:
        username: Username to check

    Returns:
        bool: True if tester exists, False otherwise

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        if not username:
            return False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM testers WHERE username = ?", (username,))
            return cursor.fetchone() is not None

    except sqlite3.Error as e:
        logger.error(f"Failed to check if tester '{username}' exists: {e}")
        raise
