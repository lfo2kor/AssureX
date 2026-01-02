"""
SQLAlchemy models for database tables
"""
from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    base_folder = Column(String(500), nullable=False)
    web_url = Column(String(500))
    browser = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    testers = relationship("Tester", back_populates="project", cascade="all, delete-orphan")
    tickets = relationship("Ticket", back_populates="project", cascade="all, delete-orphan")
    test_executions = relationship("TestExecution", back_populates="project", cascade="all, delete-orphan")
    configs = relationship("ProjectConfig", back_populates="project", cascade="all, delete-orphan")


class Tester(Base):
    __tablename__ = "testers"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="testers")


class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(100), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    title = Column(Text)
    module = Column(String(255))
    description = Column(Text)
    acceptance_criteria = Column(Text)
    file_path = Column(String(500))
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="tickets")
    test_executions = relationship("TestExecution", back_populates="ticket", cascade="all, delete-orphan")


class TestExecution(Base):
    __tablename__ = "test_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    ticket_id = Column(String(100), ForeignKey("tickets.ticket_id", ondelete="CASCADE"), index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    status = Column(String(50), nullable=False, index=True)  # "running", "completed", "failed"
    overall_status = Column(String(50))  # "PASSED", "FAILED", "UNKNOWN"
    started_at = Column(TIMESTAMP, server_default=func.now())
    completed_at = Column(TIMESTAMP)
    total_execution_time = Column(Float)
    report_path = Column(String(500))
    script_path = Column(String(500))
    video_path = Column(String(500))
    error_message = Column(Text)
    
    # Relationships
    project = relationship("Project", back_populates="test_executions")
    ticket = relationship("Ticket", back_populates="test_executions")
    steps = relationship("ExecutionStep", back_populates="execution", cascade="all, delete-orphan")


class ExecutionStep(Base):
    __tablename__ = "execution_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(100), ForeignKey("test_executions.execution_id", ondelete="CASCADE"), index=True)
    step_num = Column(Integer, nullable=False)
    step_text = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)  # "PASSED", "FAILED"
    selector_used = Column(Text)
    level_used = Column(String(50))  # "L1", "L2", "L3"
    confidence = Column(Float)
    execution_time = Column(Float)
    screenshot_before = Column(String(500))
    screenshot_after = Column(String(500))
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    execution = relationship("TestExecution", back_populates="steps")


class ProjectConfig(Base):
    __tablename__ = "project_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    config_key = Column(String(255), nullable=False)
    config_value = Column(Text)
    
    # Relationships
    project = relationship("Project", back_populates="configs")