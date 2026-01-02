"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============================================================================
# PROJECT SCHEMAS
# ============================================================================

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=255, description="Project name")
    base_folder: str = Field(..., description="Base folder path")
    web_url: Optional[str] = Field(None, description="Web application URL")
    browser: str = Field(default="edge", description="Browser type (edge, chrome, firefox)")


class ProjectResponse(BaseModel):
    id: int
    name: str
    base_folder: str
    web_url: Optional[str]
    browser: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# TICKET SCHEMAS
# ============================================================================

class TicketStep(BaseModel):
    num: int
    text: str


class TicketCreate(BaseModel):
    ticket_id: str = Field(..., description="Jira ticket ID (e.g., RBPLCD-8835)")
    project_id: int = Field(..., description="Project ID this ticket belongs to")
    title: Optional[str] = None
    module: Optional[str] = None
    description: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    file_path: Optional[str] = None


class TicketResponse(BaseModel):
    id: int
    ticket_id: str
    project_id: int
    title: Optional[str]
    module: Optional[str]
    description: Optional[str]
    acceptance_criteria: Optional[str]
    file_path: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class TicketDetailResponse(TicketResponse):
    """Extended ticket response with parsed steps"""
    steps: List[TicketStep] = []


# ============================================================================
# TEST EXECUTION SCHEMAS
# ============================================================================

class ExecutionStepResult(BaseModel):
    step_num: int
    step_text: str
    status: str  # "PASSED", "FAILED"
    selector_used: Optional[str]
    level_used: Optional[str]  # "L1", "L2", "L3"
    confidence: Optional[float]
    execution_time: Optional[float]
    screenshot_before: Optional[str]
    screenshot_after: Optional[str]
    error_message: Optional[str]


class TestExecutionResponse(BaseModel):
    id: int
    execution_id: str
    ticket_id: str
    project_id: int
    status: str  # "running", "completed", "failed"
    overall_status: Optional[str]  # "PASSED", "FAILED", "UNKNOWN"
    started_at: datetime
    completed_at: Optional[datetime]
    total_execution_time: Optional[float]
    report_path: Optional[str]
    script_path: Optional[str]
    video_path: Optional[str]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class TestExecutionDetailResponse(TestExecutionResponse):
    """Extended execution response with step details"""
    steps: List[ExecutionStepResult] = []