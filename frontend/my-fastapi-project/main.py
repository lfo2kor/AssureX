"""
FastAPI Application for Test Automation Backend - CLEANED VERSION
"""
from unittest import result
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
from datetime import timezone
import logging
import re  # Add this import
import json
from openai import AzureOpenAI
from fastapi import Query
from config_loader import load_config, get_azure_client
from fastapi import BackgroundTasks
from fastapi import Body
import time

from pydantic import BaseModel
from typing import List, Optional
from config import settings
from utils import setup_logging
from database import engine, get_db, Base, SessionLocal
from models import Ticket, TestExecution, ExecutionStep
from services import TestExecutionService
from jira_api import router as jira_router  # 🟢 ADD THIS LINE
from typing import Optional
import os
import requests
import subprocess
from dotenv import load_dotenv
from selector_feedback import router as selector_feedback_router
from models import Base

load_dotenv()
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")

# Setup logging
logger = setup_logging(settings.log_level)
logger.info("Starting Test Automation API...")

# Create all tables
Base.metadata.create_all(bind=engine)
logger.info("Database tables verified")
print("EXTERNAL_PROJECT_PATH =", os.getenv("EXTERNAL_PROJECT_PATH"))

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Test Automation API",
    description="AI-Powered Vision-Based Test Automation Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

azure_client = AzureOpenAI(
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint
)

PENDING_DIR = r"C:\AssureX\backend\insights\pending"

class Feedback(BaseModel):
    step: str
    selector: str
    ticket_id: str
    step_number: int

# def generate_embedding(text: str):
#     # Dummy embedding, replace with actual model
#     return [0.1, 0.2, 0.3]


# def generate_embedding(text: str):
#     response = azure_client.embeddings.create(
#         input=text,
#         model=settings.azure_openai_embedding_model
#     )
#     return response.data[0].embedding  # This will be a list of 1536 floats

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # In production, specify exact origins
    allow_origins=["http://localhost:4200", "http://si0vm10371.de.bosch.com"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🟢 INCLUDE JIRA ROUTER
app.include_router(jira_router)
app.include_router(selector_feedback_router)

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Root endpoint - API health check"""
    return {
        "message": "Test Automation API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }




# ============================================================================
# CORE TEST EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/execute-test")
async def execute_test(
    ticket_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Fetch ticket from Jira (not from local DB) and run the same Playwright test.
    """

     # 🔥 ADD THIS AT THE VERY TOP
    print(f"🔥🔥🔥 ENDPOINT CALLED: /api/execute-test with ticket_id={ticket_id}")
    logger.info(f"🔥🔥🔥 ENDPOINT CALLED: /api/execute-test with ticket_id={ticket_id}")

    try:
        logger.info(f"📥 Fetching ticket from Jira: {ticket_id}")

        # Fetch ticket from Jira instead of local database
        url = f"{JIRA_BASE_URL}/rest/api/2/issue/{ticket_id}"
        headers = {
            "Authorization": f"Bearer {JIRA_API_TOKEN}",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to fetch Jira ticket '{ticket_id}': {response.text}"
            )

        jira_data = response.json()
        fields = jira_data.get("fields", {})
        summary = fields.get("summary", "")
        description = fields.get("description", "")

        logger.info(f"✅ Fetched from Jira: {summary}")

        # 🔥 VALIDATION: Check if description exists and is not empty
        if not description or (isinstance(description, str) and not description.strip()):
            error_msg = f"No description available for Jira ticket '{ticket_id}'. Please add test steps/description in Jira before executing."
            logger.warning(f"⚠️  {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        logger.info(f"✅ Description present: {len(description)} characters")

        # Create execution record (optional - can be removed if you don't want ANY database)
        service = TestExecutionService(db)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"exec_{ticket_id}_{timestamp}"

        # If you removed the foreign key constraint, this will work:
        execution = TestExecution(
            execution_id=execution_id,
            ticket_id=ticket_id,
            project_id=None,  # No local project
            status="pending",
            overall_status="UNKNOWN",
            started_at=datetime.now()
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        logger.info(f"✅ Created execution: {execution.execution_id}")

        # Use your EXISTING background task (same Playwright execution)
        background_tasks.add_task(
            execute_test_in_background,
            execution_id=execution.execution_id,
            ticket_id=ticket_id,
            project_id=None  # Pass None since no local project
        )

        return {
            "execution_id": execution.execution_id,
            "ticket_id": ticket_id,
            "status": "pending",
            "message": f"Test execution started for: {summary}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






# Add this to your main.py after the execute_test endpoint

# ============================================================================
# RERUN TEST EXECUTION ENDPOINT
# ============================================================================

@app.post("/api/rerun-test")
async def rerun_test(
    ticket_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Rerun test for a ticket using the latest generated script

    Query Params:
        - ticket_id: JIRA ticket ID (e.g., RBPLCD-8835)

    Returns:
        {
            "execution_id": "rerun_RBPLCD-8835_20251126_120000",
            "ticket_id": "RBPLCD-8835",
            "status": "pending",
            "script_path": "path/to/script.py",
            "message": "Test rerun started"
        }
    """
    try:
        logger.info(f"📥 Received rerun request for ticket: {ticket_id}")


        # REMOVE ticket DB check
        # ticket = None
        project_id = None


        # Find the latest generated script for this ticket
        external_path = Path(settings.external_project_path)
        scripts_folder = external_path / "Generated_Scripts"

        if not scripts_folder.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Scripts folder not found: {scripts_folder}"
            )

        # Find all scripts matching the ticket_id pattern
        script_pattern = f"*{ticket_id}*.py"
        matching_scripts = list(scripts_folder.glob(script_pattern))

        if not matching_scripts:
            raise HTTPException(
                status_code=404,
                detail=f"No generated script found for ticket '{ticket_id}'. Please run the test first."
            )

        # Get the latest script by modification time
        latest_script = max(matching_scripts, key=lambda p: p.stat().st_mtime)

        logger.info(f"📜 Found latest script: {latest_script.name}")

        # Create execution record for rerun
        service = TestExecutionService(db)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"rerun_{ticket_id}_{timestamp}"

        execution = TestExecution(
            execution_id=execution_id,
            ticket_id=ticket_id,
            # project_id=ticket.project_id,
            project_id=project_id,
            status="pending",
            overall_status="UNKNOWN",
            started_at=datetime.now()
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        logger.info(f"✅ Created rerun execution: {execution.execution_id}")

        # Start background task for rerun
        background_tasks.add_task(
            rerun_test_in_background,
            execution_id=execution.execution_id,
            ticket_id=ticket_id,
            script_path=str(latest_script),
            # project_id=ticket.project_id
            project_id=project_id
        )

        return {
            "execution_id": execution.execution_id,
            "ticket_id": ticket_id,
            "status": "pending",
            "script_path": str(latest_script),
            "message": f"Test rerun started using script: {latest_script.name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting rerun: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RERUN BACKGROUND TASK
# ============================================================================

def rerun_test_in_background(
    execution_id: str,
    ticket_id: str,
    script_path: str,
    project_id: int
):
    """
    Background task to rerun test by executing the generated script directly
    Similar to calling python plcd_taseq.py
    """
    import subprocess
    import re

    db = SessionLocal()
    service = TestExecutionService(db)

    logger.info("="*70)
    logger.info(f"🔄 RERUN TASK STARTED")
    logger.info(f"   Execution ID: {execution_id}")
    logger.info(f"   Ticket ID: {ticket_id}")
    logger.info(f"   Script: {script_path}")
    logger.info(f"   Started at: {datetime.now().isoformat()}")
    logger.info("="*70)

    try:
        # Update status to running
        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()
        execution.status = "running"
        db.commit()
        logger.info("✅ Status updated to 'running'")

        # Path to external project
        external_project_path = Path(settings.external_project_path)

        # Find Python executable
        python_exe = None
        venv_paths = [
            external_project_path / "venv" / "Scripts" / "python.exe",  # Windows
            external_project_path / "venv" / "bin" / "python",  # Linux/Mac
        ]

        for venv_path in venv_paths:
            if venv_path.exists():
                python_exe = str(venv_path)
                logger.info(f"✅ Found Python: {python_exe}")
                break

        if not python_exe:
            # Fallback to system Python
            import shutil
            python_exe = shutil.which("python") or shutil.which("python3")
            if not python_exe:
                raise FileNotFoundError("Python executable not found")
            logger.info(f"⚠️  Using system Python: {python_exe}")

        # Execute the script
        # logger.info(f"🏃 Executing script: {script_path}")

        # result = subprocess.run(
        #     [python_exe, script_path],
        #     cwd=str(external_project_path),
        #     capture_output=True,
        #     text=True,
        #     timeout=600  # 10 minutes timeout
        # )

        # logger.info(f"📤 Script execution completed with return code: {result.returncode}")

        # # Log output
        # if result.stdout:
        #     logger.info(f"STDOUT:\n{result.stdout[:1000]}")  # First 1000 chars
        # if result.stderr:
        #     logger.warning(f"STDERR:\n{result.stderr[:1000]}")
        logger.info(f"🏃 [RERUN_BG] About to execute script: {script_path}")
        #      result = subprocess.run(
        #     [python_exe, script_path],
        #     cwd=str(external_project_path),
        #     capture_output=True,
        #     text=True,
        #     timeout=600  # 10 minutes timeout
        # )   
        logger.info(f"🏃 [RERUN_BG] About to execute CLI pipeline for ticket: {ticket_id}")

        # result = subprocess.run(
        #     [
        #         python_exe,
        #         "--rerun"
        #     ],
        #     cwd=str(external_project_path),
        #     capture_output=True,
        #     text=True,
        #     timeout=600
        # )
        result = subprocess.run(
    [
        python_exe,
        "plcd_taseq.py",
        ticket_id,
        "--rerun",
        "--no-feedback"
    ],
    cwd=str(external_project_path),
    capture_output=True,
    text=True,
    timeout=600
)


        logger.info(f"🏃 [RERUN_BG] Script execution completed with return code: {result.returncode}")
        if result.stdout:
            logger.info(f"🏃 [RERUN_BG] STDOUT:\n{result.stdout[:1000]}")
        if result.stderr:
            logger.warning(f"🏃 [RERUN_BG] STDERR:\n{result.stderr[:1000]}")
            
         # 🔥 ADD THIS SECTION HERE (after line 447)
    # =====================================================
    # LOAD STEPS FROM JSON AND SAVE TO DB
    # =====================================================
        steps_file = external_project_path / "Reports" / "steps" / f"steps_{ticket_id}.json"
        steps_saved = False
    
        if steps_file.exists():
            with open(steps_file, "r", encoding="utf-8") as f:
                raw_steps = json.load(f)
        
            logger.info(f"Loaded {len(raw_steps)} steps from JSON")
        
        # Delete old steps for this execution
            db.query(ExecutionStep).filter(
                ExecutionStep.execution_id == execution_id
            ).delete()
            db.commit()
        
        # Save new steps to DB
            for step in raw_steps:
                db.add(ExecutionStep(
                    execution_id=execution_id,
                    step_num=step.get("step_number"),
                    step_text=step.get("step_text"),
                    status=step.get("status"),
                    selector_used=step.get("selector", ""),
                    agent_used=step.get("agent_used", ""),
                    confidence=step.get("confidence", 0.0),
                    action_type=step.get("action_type", "")
                ))    
        
            db.commit()
            steps_saved = True
            logger.info(f"✅ Saved {len(raw_steps)} steps to DB for {execution_id}")
        else:
            logger.error(f"❌ Steps file not found: {steps_file}")
            steps_saved = False    
        
# =====================================================
# LOAD STEPS FROM JSON AND SAVE TO DB
# =====================================================
            
        # Parse results from output or find generated files
        # Look for the latest report/video files
        reports_folder = external_project_path / "Reports"
        videos_folder = external_project_path / "Videos"

        # Find latest report for this ticket
        report_path = None
        if reports_folder.exists():
            reports = sorted(
                reports_folder.glob(f"*{ticket_id}*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if reports:
                report_path = str(reports[0])
                logger.info(f"📄 Found report: {reports[0].name}")

        # Find latest video (videos may not have ticket_id in name)
        video_path = None
        if videos_folder.exists():
            videos = sorted(
                videos_folder.glob("*.webm"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if videos:
                video_path = str(videos[0])
                logger.info(f"🎥 Found video: {videos[0].name}")

        # Parse overall status from report if available
        overall_status = "UNKNOWN"
        
        # 🔥 FIX: Calculate overall_status from steps (more reliable than parsing HTML)
        if steps_saved and raw_steps:
            overall_status = "FAILED" if any(
                s.get("status") == "FAILED" for s in raw_steps
            ) else "PASSED"
            logger.info(f"✅ Calculated overall_status from steps: {overall_status}")
        elif report_path and Path(report_path).exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Try multiple patterns to find status
                patterns = [
                    r'<h2[^>]*>\s*Overall\s+Status:\s*(PASSED|FAILED)\s*</h2>',
                    r'<div[^>]*class=["\']overall-status[^"\']*["\'][^>]*>\s*(PASSED|FAILED)',
                    r'Overall\s+Status:\s*<[^>]+>\s*(PASSED|FAILED)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, html_content, re.IGNORECASE)
                    if match:
                        overall_status = match.group(1).upper()
                        logger.info(f"✅ Parsed overall status: {overall_status}")
                        break

                # Fallback: count PASSED/FAILED in table
                if overall_status == "UNKNOWN":
                    passed_count = len(re.findall(r'>\s*PASSED\s*<', html_content, re.IGNORECASE))
                    failed_count = len(re.findall(r'>\s*FAILED\s*<', html_content, re.IGNORECASE))
                    if failed_count > 0:
                        overall_status = "FAILED"
                    elif passed_count > 0:
                        overall_status = "PASSED"
                    logger.info(f"📊 Inferred status from counts: {overall_status} (P:{passed_count}, F:{failed_count})")

            except Exception as e:
                logger.warning(f"Could not parse report status: {e}")

        # Parse steps from report if available
        if report_path and Path(report_path).exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Extract table rows
                table_match = re.search(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
                if table_match:
                    table_content = table_match.group(1)
                    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL | re.IGNORECASE)

                    step_num = 1
                    for row in rows[1:]:  # Skip header row
                        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
                        if len(cells) >= 3:
                            step_text = re.sub(r'<[^>]+>', '', cells[1]).strip()
                            status = re.sub(r'<[^>]+>', '', cells[2]).strip().upper()

                            if step_text and status in ['PASSED', 'FAILED']:
                                step = ExecutionStep(
                                    execution_id=execution_id,
                                    step_num=step_num,
                                    step_text=step_text,
                                    status=status,
                                    # screenshot_path=None
                                )
                                db.add(step)
                                step_num += 1

                    db.commit()
                    logger.info(f"✅ Saved {step_num-1} steps to database")

            except Exception as e:
                logger.warning(f"Could not parse steps from report: {e}")


        # Update execution with results
        execution.status = "completed"
        execution.overall_status = overall_status
        execution.report_path = report_path
        execution.script_path = script_path
        execution.video_path = video_path
        execution.completed_at = datetime.now()
        execution.error_message = None
        db.commit()

# 🔥 ADD THIS LINE HERE
        service._generate_summary_from_db(execution_id, ticket_id)
        logger.info("="*70)
        logger.info("✅ TEST RERUN COMPLETED SUCCESSFULLY")
        logger.info(f"   Execution ID: {execution_id}")
        logger.info(f"   Overall Status: {overall_status}")
        logger.info(f"   📄 Report: {report_path or 'N/A'}")
        logger.info(f"   📜 Script: {script_path}")
        logger.info(f"   🎥 Video: {video_path or 'N/A'}")
        logger.info(f"   Completed at: {datetime.now().isoformat()}")
        logger.info("="*70)

    except subprocess.TimeoutExpired:
        error_msg = "Test execution timed out (10 minutes limit)"
        logger.error(f"❌ {error_msg}")

        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()
        execution.status = "failed"
        execution.overall_status = "FAILED"
        execution.error_message = error_msg
        execution.completed_at = datetime.now()
        db.commit()

    except Exception as e:
        logger.error("="*70)
        logger.error(f"❌ RERUN TASK FAILED")
        logger.error(f"   Execution ID: {execution_id}")
        logger.error(f"   Error: {e}")
        logger.error(f"   Failed at: {datetime.now().isoformat()}")
        logger.error("="*70)

        error_message = str(e)[:500]

        try:
            execution = db.query(TestExecution).filter(
                TestExecution.execution_id == execution_id
            ).first()
            execution.status = "failed"
            execution.overall_status = "FAILED"
            execution.error_message = error_message
            execution.completed_at = datetime.now()
            db.commit()
            logger.info("✅ Updated execution status to 'failed' in database")
        except Exception as db_error:
            logger.error(f"❌ Could not update database with failure: {db_error}")

        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

    finally:
        db.close()
        logger.info(f"🏁 Rerun task ended for {execution_id}")
        logger.info("")


# ============================================================================
# HELPER ENDPOINT - List Available Scripts
# ============================================================================

@app.get("/api/scripts/{ticket_id}")
def list_generated_scripts(ticket_id: str):
    """
    List all generated scripts for a ticket

    Returns:
        {
            "ticket_id": "RBPLCD-8835",
            "scripts": [
                {
                    "filename": "RBPLCD-8835_20251126_120000.py",
                    "path": "full/path/to/script.py",
                    "created": "2025-11-26T12:00:00",
                    "size": 15234
                }
            ]
        }
    """
    try:
        external_path = Path(settings.external_project_path)
        scripts_folder = external_path / "Generated_Scripts"

        if not scripts_folder.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Scripts folder not found: {scripts_folder}"
            )

        # Find all scripts for this ticket
        script_pattern = f"*{ticket_id}*.py"
        matching_scripts = list(scripts_folder.glob(script_pattern))

        scripts = []
        for script_path in sorted(matching_scripts, key=lambda p: p.stat().st_mtime, reverse=True):
            stat = script_path.stat()
            scripts.append({
                "filename": script_path.name,
                "path": str(script_path),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size
            })

        return {
            "ticket_id": ticket_id,
            "scripts_count": len(scripts),
            "scripts": scripts
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing scripts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/execution-status/{execution_id}")
def get_execution_status(
    execution_id: str,
    db: Session = Depends(get_db)
):
    """
    Get real-time execution status with progress
    """
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Get step information
    total_steps = db.query(ExecutionStep).filter(
        ExecutionStep.execution_id == execution_id
    ).count()

    completed_steps = db.query(ExecutionStep).filter(
        ExecutionStep.execution_id == execution_id,
        ExecutionStep.status.in_(["PASSED", "FAILED"])
    ).count()

    # Calculate progress
    progress = 0
    message = "Initializing..."
    current_step = None

    if execution.status == "pending":
        progress = 0
        message = "Test execution queued..."
    elif execution.status == "running":
        if total_steps > 0:
            # 🔥 FIX: Better progress calculation
            progress = min(int((completed_steps / total_steps) * 90), 90)

            last_step = db.query(ExecutionStep).filter(
                ExecutionStep.execution_id == execution_id
            ).order_by(ExecutionStep.step_num.desc()).first()

            if last_step:
                current_step = last_step.description
                message = f"Executing Step {completed_steps + 1}/{total_steps}: {last_step.description[:50]}..."
            else:
                message = f"Processing steps... ({completed_steps}/{total_steps})"
        else:
            # 🔥 FIX: Show incremental progress based on time elapsed
            if execution.started_at:
                # elapsed_seconds = (datetime.utcnow() - execution.started_at).total_seconds()
                elapsed_seconds = (datetime.now() - execution.started_at).total_seconds()
                # Estimate: 60 seconds = 80% progress
                estimated_progress = min(int((elapsed_seconds / 60) * 80), 80)
                progress = max(10, estimated_progress)
                message = f"Parsing JIRA ticket and preparing test steps... ({int(elapsed_seconds)}s elapsed)"
            else:
                progress = 10
                message = "Parsing JIRA ticket and preparing test steps..."
    elif execution.status == "completed":
        progress = 100
        message = f"Test execution completed - {execution.overall_status}"
    elif execution.status == "failed":
        progress = 100
        message = execution.error_message or "Test execution failed"

    # Check if summary exists
    summary_available = False
    if execution.status == "completed" and execution.ticket_id:
        external_path = Path(settings.external_project_path)
        summary_path = external_path / "Reports" / "summaries" / f"summary_{execution.ticket_id}_latest.json"
        summary_available = summary_path.exists()
        logger.info(f"📊 Summary check for {execution.ticket_id}: {summary_available} (path: {summary_path})")

        # 🔥 ADD THIS: List what files ARE in summaries folder
        if not summary_available:
            summaries_folder = external_path / "Reports" / "summaries"
            if summaries_folder.exists():
                existing_files = list(summaries_folder.glob("*.json"))
                logger.warning(f"⚠️ Summary NOT found. Existing summaries: {[f.name for f in existing_files]}")

    return {
        "execution_id": execution.execution_id,
        "ticket_id": execution.ticket_id,  # 🔥 CRITICAL: Return ticket_id
        "status": execution.status,
        "progress": progress,
        "overall_status": execution.overall_status,
        "message": message,
        "current_step": current_step,
        "steps_completed": completed_steps,
        "steps_total": total_steps,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "report_path": execution.report_path,
        "script_path": execution.script_path,
        "video_path": execution.video_path,
        "summary_available": summary_available
    }


# ============================================================================
# DOWNLOAD ENDPOINTS
# ============================================================================

@app.get("/api/download-report/{execution_id}")
def download_html_report(execution_id: str, db: Session = Depends(get_db)):
    """
    Download the HTML report for a completed execution
    """
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if not execution.report_path:
        raise HTTPException(
            status_code=400,
            detail="Report not generated yet. Please wait for test completion."
        )

    report_path = Path(execution.report_path)

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found at: {execution.report_path}"
        )

    logger.info(f"📥 Serving report: {report_path.name}")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=f"{execution.ticket_id}_report.html"
    )
# from pathlib import Path
# import json

# ============================================================================
# SUMMARY JSON ENDPOINTS
# ============================================================================

@app.get("/api/summary/{ticket_id}")
def get_test_summary(ticket_id: str):
    """
    Get latest JSON summary for a ticket
    Returns lightweight summary data before downloading full report

    Example: GET /api/summary/RBPLCD-8001

    Returns:
        {
            "ticket_id": "RBPLCD-8001",
            "ticket_title": "Edit teststep measurement...",
            "summary": {
                "overall_status": "PASSED",
                "total_steps": 9,
                "passed": 8,
                "execution_time": "77.7s",
                "avg_confidence": 0.90
            },
            "agent_usage": {...},
            "insights": {...},
            "artifacts": {...}
        }
    """
    try:
        # Path to external TA_AI_Project
        external_path = Path(settings.external_project_path)
        summaries_folder = external_path / "Reports" / "summaries"

        if not summaries_folder.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Summaries folder not found. Please run a test first."
            )

        # Load latest summary
        latest_summary_path = summaries_folder / f"summary_{ticket_id}_latest.json"

        if not latest_summary_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No summary found for ticket '{ticket_id}'. Please run the test first."
            )

        with open(latest_summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        logger.info(f"📊 Served summary for {ticket_id}")

        return summary_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summary/{ticket_id}/{timestamp}")
def get_test_summary_by_timestamp(ticket_id: str, timestamp: str):
    """
    Get specific summary by timestamp

    Example: GET /api/summary/RBPLCD-8001/20250115_143000
    """
    try:
        external_path = Path(settings.external_project_path)
        summaries_folder = external_path / "Reports" / "summaries"

        summary_path = summaries_folder / f"summary_{ticket_id}_{timestamp}.json"

        if not summary_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Summary not found for {ticket_id} at {timestamp}"
            )

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        return summary_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summaries")
def list_all_summaries(
    limit: int = 50,
    status: str = None,
    module: str = None
):
    """
    List all available test summaries with filtering

    Query params:
        - limit: Number of results (default: 50)
        - status: Filter by status (PASSED/FAILED)
        - module: Filter by module

    Example: GET /api/summaries?limit=10&status=PASSED
    """
    try:
        external_path = Path(settings.external_project_path)
        summaries_folder = external_path / "Reports" / "summaries"

        if not summaries_folder.exists():
            return {
                "count": 0,
                "summaries": []
            }

        summaries = []

        # Find all summary files (excluding _latest.json)
        for json_file in summaries_folder.glob("summary_*.json"):
            if '_latest.json' in str(json_file):
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Apply filters
                if status and data['summary']['overall_status'] != status:
                    continue

                if module and data.get('module') != module:
                    continue

                summaries.append({
                    'ticket_id': data['ticket_id'],
                    'ticket_title': data['ticket_title'],
                    'module': data['module'],
                    'execution_date': data['execution_date'],
                    'overall_status': data['summary']['overall_status'],
                    'total_steps': data['summary']['total_steps'],
                    'passed': data['summary']['passed'],
                    'failed': data['summary']['failed'],
                    'execution_time': data['summary']['execution_time'],
                    'avg_confidence': data['summary']['avg_confidence'],
                    'status_emoji': data['insights']['status_emoji'],
                    'file_path': str(json_file)
                })

            except Exception as e:
                logger.warning(f"Could not read summary {json_file}: {e}")
                continue

        # Sort by execution date (newest first)
        summaries.sort(key=lambda x: x['execution_date'], reverse=True)

        # Apply limit
        summaries = summaries[:limit]

        return {
            "count": len(summaries),
            "summaries": summaries
        }

    except Exception as e:
        logger.error(f"Error listing summaries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summary-stats")
def get_summary_statistics():
    """
    Get overall statistics across all test executions

    Returns:
        {
            "total_executions": 25,
            "total_passed": 20,
            "total_failed": 5,
            "success_rate": 80.0,
            "avg_execution_time": 75.3,
            "avg_confidence": 0.87,
            "recent_executions": [...]
        }
    """
    try:
        external_path = Path(settings.external_project_path)
        summaries_folder = external_path / "Reports" / "summaries"

        if not summaries_folder.exists():
            return {
                "total_executions": 0,
                "total_passed": 0,
                "total_failed": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_confidence": 0.0,
                "recent_executions": []
            }

        summaries = []

        for json_file in summaries_folder.glob("summary_*.json"):
            if '_latest.json' in str(json_file):
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summaries.append(data)
            except:
                continue

        if not summaries:
            return {
                "total_executions": 0,
                "total_passed": 0,
                "total_failed": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_confidence": 0.0,
                "recent_executions": []
            }

        total_executions = len(summaries)
        total_passed = sum(1 for s in summaries if s['summary']['overall_status'] == 'PASSED')
        total_failed = sum(1 for s in summaries if s['summary']['overall_status'] == 'FAILED')

        # Calculate averages
        total_time = 0.0
        for s in summaries:
            try:
                time_str = s['summary']['execution_time'].replace('s', '')
                total_time += float(time_str)
            except:
                pass

        avg_time = total_time / total_executions if total_executions > 0 else 0.0
        avg_confidence = sum(s['summary'].get('avg_confidence', 0.0) for s in summaries) / total_executions

        # Get recent executions
        recent = sorted(summaries, key=lambda x: x['execution_date'], reverse=True)[:10]
        recent_list = [
            {
                'ticket_id': s['ticket_id'],
                'status': s['summary']['overall_status'],
                'execution_date': s['execution_date']
            }
            for s in recent
        ]

        return {
            "total_executions": total_executions,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": round((total_passed / total_executions * 100), 1),
            "avg_execution_time": round(avg_time, 1),
            "avg_confidence": round(avg_confidence, 2),
            "recent_executions": recent_list
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel
from fastapi import Query

class FeedbackProcessRequest(BaseModel):
    feedback: str

@app.post("/api/feedback")
async def receive_feedback(feedback: Feedback):
    """
    🔥 FIXED: Use SAME feedback processing pipeline as CLI + Force confidence to 1.0
    """
    from config_loader import load_config
    import sys
    
    logger.info(f"📝 [UI] Processing feedback for {feedback.ticket_id} Step {feedback.step_number}")
    
    # Load config and external project path
    config = load_config()
    external_project_path = Path(settings.external_project_path)
    
    # =====================================================
    # 1️⃣ LOAD SUMMARY (Source of truth for canonical step text)
    # =====================================================
    summaries_folder = external_project_path / "Reports" / "summaries"
    summary_path = summaries_folder / f"summary_{feedback.ticket_id}_latest.json"
    
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Summary not found for {feedback.ticket_id}. Run test first."
        )
    
    with open(summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 🔥 VERIFY: Summary has steps
    if not results.get("step_results") or len(results["step_results"]) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Summary has no steps. Run test first to generate valid summary."
        )
    
    # =====================================================
    # 2️⃣ BUILD FEEDBACK TEXT (Same format as CLI input)
    # =====================================================
    # 🔥 FIX: Normalize selector BEFORE building feedback text
    sys.path.append(str(external_project_path))
    from plcd_taseq import _normalize_css_selector
    
    normalized_selector = _normalize_css_selector(feedback.selector)
    
    logger.info(f"   📌 Raw selector: {feedback.selector}")
    logger.info(f"   📌 Normalized: {normalized_selector}")
    
    feedback_text = f"Step {feedback.step_number} is false positive, should use {normalized_selector}"
    
    logger.info(f"   📌 Feedback text: {feedback_text}")
    
    # =====================================================
    # 3️⃣ CALL CANONICAL FEEDBACK PROCESSOR
    # =====================================================
    from plcd_taseq import PLCDTestingAssistantSeq, collect_and_process_feedback
    
    assistant = PLCDTestingAssistantSeq(config)
    
    # 🔥 THIS WILL:
    # - Parse feedback text via LLM
    # - Normalize selector again (redundant but safe)
    # - Set confidence to 1.0
    # - Generate embedding
    # - Save to insights/pending/
    feedback_collected = collect_and_process_feedback(
        feedback.ticket_id,
        results,
        config,
        assistant,
        feedback_text=feedback_text
    )
    
    if feedback_collected:
        logger.info("✅ [UI] Feedback processed via canonical pipeline")
        
        # Verify feedback file was created
        pending_folder = external_project_path / "insights" / "pending"
        feedback_files = list(pending_folder.glob(
            f"{feedback.ticket_id}_step{feedback.step_number}_*.json"
        ))
        
        if feedback_files:
            # Return the actual saved feedback file content
            latest_file = max(feedback_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, "r", encoding="utf-8") as f:
                saved_feedback = json.load(f)
            
            # 🔥 ADD VERIFICATION LOGS
            logger.info(f"✅ Saved feedback file: {latest_file.name}")
            logger.info(f"   📌 Selector: {saved_feedback.get('selector')}")
            logger.info(f"   📌 Confidence: {saved_feedback.get('confidence')}")
            logger.info(f"   📌 Category: {saved_feedback.get('category')}")
            
            return {
                "status": "success",
                "file": latest_file.name,
                "feedback": saved_feedback  # Show what was actually saved
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Feedback processing failed - no file created"
            )
    else:
        raise HTTPException(
            status_code=500,
            detail="Feedback collection returned False"
        )


class RerunFeedbackRequest(BaseModel):
    ticket_id: str
    feedback_text: str = ""
    
@app.post("/api/rerun-with-feedback")
async def rerun_with_feedback(
    request: RerunFeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process feedback and rerun test WITH EXECUTION ID TRACKING
    
    Body:
        {
            "ticket_id": "RBPLCD-8835",
            "feedback_text": "Step 2 is false positive, should use [data-test='...']"
        }
    
    Returns:
        {
            "execution_id": "rerun_RBPLCD-8835_20260124_010000",
            "ticket_id": "RBPLCD-8835",
            "status": "pending"
        }
    """
    ticket_id = request.ticket_id
    feedback_text = request.feedback_text

    logger.info(f"📥 [RERUN] Request received for {ticket_id}")
    logger.info(f"   Feedback: {feedback_text[:100]}...")

    try:
        # =====================================================
        # 1️⃣ CREATE EXECUTION RECORD FIRST (DATABASE)
        # =====================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"rerun_{ticket_id}_{timestamp}"

        execution = TestExecution(
            execution_id=execution_id,
            ticket_id=ticket_id,
            project_id=None,  # No local project
            status="pending",
            overall_status="UNKNOWN",
            started_at=datetime.now()
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        logger.info(f"✅ Created execution: {execution_id}")

        # =====================================================
        # 2️⃣ ADD BACKGROUND TASK
        # =====================================================
        background_tasks.add_task(
            process_feedback_and_rerun,
            execution_id,
            ticket_id,
            feedback_text
        )

        logger.info(f"✅ Background task queued for {execution_id}")

        # =====================================================
        # 3️⃣ RETURN execution_id TO FRONTEND 🔥
        # =====================================================
        return {
            "execution_id": execution_id,  # 🔥 CRITICAL: Frontend needs this
            "ticket_id": ticket_id,
            "status": "pending",
            "message": f"Test rerun started with feedback processing"
        }

    except Exception as e:
        logger.error(f"❌ Error starting rerun: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER: Check if summary exists
# ============================================================================

@app.get("/api/summary-exists/{ticket_id}")
def check_summary_exists(ticket_id: str):
    """
    Quick check if summary exists for a ticket
    Useful for frontend to decide whether to show summary preview

    Returns:
        {
            "exists": true,
            "ticket_id": "RBPLCD-8001",
            "latest_execution": "2025-11-24 13:53:02"
        }
    """
    try:
        external_path = Path(settings.external_project_path)
        summaries_folder = external_path / "Reports" / "summaries"
        latest_summary_path = summaries_folder / f"summary_{ticket_id}_latest.json"

        if not latest_summary_path.exists():
            return {
                "exists": False,
                "ticket_id": ticket_id,
                "latest_execution": None
            }

        with open(latest_summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "exists": True,
            "ticket_id": ticket_id,
            "latest_execution": data.get('execution_date'),
            "overall_status": data['summary']['overall_status'],
            "has_report": data['artifacts']['has_report'],
            "has_video": data['artifacts']['has_video']
        }

    except Exception as e:
        logger.error(f"Error checking summary: {e}")
        return {
            "exists": False,
            "ticket_id": ticket_id,
            "latest_execution": None
        }

@app.get("/api/download-script/{execution_id}")
def download_playwright_script(execution_id: str, db: Session = Depends(get_db)):
    """
    Download the Playwright script for a completed execution
    """
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if not execution.script_path:
        raise HTTPException(
            status_code=400,
            detail="Script not generated yet. Please wait for test completion."
        )

    script_path = Path(execution.script_path)

    if not script_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Script file not found at: {execution.script_path}"
        )

    logger.info(f"📥 Serving script: {script_path.name}")

    return FileResponse(
        path=str(script_path),
        media_type="text/x-python",
        filename=f"{execution.ticket_id}_script.py"
    )


@app.get("/api/download-video/{execution_id}")
def download_test_video(execution_id: str, db: Session = Depends(get_db)):
    """
    Download the test execution video for a completed execution
    Supports HTTP Range requests for video streaming
    """
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    if not execution:
        logger.error(f"❌ Video: Execution {execution_id} not found")
        raise HTTPException(status_code=404, detail="Execution not found")

    if not execution.video_path:
        logger.error(f"❌ Video: No video_path for {execution_id}")
        raise HTTPException(
            status_code=400,
            detail="Video not generated yet. Please wait for test completion."
        )

    video_path = Path(execution.video_path)
    
    logger.info(f"🎬 Video request: {execution_id}")
    logger.info(f"   Path: {video_path}")
    logger.info(f"   Exists: {video_path.exists()}")
    
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {video_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found at: {execution.video_path}"
        )

    # Get file size
    file_size = video_path.stat().st_size
    logger.info(f"   Size: {file_size} bytes")
    logger.info(f"   Extension: {video_path.suffix}")

    # 🔥 FIX: Use streaming with proper headers
    return FileResponse(
        path=str(video_path),
        media_type="video/webm; codecs=\"vp8, vorbis\"",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Content-Length": str(file_size),
            "Content-Disposition": f"inline; filename=\"{execution.ticket_id}_video.webm\"",
            "X-Content-Type-Options": "nosniff"
        }
    )

# 🔥 NEW: DEBUG ENDPOINT FOR VIDEO ISSUES
@app.get("/api/debug/video/{execution_id}")
def debug_video(execution_id: str, db: Session = Depends(get_db)):
    """
    Debug video file for a completed execution
    """
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    result = {
        "execution_id": execution_id,
        "found": False,
        "video_path": None,
        "file_exists": False,
        "file_size": 0,
        "file_readable": False,
        "warnings": []
    }

    if not execution:
        result["warnings"].append("Execution not found in database")
        return result

    result["found"] = True
    result["video_path"] = execution.video_path

    if not execution.video_path:
        result["warnings"].append("No video_path set on execution record")
        return result

    video_path = Path(execution.video_path)
    result["file_exists"] = video_path.exists()

    if not video_path.exists():
        result["warnings"].append(f"Video file not found at: {video_path}")
        return result

    try:
        stat = video_path.stat()
        result["file_size"] = stat.st_size
        result["file_readable"] = os.access(video_path, os.R_OK)
        result["created_at"] = stat.st_ctime
        result["modified_at"] = stat.st_mtime
        
        if stat.st_size == 0:
            result["warnings"].append("⚠️ Video file is 0 bytes (empty)!")
        else:
            result["warnings"].append(f"✅ Video file is {stat.st_size} bytes")
            
    except Exception as e:
        result["warnings"].append(f"Error reading file stats: {e}")

    return result

# Add these debug endpoints to your main.py after the other endpoints

@app.get("/api/debug/execution-logs/{execution_id}")
def get_execution_logs(execution_id: str):
    """
    Get recent log entries for an execution (for debugging)
    """
    try:
        log_file = Path("logs") / "app.log"
        if not log_file.exists():
            return {"logs": "Log file not found", "log_path": str(log_file)}

        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find lines related to this execution
        relevant_lines = []
        for line in lines:
            if execution_id in line:
                relevant_lines.append(line.rstrip())

        # Get last 100 lines
        return {
            "execution_id": execution_id,
            "total_lines": len(relevant_lines),
            "logs": '\n'.join(relevant_lines[-100:])
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/artifacts/{ticket_id}")
def debug_artifacts(ticket_id: str):
    """
    Debug endpoint to check what artifact files exist for a ticket
    """
    try:
        from config import settings

        external_path = Path(settings.external_project_path)

        if not external_path.exists():
            return {"error": f"External project path not found: {external_path}"}

        # Search all artifact folders
        reports_folder = external_path / "Reports"
        scripts_folder = external_path / "Generated_Scripts"
        videos_folder = external_path / "Videos"

        def get_file_info(path: Path):
            """Get file info with timestamp"""
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

        # Find all files for this ticket
        reports = []
        if reports_folder.exists():
            reports = [
                get_file_info(p)
                for p in sorted(
                    reports_folder.glob(f"*{ticket_id}*.html"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
            ]

        scripts = []
        if scripts_folder.exists():
            scripts = [
                get_file_info(p)
                for p in sorted(
                    scripts_folder.glob(f"*{ticket_id}*.py"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
            ]

        videos = []
        if videos_folder.exists():
            # Get last 5 videos
            videos = [
                get_file_info(p)
                for p in sorted(
                    videos_folder.glob("*.webm"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )[:5]
            ]

        return {
            "ticket_id": ticket_id,
            "external_path": str(external_path),
            "reports": reports,
            "scripts": scripts,
            "recent_videos": videos,
            "current_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Debug artifacts error: {e}", exc_info=True)
        return {"error": str(e)}


execution_start_timestamp = time.time()

def process_feedback_and_rerun(
    execution_id: str,
    ticket_id: str,
    feedback_text: str
):
    """
    🔥 SINGLE SOURCE OF TRUTH
    UI == CLI feedback execution
    """
    logger.info(f"🔁 [RERUN] Starting feedback + rerun for {execution_id}")
    db = SessionLocal()

    try:
        # =====================================================
        # 0️⃣ MARK EXECUTION AS RUNNING
        # =====================================================
        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()

        if not execution:
            raise Exception(f"Execution {execution_id} not found")

        execution.status = "running"
        execution.started_at = datetime.now()
        db.commit()

        # =====================================================
        # 1️⃣ LOCATE EXTERNAL PROJECT
        # =====================================================
        external_project_path = Path(settings.external_project_path)
        python_exe = external_project_path / "venv" / "Scripts" / "python.exe"

        if not python_exe.exists():
            import shutil
            python_exe = shutil.which("python") or shutil.which("python3")

        if not python_exe:
            raise Exception("Python executable not found")

        # =====================================================
        # 2️⃣ RUN SUBPROCESS (FEEDBACK + RERUN)
        # =====================================================
        logger.info("🏃 Running CLI pipeline with feedback")

        cmd = [
            str(python_exe),
            "plcd_taseq.py",
            ticket_id,
            "--process-feedback",
            feedback_text
        ]

        result = subprocess.run(
            cmd,
            cwd=str(external_project_path),
            capture_output=True,
            text=True,
            timeout=600
        )

        logger.info(f"📤 Subprocess exit code: {result.returncode}")
        if result.stdout:
            logger.info(f"📝 STDOUT:\n{result.stdout[-2000:]}")
        if result.stderr:
            logger.warning(f"⚠️ STDERR:\n{result.stderr[-2000:]}")

        # 🔥 CRITICAL FIX: DON'T RAISE EXCEPTION ON NON-ZERO EXIT CODE
        # The subprocess might fail a step but still generate valid artifacts
        if result.returncode != 0:
            logger.warning(f"⚠️ Subprocess exited with code {result.returncode}, but continuing to load artifacts")

        # =====================================================
        # 3️⃣ LOAD STEPS JSON (SOURCE OF TRUTH)
        # =====================================================
        steps_file = external_project_path / "Reports" / "steps" / f"steps_{ticket_id}.json"

        if not steps_file.exists():
            # 🔥 FIX: Wait up to 5 seconds for the file to appear
            logger.info(f"⏳ Waiting for steps file: {steps_file}")
            for i in range(50):  # 50 * 0.1s = 5 seconds
                if steps_file.exists():
                    logger.info(f"✅ Steps file found after {i * 0.1}s")
                    break
                time.sleep(0.1)

        steps_saved = False
        if steps_file.exists():
            with open(steps_file, "r", encoding="utf-8") as f:
                steps = json.load(f)

            logger.info(f"✅ Loaded {len(steps)} steps from JSON")

            # Clear old steps
            db.query(ExecutionStep).filter(
                ExecutionStep.execution_id == execution_id
            ).delete()
            db.commit()

            for s in steps:
                db.add(ExecutionStep(
                    execution_id=execution_id,
                    step_num=s["step_number"],
                    step_text=s["step_text"],
                    status=s["status"],
                    selector_used=s.get("selector"),
                    agent_used=s.get("agent_used"),
                    confidence=s.get("confidence", 0.0),
                    action_type=s.get("action_type")
                ))

            db.commit()
            steps_saved = True
            logger.info(f"✅ {len(steps)} steps saved to DB")
        else:
            logger.error(f"❌ Steps file not found: {steps_file}")

        # =====================================================
        # 4️⃣ GENERATE SUMMARY (EVEN IF SUBPROCESS FAILED)
        # =====================================================
        if steps_saved:
            try:
                service = TestExecutionService(db)
                logger.info(f"📊 Generating summary for {execution_id}")
                service._generate_summary_from_db(execution_id, ticket_id)
                
                # Verify summary was created
                summary_path = external_project_path / "Reports" / "summaries" / f"summary_{ticket_id}_latest.json"
                if summary_path.exists():
                    logger.info(f"✅ Summary created: {summary_path}")
                else:
                    logger.error(f"❌ Summary not created at: {summary_path}")
            except Exception as summary_error:
                logger.error(f"❌ Summary generation failed: {summary_error}")
                import traceback
                logger.error(traceback.format_exc())

        # =====================================================
        # 5️⃣ FINALIZE EXECUTION
        # =====================================================
        # Find report
        reports = sorted(
            (external_project_path / "Reports").glob(f"*{ticket_id}*.html"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        report_path = str(reports[0]) if reports else None

        # Find video
        videos = sorted(
            (external_project_path / "Videos").glob("*.webm"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        video_path = str(videos[0]) if videos else None

        # Determine overall status
        overall_status = "FAILED" if any(
            s["status"] == "FAILED" for s in steps
        ) else "PASSED"

        execution.status = "completed"
        execution.overall_status = overall_status
        execution.completed_at = datetime.now()
        execution.report_path = report_path
        execution.video_path = video_path
        execution.error_message = None
        db.commit()

        logger.info(f"✅ RERUN COMPLETED for {execution_id}")
        logger.info(f"   Status: {overall_status}")
        logger.info(f"   Report: {report_path}")
        logger.info(f"   Summary: Reports/summaries/summary_{ticket_id}_latest.json")

    except Exception as e:
        logger.error(f"❌ RERUN FAILED: {e}", exc_info=True)

        # 🔥 TRY TO GENERATE SUMMARY EVEN ON ERROR
        try:
            steps_file = external_project_path / "Reports" / "steps" / f"steps_{ticket_id}.json"
            if steps_file.exists():
                logger.info("🔄 Attempting summary generation despite error...")
                service = TestExecutionService(db)
                service._generate_summary_from_db(execution_id, ticket_id)
        except:
            pass

        execution.status = "failed"
        execution.overall_status = "FAILED"
        execution.error_message = str(e)[:500]
        execution.completed_at = datetime.now()
        db.commit()

    finally:
        db.close()
        logger.info(f"🔒 Session closed for {execution_id}")

@app.get("/api/debug/parse-report/{execution_id}")
async def debug_parse_report(execution_id: str, db: Session = Depends(get_db)):
    """
    🔍 DEBUG ENDPOINT: Parse HTML report and show step extraction
    
    Example: GET http://localhost:8000/api/debug/parse-report/exec_RBPLCD-8960_20260111_003255
    
    Returns:
        {
            "execution_id": "exec_RBPLCD-8960_20260111_003255",
            "report_path": "C:\\path\\to\\report.html",
            "total_rows": 10,
            "rows_parsed": [
                {
                    "row_index": 0,
                    "is_header": true,
                    "num_cells": 6,
                    "cells": ["Step #", "Description", "Status", "Selector", "Agent", "Confidence"],
                    "raw_html": "<tr><th>Step #</th>..."
                },
                {
                    "row_index": 1,
                    "is_header": false,
                    "num_cells": 6,
                    "cells": ["1", "Click login button", "PASSED", "#login", "L1", "0.95"],
                    "raw_html": "<tr><td>1</td>..."
                }
            ],
            "table_preview": "<thead><tr>..."
        }
    """
    try:
        # Find execution
        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()
        
        if not execution:
            return {
                "error": f"Execution {execution_id} not found",
                "available_executions": [
                    e.execution_id for e in db.query(TestExecution).order_by(
                        TestExecution.started_at.desc()
                    ).limit(10).all()
                ]
            }
        
        if not execution.report_path or not Path(execution.report_path).exists():
            return {
                "error": f"Report not found",
                "report_path": execution.report_path,
                "execution_status": execution.status,
                "overall_status": execution.overall_status
            }
        
        logger.info(f"DEBUG: Parsing report for {execution_id}")
        logger.info(f"   Report path: {execution.report_path}")
        
        # Read HTML
        with open(execution.report_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        logger.info(f"   HTML size: {len(html)} bytes")
        
        # Extract table
        table_match = re.search(r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE)
        
        if not table_match:
            return {
                "error": "No <table> found in HTML",
                "html_preview": html[:1000],
                "report_path": execution.report_path,
                "html_size": len(html),
                "has_table_tag": "<table" in html.lower()
            }
        
        table_html = table_match.group(1)
        
        # Extract rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        
        logger.info(f"   Found {len(rows)} rows in table")
        
        debug_info = {
            "execution_id": execution_id,
            "report_path": execution.report_path,
            "total_rows": len(rows),
            "html_size": len(html),
            "table_size": len(table_html),
            "rows_parsed": []
        }
        
        # Parse each row
        for idx, row in enumerate(rows):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            
            # Also check for <th> tags (header row)
            if idx == 0 and len(cells) == 0:
                cells = re.findall(r'<th[^>]*>(.*?)</th>', row, re.DOTALL | re.IGNORECASE)
            
            # Clean HTML tags from cells
            clean_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
            
            debug_info["rows_parsed"].append({
                "row_index": idx,
                "is_header": idx == 0,
                "num_cells": len(cells),
                "cells": clean_cells[:10],  # Limit to first 10 cells
                "raw_html": row[:200]  # First 200 chars of raw HTML
            })
        
        # Show first few rows of actual HTML table for inspection
        debug_info["table_preview"] = table_html[:1000]
        
        # Add recommendations based on what we found
        if len(rows) > 0:
            first_data_row = debug_info["rows_parsed"][1] if len(debug_info["rows_parsed"]) > 1 else None
            if first_data_row:
                num_cells = first_data_row["num_cells"]
                debug_info["recommendations"] = {
                    "cells_per_row": num_cells,
                    "suggested_mapping": _suggest_cell_mapping(num_cells, first_data_row["cells"])
                }
        
        return debug_info
    
    except Exception as e:
        import traceback
        logger.error(f"❌ DEBUG endpoint error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _suggest_cell_mapping(num_cells: int, sample_cells: list) -> dict:
    """
    Helper function to suggest cell mapping based on content
    """
    mapping = {}
    
    if num_cells >= 3:
        mapping["step_num_index"] = 0
        mapping["step_text_index"] = 1
        mapping["status_index"] = 2
    
    if num_cells >= 4:
        mapping["selector_index"] = 3
    
    if num_cells >= 5:
        mapping["agent_index"] = 4
    
    if num_cells >= 6:
        mapping["confidence_index"] = 5
    
    mapping["example"] = f"""
    # Example parsing code for {num_cells} columns:
    step_num = int(re.sub(r'<[^>]+>', '', cells[0]).strip())
    step_text = re.sub(r'<[^>]+>', '', cells[1]).strip()[:200]
    status = re.sub(r'<[^>]+>', '', cells[2]).strip().upper()
    """
    
    if num_cells >= 4:
        mapping["example"] += """
    selector = re.sub(r'<[^>]+>', '', cells[3]).strip()
    agent = re.sub(r'<[^>]+>', '', cells[4]).strip() if len(cells) > 4 else ""
    confidence = float(re.sub(r'<[^>]+>', '', cells[5]).strip()) if len(cells) > 5 else 0.0
    """
    
    return mapping

# ...existing code...





@app.get("/api/debug/steps/{execution_id}")
def debug_steps_count(execution_id: str, db: Session = Depends(get_db)):
    count = db.query(ExecutionStep).filter(ExecutionStep.execution_id == execution_id).count()
    return {"execution_id": execution_id, "steps_count": count}


# Add this to your main.py after the other debug endpoints
def test_python_executable():
    """
    Test if Python executable can be found and works
    """
    try:
        from config import settings
        import shutil

        external_path = Path(settings.external_project_path)

        results = {
            "external_path": str(external_path),
            "external_path_exists": external_path.exists(),
            "python_checks": []
        }

        # Check venv paths
        venv_paths = [
            external_path / "venv" / "Scripts" / "python.exe",
            external_path / "venv" / "bin" / "python",
        ]

        for venv_path in venv_paths:
            check = {
                "path": str(venv_path),
                "exists": venv_path.exists(),
                "version": None,
                "works": False
            }

            if venv_path.exists():
                try:
                    result = subprocess.run(
                        [str(venv_path), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    check["version"] = result.stdout.strip()
                    check["works"] = result.returncode == 0
                except Exception as e:
                    check["error"] = str(e)

            results["python_checks"].append(check)

        # Check system Python
        for cmd in ["python", "python3", "py"]:
            python_path = shutil.which(cmd)
            if python_path:
                check = {
                    "path": python_path,
                    "exists": True,
                    "command": cmd,
                    "version": None,
                    "works": False
                }

                try:
                    result = subprocess.run(
                        [python_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    check["version"] = result.stdout.strip()
                    check["works"] = result.returncode == 0
                except Exception as e:
                    check["error"] = str(e)

                results["python_checks"].append(check)

        return results

    except Exception as e:
        logger.error(f"Test Python error: {e}", exc_info=True)
        return {"error": str(e)}
# ============================================================================
# TICKET MANAGEMENT (MINIMAL)
# ============================================================================

@app.get("/api/tickets/{ticket_id}")
def get_ticket_details(ticket_id: str, db: Session = Depends(get_db)):
    """
    Get ticket details by ticket_id
    """
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket '{ticket_id}' not found. Please upload the ticket file to Jira_Tickets/ folder."
        )

    # Parse steps from file if available
    steps = []
    if ticket.file_path and Path(ticket.file_path).exists():
        try:
            with open(ticket.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple step extraction (you can improve this)
                lines = content.split('\n')
                step_num = 1
                for line in lines:
                    if re.match(r'^\s*(Step\s*\d+\.?|\d+\.)', line.strip()):
                        steps.append({
                            'num': step_num,
                            'text': line.strip()
                        })
                        step_num += 1
        except Exception as e:
            logger.warning(f"Could not parse steps from ticket file: {e}")

    return {
        "id": ticket.id,
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        "module": ticket.module,
        "project_id": ticket.project_id,
        "file_path": ticket.file_path,
        "steps": steps,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None
    }

class UpdateTitleRequest(BaseModel):
    session_id: str
    new_title: str

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    date: str
    messages: List[dict]

@app.put("/api/chat-session/title")
def update_chat_title(
    request: UpdateTitleRequest,
    db: Session = Depends(get_db)
):
    """
    Update chat session title

    Body:
        {
            "session_id": "1702888800000",
            "new_title": "Updated title here"
        }

    Returns:
        {
            "session_id": "1702888800000",
            "title": "Updated title here",
            "updated_at": "2025-12-15T10:30:00",
            "message": "Title updated successfully"
        }
    """
    try:
        logger.info(f"📝 Updating title for session: {request.session_id}")

        # Here you would update your database
        # For now, we'll just validate and return success
        # You'll need to add a ChatSession table to your database

        # TODO: Add actual database update
        # session = db.query(ChatSession).filter(
        #     ChatSession.id == request.session_id
        # ).first()
        #
        # if not session:
        #     raise HTTPException(status_code=404, detail="Session not found")
        #
        # session.title = request.new_title
        # session.updated_at = datetime.now()
        # db.commit()

        return {
            "session_id": request.session_id,
            "title": request.new_title,
            "updated_at": datetime.now().isoformat(),
            "message": "Title updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating title: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat-sessions")
def get_chat_sessions(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get all chat sessions for current user

    Query params:
        - limit: Number of sessions to return (default: 50)

    Returns:
        {
            "count": 10,
            "sessions": [
                {
                    "id": "1702888800000",
                    "title": "Chat title",
                    "date": "2025-12-15T10:00:00",
                    "messages": [...]
                }
            ]
        }
    """
    try:
        logger.info(f"📋 Fetching chat sessions (limit: {limit})")


        return {
            "count": 0,
            "sessions": []
        }

    except Exception as e:
        logger.error(f"❌ Error fetching sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat-session/{session_id}")
def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a chat session

    Path params:
        - session_id: ID of session to delete

    Returns:
        {
            "message": "Session deleted successfully",
            "session_id": "1702888800000"
        }
    """
    try:
        logger.info(f"🗑️ Deleting session: {session_id}")

        # TODO: Add actual database delete
        # session = db.query(ChatSession).filter(
        #     ChatSession.id == session_id
        # ).first()
        #
        # if not session:
        #     raise HTTPException(status_code=404, detail="Session not found")
        #
        # db.delete(session)
        # db.commit()

        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# BACKGROUND TASK
# ============================================================================

"""
REPLACE the execute_test_in_background function in your main.py with this version
This ensures proper completion and error handling
"""

# Add this BEFORE execute_test_in_background function
def _mark_execution_as_failed(
    db: Session, 
    execution_id: str, 
    error_message: str,
    service: TestExecutionService,
    ticket_id: str
):
    """
    Helper function to mark execution as failed and generate summary
    """
    try:
        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()

        if execution:
            execution.status = "failed"
            execution.overall_status = "FAILED"
            execution.completed_at = datetime.now()
            execution.error_message = error_message[:500]
            db.commit()
            logger.info(f"✅ Execution {execution_id} marked as 'failed'")
            
            # Try to generate summary for failed execution
            try:
                service._generate_summary_from_db(execution_id, ticket_id)
                logger.info("✅ Generated summary for failed execution")
            except Exception as e:
                logger.warning(f"⚠️ Could not generate summary for failed execution: {e}")

    except Exception as e:
        logger.error(f"❌ Failed to mark execution as failed: {e}")
        db.rollback()


class StepData(BaseModel):
    execution_id: str
    steps: list

@app.post("/api/save-steps")
def save_steps(
    data: StepData,
    db: Session = Depends(get_db)
):
    """
    Save execution steps from external runner.
    """
    execution_id = data.execution_id
    steps = data.steps
    
    # 🔥 FIX: Ensure execution record exists before inserting steps
    execution = db.query(TestExecution).filter(
        TestExecution.execution_id == execution_id
    ).first()

    if not execution:
        # Create execution record if it doesn't exist
        logger.warning(f"Execution {execution_id} not found, creating it now")
        
        # Extract ticket_id from execution_id (format: exec_RBPLCD-8961_20260126_143805)
        import re
        match = re.search(r'exec_([A-Z]+-\d+)_', execution_id)
        ticket_id = match.group(1) if match else "UNKNOWN"
        
        execution = TestExecution(
            execution_id=execution_id,
            ticket_id=ticket_id,
            project_id=None,
            status="running",
            overall_status="UNKNOWN",
            started_at=datetime.now()
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        logger.info(f"✅ Created missing execution record: {execution_id}")

    # Delete old steps
    deleted = db.query(ExecutionStep).filter(
        ExecutionStep.execution_id == execution_id
    ).delete()
    db.commit()

    # Delete old steps
    # db.query(ExecutionStep).filter(ExecutionStep.execution_id == execution_id).delete()
    # db.commit()

    saved_count = 0
    for step in steps:
        db.add(ExecutionStep(
            execution_id=execution_id,
            step_num=step.get("step_num"),
            step_text=step.get("step_text"),
            status=step.get("status"),
            selector_used=step.get("selector_used", ""),
            agent_used=step.get("agent_used", ""),
            confidence=step.get("confidence", 0.0),
            action_type=step.get("action_type", ""),
            # screenshot_path=step.get("screenshot_path")
        ))
        saved_count += 1
    db.commit()
    return {"saved": saved_count}

logger.info("execute_test_in_background CALLED")

def execute_test_in_background(
    execution_id: str,
    ticket_id: str,
    project_id: Optional[int]
):
    """
    Background task - FIXED VERSION WITH PROPER EXECUTION ID HANDLING
    """
    db = SessionLocal()
    service = TestExecutionService(db)
    final_status_set = False

    # 🔥 LOG THE EXECUTION ID AT THE VERY START
    logger.info("="*70)
    logger.info(f"🚀 BACKGROUND TASK STARTED")
    logger.info(f"   🆔 Execution ID: {execution_id}")
    logger.info(f"   🎫 Ticket ID: {ticket_id}")
    logger.info("="*70)

    try:
        # Step 1: Mark as running
        execution = db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()
        
        if not execution:
            logger.error(f"❌ Execution {execution_id} not found in database!")
            return
            
        execution.status = "running"
        execution.started_at = datetime.now()
        db.commit()
        logger.info(f"✅ Status updated to 'running' for {execution_id}")

        # Step 2: Validate paths
        external_project_path = Path(settings.external_project_path)
        plcd_script = external_project_path / "plcd_taseq.py"
        
        if not plcd_script.exists():
            raise FileNotFoundError(f"plcd_taseq.py not found: {plcd_script}")

        # Step 3: Find Python executable
        python_exe = str(external_project_path / "venv" / "Scripts" / "python.exe")
        if not Path(python_exe).exists():
            import shutil
            python_exe = shutil.which("python") or shutil.which("python3")

        # Step 4: Execute test with --no-feedback
        logger.info(f"🏃 Executing: {ticket_id} --no-feedback for {execution_id}")
        
        result = subprocess.run(
            [python_exe, str(plcd_script), ticket_id, "--no-feedback", "--visible"],
            cwd=str(external_project_path),
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # 🔥 LOG SUBPROCESS OUTPUT (even if it failed)
        logger.info(f"📤 Subprocess return code: {result.returncode}")
        if result.stdout:
            logger.info(f"📝 STDOUT:\n{result.stdout[-2000:]}")  # Last 2000 chars
        if result.stderr:
            logger.error(f"❌ STDERR:\n{result.stderr[-2000:]}")

        # 🔥 CRITICAL FIX: Load steps from JSON even if subprocess failed
        steps_file = external_project_path / "Reports" / "steps" / f"steps_{ticket_id}.json"
        steps_saved = False

        if steps_file.exists():
            with open(steps_file, "r", encoding="utf-8") as f:
                raw_steps = json.load(f)

            logger.info(f"✅ Loaded {len(raw_steps)} steps from JSON (subprocess exit code: {result.returncode})")

            # 🔥 NORMALIZE STEP FORMAT FOR DB
            normalized_steps = []
            for step in raw_steps:
                normalized_steps.append({
                    "step_number": step.get("step_number"),
                    "step_text": step.get("step_text"),
                    "selector": step.get("selector"),
                    "status": step.get("status"),
                    "confidence": step.get("confidence", 0.0),
                    "agent_used": step.get("agent_used"),
                    "action_type": step.get("action_type"),
                })

            # 🔥 DELETE OLD STEPS FOR THIS EXECUTION
            deleted = db.query(ExecutionStep).filter(
                ExecutionStep.execution_id == execution_id
            ).delete()
            db.commit()
            logger.info(f"🧹 Deleted {deleted} old steps for {execution_id}")

            # Save steps to DB
            service._save_steps_to_db(execution_id, normalized_steps)
            steps_saved = True
            logger.info(f"✅ Saved {len(normalized_steps)} steps to DB for {execution_id}")
        else:
            logger.error(f"❌ Steps JSON not found: {steps_file}")

        # 🔥 ALWAYS GENERATE SUMMARY (even if subprocess failed)
        if steps_saved:
            try:
                logger.info(f"📊 Generating summary for {execution_id}")
                service._generate_summary_from_db(execution_id, ticket_id)
                
                # Verify summary was created
                summary_path = external_project_path / "Reports" / "summaries" / f"summary_{ticket_id}_latest.json"
                if summary_path.exists():
                    logger.info(f"✅ Summary successfully created at: {summary_path}")
                else:
                    logger.error(f"❌ Summary file NOT created at: {summary_path}")
            except Exception as summary_error:
                logger.error(f"❌ Summary generation failed: {summary_error}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"⚠️ No steps to generate summary from")

        # Step 5: Find artifacts
        report_path = None
        script_path = None
        video_path = None
        overall_status = "UNKNOWN"

        reports_folder = external_project_path / "Reports"
        scripts_folder = external_project_path / "Generated_Scripts"
        videos_folder = external_project_path / "Videos"
        
        if reports_folder.exists():
            reports = sorted(
                reports_folder.glob(f"*{ticket_id}*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if reports:
                report_path = str(reports[0])
                logger.info(f"📄 Found report: {reports[0].name}")

        # 🔥 NEW: Find script for this ticket
        if scripts_folder.exists():
            scripts = sorted(
                scripts_folder.glob(f"*{ticket_id}*.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if scripts:
                script_path = str(scripts[0])
                logger.info(f"📜 Found script: {scripts[0].name}")

        # 🔥 NEW: Find latest video
        if videos_folder.exists():
            videos = sorted(
                videos_folder.glob("*.webm"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if videos:
                video_path = str(videos[0])
                logger.info(f"🎥 Found video: {videos[0].name}")

        # Parse overall status from report
        if report_path and Path(report_path).exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                
                match = re.search(r'<div[^>]*class=["\'][^"\']*status-(PASSED|FAILED)[^"\']*["\'][^>]*>\s*(PASSED|FAILED)\s*</div>', html, re.IGNORECASE)
                if match:
                    overall_status = match.group(2).upper()
                    logger.info(f"✅ Parsed status: {overall_status}")
            except Exception as e:
                logger.warning(f"Could not parse status: {e}")

        # Step 6: Update execution record
        final_status = "completed" if steps_saved else "failed"
        db.refresh(execution)
        execution.status = final_status
        execution.overall_status = overall_status if steps_saved else "FAILED"
        execution.completed_at = datetime.now()
        execution.report_path = report_path
        execution.script_path = script_path
        execution.video_path = video_path
        execution.error_message = None if steps_saved else f"Subprocess exit code: {result.returncode}"
        db.commit()
        final_status_set = True

        logger.info(f"✅ Execution {execution_id} marked as '{final_status}'")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ EXECUTION FAILED for {execution_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 🔥 TRY TO GENERATE SUMMARY EVEN ON ERROR
        try:
            steps_file = external_project_path / "Reports" / "steps" / f"steps_{ticket_id}.json"
            if steps_file.exists():
                logger.info("🔄 Attempting summary generation despite error...")
                service._generate_summary_from_db(execution_id, ticket_id)
        except:
            pass
        
        _mark_execution_as_failed(db, execution_id, str(e)[:500], service, ticket_id)
        final_status_set = True

    finally:
        if not final_status_set:
            logger.warning(f"⚠️ Forcing {execution_id} to failed state")
            execution = db.query(TestExecution).filter(
                TestExecution.execution_id == execution_id
            ).first()
            if execution and execution.status == "running":
                execution.status = "failed"
                execution.completed_at = datetime.now()
                db.commit()
        db.close()
        logger.info(f"🔒 Session closed: {execution_id}\n")
        
@app.get("/api/execution-summary/{ticket_id}")
async def get_execution_summary(ticket_id: str):
    """Get test execution summary (ONLY from plcd_taseq.py summary)"""
    try:
        summary_path = Path(settings.external_project_path) / "Reports" / "summaries" / f"summary_{ticket_id}_latest.json"
        
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Summary not found")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        return summary
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Summary file not found")
    except Exception as e:
        logger.error(f"Failed to load summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
