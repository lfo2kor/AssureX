"""
Service layer for test execution - ROBUST VERSION
Handles partial test completion and always provides available artifacts
"""
import logging
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.orm import Session

from models import TestExecution, ExecutionStep


class TestExecutionService:
    """Service to handle test execution workflow"""
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger("TestExecutionService")
        
    def create_execution_record(self, ticket_id: str, project_id: int) -> TestExecution:
        """Create a new test execution record"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"exec_{ticket_id}_{timestamp}"
        
        execution = TestExecution(
            execution_id=execution_id,
            ticket_id=ticket_id,
            project_id=project_id,
            status="pending",
            started_at=datetime.now()
        )
        
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        self.logger.info(f"✅ Created execution record: {execution_id}")
        return execution
    
    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        overall_status: Optional[str] = None,
        error_message: Optional[str] = None,
        report_path: Optional[str] = None,
        script_path: Optional[str] = None,
        video_path: Optional[str] = None
    ):
        """Update execution status"""
        execution = self.db.query(TestExecution).filter(
            TestExecution.execution_id == execution_id
        ).first()
        
        if not execution:
            self.logger.error(f"❌ Execution not found: {execution_id}")
            return
        
        execution.status = status
        
        if overall_status:
            execution.overall_status = overall_status
        
        if error_message:
            execution.error_message = error_message
            
        if report_path:
            execution.report_path = report_path
            self.logger.info(f"   📄 Report: {report_path}")
            
        if script_path:
            execution.script_path = script_path
            self.logger.info(f"   📜 Script: {script_path}")
            
        if video_path:
            execution.video_path = video_path
            self.logger.info(f"   🎥 Video: {video_path}")
        
        if status in ["completed", "failed"]:
            execution.completed_at = datetime.now()
        
        self.db.commit()
        self.logger.info(f"✅ Updated execution {execution_id}: status={status}")
    
    def run_test_workflow(
        self,
        ticket_id: str,
        project_id: int,
        execution_id: str,
        external_project_path: str
    ) -> Dict:
        """
        Run test workflow - ROBUST VERSION
        Always returns state with available artifacts, even on failure
        """
        self.logger.info(f"🚀 Starting test workflow for {ticket_id}")
        
        ext_path = Path(external_project_path).resolve()
        
        # Record start time to filter old files
        execution_start_time = datetime.now()
        
        # Initialize state with defaults
        state = {
            'ticket_id': ticket_id,
            'execution_id': execution_id,
            'overall_status': 'UNKNOWN',
            'report_path': None,
            'script_path': None,
            'video_path': None,
            'step_results': [],
            'error_message': None
        }
        
        try:
            # Validate paths
            if not ext_path.exists():
                raise FileNotFoundError(f"External project not found: {ext_path}")
            
            script_path = ext_path / "plcd_taseq.py"
            if not script_path.exists():
                raise FileNotFoundError(f"plcd_taseq.py not found at {script_path}")
            
            # Get Python executable
            python_exe = self._get_python_executable(ext_path)
            self.logger.info(f"🐍 Using Python: {python_exe}")
            
            # Build command
            cmd = [str(python_exe), str(script_path), ticket_id]
            
            self.logger.info(f"🏃 Command: {' '.join(cmd)}")
            self.logger.info(f"📂 Working dir: {ext_path}")
            self.logger.info("⏳ Starting test execution (may take several minutes)...")
            
            # Execute with timeout
            process = subprocess.Popen(
                cmd,
                cwd=str(ext_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            try:
                stdout, stderr = process.communicate(timeout=600)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                self.logger.error("❌ Test execution timed out (10 minutes)")
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                state['error_message'] = "Test execution timed out after 10 minutes"
            
            self.logger.info(f"✅ Process completed with return code: {return_code}")
            
            # ALWAYS search for generated artifacts, regardless of return code
            self.logger.info("🔍 Searching for generated artifacts...")
            
            # Only find files created AFTER execution started
            artifacts = self._find_artifacts(ext_path, ticket_id, execution_start_time)
            
            state['report_path'] = artifacts['report']
            state['script_path'] = artifacts['script']
            state['video_path'] = artifacts['video']
            
            self.logger.info(f"   📄 Report: {'Found' if artifacts['report'] else 'Not found'}")
            self.logger.info(f"   📜 Script: {'Found' if artifacts['script'] else 'Not found'}")
            self.logger.info(f"   🎥 Video: {'Found' if artifacts['video'] else 'Not found'}")
            
            # Determine overall status from output
            state['overall_status'] = self._parse_status(stdout, return_code, artifacts)
            
            # If report exists, parse it for the TRUE status (most reliable)
            if artifacts['report']:
                report_status = self._parse_report_status(artifacts['report'])
                if report_status and report_status != 'UNKNOWN':
                    self.logger.info(f"   📄 Report status: {report_status} (overriding stdout parsing)")
                    state['overall_status'] = report_status
            
            # Parse step results
            step_results = self._parse_steps(stdout)
            state['step_results'] = step_results
            
            # Save steps to database
            self._save_steps_to_db(execution_id, step_results)
            
            # If return code is non-zero and no artifacts, extract error
            if return_code != 0 and not artifacts['report']:
                self.logger.error("❌ Test failed without generating report")
                state['error_message'] = self._extract_error(stderr, stdout)
            
            # Log execution summary
            self._log_summary(state, stdout, stderr)
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Test workflow exception: {e}", exc_info=True)
            
            # Even on exception, try to find any generated artifacts
            try:
                artifacts = self._find_artifacts(ext_path, ticket_id, execution_start_time)
                state['report_path'] = artifacts['report']
                state['script_path'] = artifacts['script']
                state['video_path'] = artifacts['video']
                
                if artifacts['report']:
                    self.logger.info("✅ Found report despite exception")
                    state['overall_status'] = 'FAILED'
            except:
                pass
            
            state['error_message'] = str(e)[:500]
            return state
    
    def _find_artifacts(self, ext_path: Path, ticket_id: str, execution_start_time: datetime) -> Dict[str, Optional[str]]:
        """Search for generated artifact files created AFTER execution started"""
        artifacts = {
            'report': None,
            'script': None,
            'video': None
        }
        
        # Convert execution start time to timestamp for comparison
        start_timestamp = execution_start_time.timestamp()
        
        self.logger.info(f"   Looking for files created after {execution_start_time.strftime('%H:%M:%S')}")
        
        # Search Reports folder
        reports_folder = ext_path / "Reports"
        if reports_folder.exists():
            report_files = [
                p for p in reports_folder.glob(f"*{ticket_id}*.html")
                if p.stat().st_mtime >= start_timestamp
            ]
            report_files = sorted(report_files, key=lambda p: p.stat().st_mtime, reverse=True)
            
            if report_files:
                artifacts['report'] = str(report_files[0])
                file_time = datetime.fromtimestamp(report_files[0].stat().st_mtime)
                self.logger.info(f"   ✅ Found report: {report_files[0].name} (created {file_time.strftime('%H:%M:%S')})")
            else:
                self.logger.warning(f"   ⚠️  No NEW report found for {ticket_id}")
                # Check if old reports exist
                old_reports = list(reports_folder.glob(f"*{ticket_id}*.html"))
                if old_reports:
                    latest_old = max(old_reports, key=lambda p: p.stat().st_mtime)
                    old_time = datetime.fromtimestamp(latest_old.stat().st_mtime)
                    self.logger.warning(f"      Found OLD report from {old_time.strftime('%H:%M:%S')} (before execution)")
        
        # Search Scripts folder (multiple possible locations)
        script_folders = [
            ext_path / "Generated_Scripts",
            ext_path / "Scripts",
            ext_path / "scripts"
        ]
        
        for scripts_folder in script_folders:
            if scripts_folder.exists():
                script_files = [
                    p for p in scripts_folder.glob(f"*{ticket_id}*.py")
                    if p.stat().st_mtime >= start_timestamp
                ]
                script_files = sorted(script_files, key=lambda p: p.stat().st_mtime, reverse=True)
                
                if script_files:
                    artifacts['script'] = str(script_files[0])
                    file_time = datetime.fromtimestamp(script_files[0].stat().st_mtime)
                    self.logger.info(f"   ✅ Found script: {script_files[0].name} (created {file_time.strftime('%H:%M:%S')})")
                    break
        
        # Search Videos folder
        videos_folder = ext_path / "Videos"
        if videos_folder.exists():
            video_files = [
                p for p in videos_folder.glob("*.webm")
                if p.stat().st_mtime >= start_timestamp
            ]
            video_files = sorted(video_files, key=lambda p: p.stat().st_mtime, reverse=True)
            
            if video_files:
                artifacts['video'] = str(video_files[0])
                file_time = datetime.fromtimestamp(video_files[0].stat().st_mtime)
                self.logger.info(f"   ✅ Found video: {video_files[0].name} (created {file_time.strftime('%H:%M:%S')})")
        
        return artifacts
    
    def _parse_status(self, stdout: str, return_code: int, artifacts: Dict) -> str:
        """Determine overall test status"""
        self.logger.info("🔍 Parsing overall status...")
        
        # Check for explicit status patterns in output (multiple formats)
        status_patterns = [
            ("PASSED", [
                "Overall Status: PASSED",
                "Overall Status:  PASSED",
                "Status: PASSED",
                "[OK] Summary:",
                "passed_steps",
                "✅ TEST EXECUTION COMPLETED",
            ]),
            ("FAILED", [
                "Overall Status: FAILED",
                "Overall Status:  FAILED", 
                "Status: FAILED",
                "failed_steps",
                "❌ TEST EXECUTION FAILED",
            ])
        ]
        
        # Count pattern matches
        passed_matches = 0
        failed_matches = 0
        
        for pattern in status_patterns[0][1]:  # PASSED patterns
            if pattern in stdout:
                passed_matches += 1
                self.logger.info(f"   Found PASSED indicator: '{pattern}'")
        
        for pattern in status_patterns[1][1]:  # FAILED patterns
            if pattern in stdout:
                failed_matches += 1
                self.logger.info(f"   Found FAILED indicator: '{pattern}'")
        
        # Determine status based on matches
        if passed_matches > failed_matches:
            self.logger.info(f"   ✅ Status: PASSED (based on {passed_matches} indicators)")
            return "PASSED"
        elif failed_matches > passed_matches:
            self.logger.info(f"   ❌ Status: FAILED (based on {failed_matches} indicators)")
            return "FAILED"
        
        # Fallback: Check return code if we have artifacts
        if artifacts['report']:
            # If report was generated and return code is 0, likely passed
            if return_code == 0:
                self.logger.info(f"   ✅ Status: PASSED (return code 0 + report exists)")
                return "PASSED"
            else:
                self.logger.info(f"   ⚠️ Status: FAILED (return code {return_code} + report exists)")
                return "FAILED"
        
        # No report and non-zero return - complete failure
        if return_code != 0:
            self.logger.info(f"   ❌ Status: FAILED (return code {return_code}, no report)")
            return "FAILED"
        
        self.logger.warning(f"   ⚠️ Status: UNKNOWN (no clear indicators)")
        return "UNKNOWN"
    
    def _parse_report_status(self, report_path: str) -> Optional[str]:
        """
        Parse HTML report to extract actual test status
        This is the most reliable source since it's generated from test results
        """
        try:
            self.logger.info(f"   📄 Parsing report for status: {Path(report_path).name}")
            
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Look for status indicators in HTML
            # Pattern 1: <h2>Status: PASSED</h2> or similar
            import re
            
            # Try different patterns
            patterns = [
                r'Status:\s*(PASSED|FAILED)',
                r'Overall Status:\s*(PASSED|FAILED)',
                r'class=["\']status["\']>\s*(PASSED|FAILED)',
                r'<h2[^>]*>\s*Status:\s*(PASSED|FAILED)',
                r'test-status["\']>\s*(PASSED|FAILED)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    status = match.group(1).upper()
                    self.logger.info(f"   ✅ Found status in report: {status}")
                    return status
            
            # Fallback: Count passed vs failed steps
            passed_count = len(re.findall(r'status["\']:\s*["\']PASSED["\']|step-passed|✅|PASSED', html_content, re.IGNORECASE))
            failed_count = len(re.findall(r'status["\']:\s*["\']FAILED["\']|step-failed|❌|FAILED', html_content, re.IGNORECASE))
            
            self.logger.info(f"   Step counts in report: {passed_count} passed, {failed_count} failed")
            
            if passed_count > 0 and failed_count == 0:
                self.logger.info(f"   ✅ All steps passed in report")
                return "PASSED"
            elif failed_count > 0:
                self.logger.info(f"   ❌ Some steps failed in report")
                return "FAILED"
            
            self.logger.warning(f"   ⚠️ Could not determine status from report")
            return None
            
        except Exception as e:
            self.logger.error(f"   ❌ Error parsing report: {e}")
            return None
    
    def _parse_steps(self, stdout: str) -> list:
        """Parse step execution results from output"""
        steps = []
        lines = stdout.split('\n')
        
        self.logger.info("🔍 Parsing step results from output...")
        
        for i, line in enumerate(lines):
            # Look for step execution patterns
            # Pattern: "Step 1/5: Some step text"
            if 'Step ' in line and '/' in line and ':' in line:
                try:
                    # Extract step number and text
                    step_part = line.split('Step ')[1]
                    step_info = step_part.split(':', 1)
                    
                    if len(step_info) < 2:
                        continue
                    
                    step_num_part = step_info[0].strip().split('/')[0]
                    step_num = int(step_num_part)
                    step_text = step_info[1].strip()
                    
                    # Default status
                    status = "UNKNOWN"
                    selector = ""
                    agent = ""
                    
                    # Check next 5 lines for status/selector info
                    for j in range(i, min(i + 5, len(lines))):
                        next_line = lines[j]
                        
                        # Status indicators
                        if "[OK]" in next_line or "✅" in next_line or "[PASSED]" in next_line:
                            status = "PASSED"
                        elif "[FAILED]" in next_line or "❌" in next_line or "[FAIL]" in next_line:
                            status = "FAILED"
                        elif "[SKIPPED]" in next_line:
                            status = "SKIPPED"
                        
                        # Extract selector if present
                        if "selector:" in next_line.lower() or "using:" in next_line.lower():
                            parts = next_line.split(':', 1)
                            if len(parts) > 1:
                                selector = parts[1].strip()[:100]
                        
                        # Extract agent if present
                        if "agent:" in next_line.lower() or "(agent:" in next_line.lower():
                            if "L1" in next_line or "l1" in next_line:
                                agent = "L1"
                            elif "L2" in next_line or "l2" in next_line:
                                agent = "L2"
                            elif "L3" in next_line or "l3" in next_line:
                                agent = "L3"
                            elif "Learning" in next_line:
                                agent = "Learning"
                    
                    steps.append({
                        'step_number': step_num,
                        'step_text': step_text[:200],
                        'status': status,
                        'selector': selector,
                        'agent_used': agent,
                        'confidence': 1.0 if status == "PASSED" else 0.0
                    })
                    
                    self.logger.info(f"   Step {step_num}: {status}")
                    
                except Exception as e:
                    self.logger.debug(f"Could not parse step line: {line} - {e}")
                    continue
        
        # If no steps found, try alternate parsing
        if not steps:
            self.logger.warning("   No steps found with main pattern, trying alternate parsing...")
            # Try finding any lines with [OK] or [FAILED]
            for i, line in enumerate(lines):
                if any(marker in line for marker in ["[OK]", "[FAILED]", "✅", "❌"]):
                    status = "PASSED" if any(m in line for m in ["[OK]", "✅"]) else "FAILED"
                    steps.append({
                        'step_number': i + 1,
                        'step_text': line.strip()[:200],
                        'status': status,
                        'selector': '',
                        'agent_used': '',
                        'confidence': 1.0 if status == "PASSED" else 0.0
                    })
        
        self.logger.info(f"   ✅ Parsed {len(steps)} steps from output")
        return steps
    
    def _save_steps_to_db(self, execution_id: str, steps: list):
        """Save parsed steps to database"""
        if not steps:
            self.logger.warning("   No steps to save")
            return
        
        self.logger.info(f"   💾 Saving {len(steps)} steps to database...")
        
        for step in steps:
            db_step = ExecutionStep(
                execution_id=execution_id,
                step_num=step.get('step_number', 0),
                step_text=step.get('step_text', '')[:200],
                selector=step.get('selector', ''),
                confidence=step.get('confidence', 0.0),
                agent_used=step.get('agent_used', ''),
                action_type=step.get('action_type', ''),
                status=step.get('status', 'UNKNOWN')
            )
            self.db.add(db_step)
        
        try:
            self.db.commit()
            self.logger.info(f"   ✅ Saved {len(steps)} steps")
        except Exception as e:
            self.logger.error(f"   ❌ Failed to save steps: {e}")
            self.db.rollback()
    
    def _extract_error(self, stderr: str, stdout: str) -> str:
        """Extract meaningful error message"""
        error_lines = []
        
        # Try stderr first
        for line in stderr.split('\n'):
            line_strip = line.strip()
            if line_strip and not line_strip.startswith('[INFO]'):
                # Skip common non-error messages
                if any(skip in line_strip.lower() for skip in ['deprecation', 'warning:', 'future']):
                    continue
                error_lines.append(line_strip)
        
        # If no meaningful stderr, check stdout for errors
        if not error_lines:
            for line in stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                    error_lines.append(line.strip())
        
        # Return first few meaningful lines
        if error_lines:
            return '\n'.join(error_lines[:5])
        
        return "Test execution failed (check logs for details)"
    
    def _log_summary(self, state: Dict, stdout: str, stderr: str):
        """Log execution summary"""
        self.logger.info("="*70)
        self.logger.info("📊 EXECUTION SUMMARY")
        self.logger.info(f"   Ticket: {state['ticket_id']}")
        self.logger.info(f"   Status: {state['overall_status']}")
        self.logger.info(f"   Steps: {len(state.get('step_results', []))}")
        self.logger.info(f"   Report: {state['report_path'] or 'None'}")
        self.logger.info(f"   Script: {state['script_path'] or 'None'}")
        self.logger.info(f"   Video: {state['video_path'] or 'None'}")
        if state.get('error_message'):
            self.logger.info(f"   Error: {state['error_message'][:100]}")
        self.logger.info("="*70)
    
    def save_execution_results(self, execution_id: str, state: Dict):
        """Save final execution results - compatibility method"""
        self.logger.info(f"💾 Final results saved for {execution_id}")
    
    def _get_python_executable(self, external_project_path: Path) -> Path:
        """Find Python executable - prioritize project venv"""
        # Try project venv first
        venv_paths = [
            external_project_path / "venv" / "Scripts" / "python.exe",  # Windows
            external_project_path / "venv" / "bin" / "python",          # Unix/Mac
        ]
        
        for venv_python in venv_paths:
            if venv_python.exists():
                self.logger.info(f"✅ Found venv Python: {venv_python}")
                return venv_python
        
        # Try system Python
        for cmd in ["python", "python3", "py"]:
            python_path = shutil.which(cmd)
            if python_path:
                self.logger.warning(f"⚠️ Using system Python: {python_path}")
                return Path(python_path)
        
        # Last resort - current interpreter
        self.logger.warning(f"⚠️ Using current interpreter: {sys.executable}")
        return Path(sys.executable)