import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface TicketDetail {
  id: number;
  ticket_id: string;
  title: string;
  module?: string;
  project_id: number;
  file_path?: string;
  steps?: { num: number; text: string }[];
  created_at?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  date: string;
  messages: any[];
}


export interface UpdateTitleRequest {
  session_id: string;
  new_title: string;
}

export interface UpdateTitleResponse {
  session_id: string;
  title: string;
  updated_at: string;
  message: string;
}
export interface ExecutionResponse {
  execution_id: string;
  ticket_id: string;
  status: string;
  message: string;
  script_path?: string;
}

export interface ExecutionStatus {
  execution_id: string;
  ticket_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  overall_status?: 'PASSED' | 'FAILED' | 'UNKNOWN';
  message: string;
  current_step?: string;
  steps_completed: number;
  steps_total: number;
  started_at?: string;
  completed_at?: string;
  report_path?: string;
  script_path?: string;
  video_path?: string;
  summary_available?: boolean;
}

export interface ScriptInfo {
  filename: string;
  path: string;
  created: string;
  size: number;
}

export interface ScriptsListResponse {
  ticket_id: string;
  scripts_count: number;
  scripts: ScriptInfo[];
}

// ============================================================================
// NEW: Summary Interfaces
// ============================================================================

export interface TestSummary {
  ticket_id: string;
  ticket_title: string;
  module: string;
  execution_date: string;
  generated_at: string;
  
  summary: {
    overall_status: 'PASSED' | 'FAILED' | 'UNKNOWN';
    total_steps: number;
    passed: number;
    failed: number;
    skipped: number;
    execution_time: string;
    execution_time_seconds: number;
    avg_confidence: number;
  };
  
  agent_usage: {
    l1_semantic_search: number;
    l2_dom_discovery: number;
    l3_vision: number;
    formatted: string;
  };
  
  step_details: Array<{
    step_number: number;
    step_text: string;
    selector: string;
    agent_used: string;
    confidence: number;
    status: string;
    error?: string;
  }>;
  
  artifacts: {
    video_path: string | null;
    report_path: string | null;
    has_video: boolean;
    has_report: boolean;
  };
  
  insights: {
    status_emoji: string;
    status_color: string;
    performance_rating: string;
    performance_emoji: string;
    automation_quality: string;
    quality_emoji: string;
    confidence_rating: string;
    confidence_emoji: string;
    recommendations: string[];
    critical_failures: Array<{
      step_number: number;
      step_text: string;
      error: string;
    }>;
  };
}

export interface SummaryListItem {
  ticket_id: string;
  ticket_title: string;
  module: string;
  execution_date: string;
  overall_status: string;
  total_steps: number;
  passed: number;
  failed: number;
  execution_time: string;
  avg_confidence: number;
  status_emoji: string;
  file_path: string;
}

export interface SummaryStats {
  total_executions: number;
  total_passed: number;
  total_failed: number;
  success_rate: number;
  avg_execution_time: number;
  avg_confidence: number;
  recent_executions: Array<{
    ticket_id: string;
    status: string;
    execution_date: string;
  }>;
}

export interface SummaryExistsResponse {
  exists: boolean;
  ticket_id: string;
  latest_execution: string | null;
  overall_status?: string;
  has_report?: boolean;
  has_video?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8000/api';  // Update with your backend URL

  constructor(private http: HttpClient) {}

  /**
   * Get ticket details by ticket ID
   */
  getTicketDetails(ticketId: string): Observable<TicketDetail> {
    return this.http.get<TicketDetail>(`${this.apiUrl}/tickets/${ticketId}`);
  }

  /**
   * Run test for a ticket (generate and execute)
   */
  runTest(ticketId: string): Observable<ExecutionResponse> {
    const params = new HttpParams().set('ticket_id', ticketId);
    return this.http.post<ExecutionResponse>(`${this.apiUrl}/execute-test`, null, { params });
  }

  /**
   * Rerun test for a ticket (using existing script)
   */
  rerunTest(ticketId: string): Observable<ExecutionResponse> {
    const params = new HttpParams().set('ticket_id', ticketId);
    return this.http.post<ExecutionResponse>(`${this.apiUrl}/rerun-test`, null, { params });
  }

  /**
   * Check if scripts exist for a ticket
   */
  listScripts(ticketId: string): Observable<ScriptsListResponse> {
    return this.http.get<ScriptsListResponse>(`${this.apiUrl}/scripts/${ticketId}`);
  }

  /**
   * Get execution status (for polling)
   */
  getExecutionStatus(executionId: string): Observable<ExecutionStatus> {
    return this.http.get<ExecutionStatus>(`${this.apiUrl}/execution-status/${executionId}`);
  }

  /**
   * Download HTML report
   */
  downloadReport(executionId: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/download-report/${executionId}`, {
      responseType: 'blob'
    });
  }

  /**
   * Download Playwright script
   */
  downloadScript(executionId: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/download-script/${executionId}`, {
      responseType: 'blob'
    });
  }

  /**
   * Download test video
   */
  downloadVideo(executionId: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/download-video/${executionId}`, {
      responseType: 'blob'
    });
  }

  updateChatTitle(sessionId: string, newTitle: string): Observable<UpdateTitleResponse> {
  const body: UpdateTitleRequest = {
    session_id: sessionId,
    new_title: newTitle
  };
  
  return this.http.put<UpdateTitleResponse>(
    `${this.apiUrl}/chat-session/title`, 
    body
  );
}

/**
 * Get all chat sessions for current user
 */
getChatSessions(): Observable<{ count: number; sessions: ChatSession[] }> {
  return this.http.get<{ count: number; sessions: ChatSession[] }>(
    `${this.apiUrl}/chat-sessions`
  );
}

/**
 * Delete a chat session
 */
deleteChatSession(sessionId: string): Observable<{ message: string; session_id: string }> {
  return this.http.delete<{ message: string; session_id: string }>(
    `${this.apiUrl}/chat-session/${sessionId}`
  );
}
  // ============================================================================
  // NEW: Summary API Methods
  // ============================================================================

  /**
   * Get latest test summary for a ticket (lightweight JSON)
   * Use this before downloading full HTML report
   */
  getTestSummary(ticketId: string): Observable<TestSummary> {
    return this.http.get<TestSummary>(`${this.apiUrl}/summary/${ticketId}`);
  }

  /**
   * Get specific summary by timestamp
   */
  getTestSummaryByTimestamp(ticketId: string, timestamp: string): Observable<TestSummary> {
    return this.http.get<TestSummary>(`${this.apiUrl}/summary/${ticketId}/${timestamp}`);
  }

  /**
   * List all available summaries with filtering
   */
  listAllSummaries(limit: number = 50, status?: string, module?: string): Observable<{ count: number; summaries: SummaryListItem[] }> {
    let params = new HttpParams().set('limit', limit.toString());
    
    if (status) {
      params = params.set('status', status);
    }
    
    if (module) {
      params = params.set('module', module);
    }
    
    return this.http.get<{ count: number; summaries: SummaryListItem[] }>(`${this.apiUrl}/summaries`, { params });
  }

  /**
   * Get overall statistics across all tests
   */
  getSummaryStats(): Observable<SummaryStats> {
    return this.http.get<SummaryStats>(`${this.apiUrl}/summary-stats`);
  }

  /**
   * Check if summary exists for a ticket
   */
  checkSummaryExists(ticketId: string): Observable<SummaryExistsResponse> {
    return this.http.get<SummaryExistsResponse>(`${this.apiUrl}/summary-exists/${ticketId}`);
  }
}