import { Component, ElementRef, ViewChild, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ApiService, TestSummary, TicketDetail } from '../services/api.service';
import { interval, Subscription } from 'rxjs';
import { switchMap, takeWhile } from 'rxjs/operators';
import { MatIconModule } from '@angular/material/icon';
import { MsalService } from '@azure/msal-angular';
import { SelectorFeedbackService } from '../services/selector-feedback.service';

interface ChatMessage {
  type: 'user' | 'bot';
  text: string;
  images?: string[];
  timestamp: Date;
  liked?: boolean;
  disliked?: boolean;
  copied?: boolean;
  isJiraTicket?: boolean;
  isRunning?: boolean;
  executionStatus?: 'success' | 'failed' | 'stopped';
  executionMessage?: string;
  waitingForConfirmation?: boolean;
  executionProgress?: number;
  reportGenerated?: boolean;
  executionId?: string;
  ticketDetails?: TicketDetail;
  reportPath?: string;
  scriptPath?: string;
  videoPath?: string;
  generatedFiles?: string[];
  hasExistingScripts?: boolean;
  scriptsCount?: number;
  summary?: TestSummary;
  showSummary?: boolean;
  correctedSelector?: string;
  feedbackSubmitted?: boolean;
  rerunSummary?: TestSummary;
  showRerunSummary?: boolean;
  consolidatedFeedback?: string;
}

interface ChatSession {
  id: string;
  title: string;
  date: Date;
  messages: ChatMessage[];
  isEditing?: boolean;
  editTitle?: string;
}

// interface JiraTicketDetails {
//   ticket_id: string;
//   summary: string;
//   description: string;
// }

@Component({
  selector: 'app-test-generation',
  standalone: true,
  imports: [CommonModule, FormsModule, MatIconModule],
  templateUrl: './test-generation.component.html',
  styleUrl: './test-generation.component.scss'
})
export class TestGenerationComponent implements OnInit, OnDestroy {
  private readonly MAX_HISTORY_COUNT = 10;
  username: string = '';
  activeTab: 'tab1' | 'tab2' = 'tab1';
  userMessage: string = '';
  currentMessage: string = '';
  chatHistories: ChatSession[] = [];
  currentChatMessages: ChatMessage[] = [];
  loading: boolean = false;
  aborting: boolean = false;

  editingIndex: number | null = null;
  backupTitle: string = '';
  showDeleteDialog: boolean = false;
  selectedIndex: number | null = null;
  deletingSessionId: string | null = null;
  overallCorrectedSelector: string = '';
overallFeedbackSubmitted: boolean = false;

  private pollingSubscription?: Subscription;

  isVideoLoaded: boolean = false; 

  @ViewChild('chatContainer') chatContainer!: ElementRef;
  @ViewChild('messageTextarea') messageTextarea!: ElementRef;

  currentChatId: string | null = null;
  selectedImages: string[] = [];
  jiraTicketId: string = '';
  jiraDetails: any = null;
  error: string = '';

// ADD THESE NEW PROPERTIES HERE:
  jiraTicket: any = null;
  isLoadingJira: boolean = false;

  constructor(
    // private jiraService: JiraService,
    private apiService: ApiService,
    private msalService: MsalService,
    private http: HttpClient,  // ADD HttpClient injection
    private selectorFeedbackService: SelectorFeedbackService
  ) {}

  ngOnInit(): void {
    this.loadChatHistories();
    const account = this.msalService.instance.getActiveAccount();
    if (account && account.name) {
      this.username = this.extractFirstName(account.name);
    }
    // Restore current chat if available
  const savedChatId = localStorage.getItem('testGenCurrentChatId');
  if (savedChatId) {
    const found = this.chatHistories.find(chat => chat.id === savedChatId);
    if (found) {
      this.currentChatMessages = [...found.messages];
      this.currentChatId = found.id;
    }
  }
  }

  private extractFirstName(fullName: string): string {
    fullName = fullName.split('–')[0];
    fullName = fullName.split('-')[0];
    fullName = fullName.split('(')[0];
    fullName = fullName.split('@')[0];
    return fullName.trim().split(' ')[0];
  }

  ngOnDestroy(): void {
    if (this.pollingSubscription) {
      this.pollingSubscription.unsubscribe();
    }
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    if (this.chatContainer) {
      this.chatContainer.nativeElement.scrollTo({
        top: this.chatContainer.nativeElement.scrollHeight,
        behavior: 'smooth'
      });
    }
  }

  private isJiraTicket(input: string): boolean {
    const jiraPattern = /^[A-Z]{2,10}-\d+$/i;
    return jiraPattern.test(input.trim());
  }

  fetchJiraDetails() {
    if (!this.jiraTicketId.trim()) return;
    this.loading = true;
    this.error = '';
    this.apiService.getJiraTicket(this.jiraTicketId.trim()).subscribe({
      next: (data) => {
        this.jiraDetails = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Could not fetch Jira ticket details.';
        this.loading = false;
      }
    });
  }

  submitMessage() {
    const message = this.currentMessage?.trim() || '';
    if (message) {
      this.loading = true;
      const isJira = this.isJiraTicket(message);
      const userMessage: ChatMessage = {
        type: 'user',
        text: message,
        timestamp: new Date(),
        isJiraTicket: isJira
      };
      this.currentChatMessages.push(userMessage);
      this.saveChatToHistory();
      if (isJira) {
        this.fetchTicketAndGenerate(message);
      } else {
        const botMessage: ChatMessage = {
          type: 'bot',
          text: '⚠️ Please provide a valid JIRA ticket number (e.g., RBPLCD-8835) to start the test generation process.',
          timestamp: new Date()
        };
        this.currentChatMessages.push(botMessage);
        this.loading = false;
        this.saveChatToHistory();
      }
      this.currentMessage = '';
    }
  }


private pollExecutionStatus(executionId: string, message: ChatMessage) {
  this.pollingSubscription = interval(5000).pipe(
    switchMap(() => {
      return this.apiService.getExecutionStatus(executionId);
    }),
    // Keep polling until test is completed/failed AND summary is available
    takeWhile((status) => {
      // Continue polling if still running or pending, or if summary is not yet available
      return (
        status.status === 'running' ||
        status.status === 'pending' ||
        ((status.status === 'completed' || status.status === 'failed') && !status.summary_available)
      );
    }, true)
  ).subscribe({
    next: (status) => {
      // Update progress
      message.executionProgress = status.progress || 0;

      // Ensure ticketDetails exists BEFORE completion
      if (!message.ticketDetails && status.ticket_id) {
        message.ticketDetails = {
          ticket_id: status.ticket_id
        } as TicketDetail;
        console.log('✅ Set ticketDetails from status:', message.ticketDetails);
      }

      if (status.status === 'running') {
        message.text = `🔄 Test execution in progress...\n\n🆔 Execution ID: ${executionId}\n📊 Progress: ${status.progress}%\n⏱️ Status: ${status.message}`;
      }

      // Only proceed if test is completed/failed AND summary is available
      if (
        (status.status === 'completed' || status.status === 'failed') &&
        status.summary_available &&
        message.ticketDetails?.ticket_id
      ) {
        message.isRunning = false;
        message.executionProgress = 100;
        message.executionStatus = status.overall_status === 'PASSED' ? 'success' : 'failed';

        const statusIcon = status.overall_status === 'PASSED' ? '✅' : '❌';
        const statusText = status.overall_status === 'PASSED' ? 'PASSED' : 'FAILED';

        message.executionMessage = `${statusIcon} Test execution completed!\n\nOverall Status: ${statusText}`;
        message.text = message.executionMessage;
        message.reportGenerated = true;
        message.reportPath = status.report_path;
        message.scriptPath = status.script_path;
        
        // 🔥 NEW: Convert video path to playable URL with proper base URL
        if (status.video_path) {
          // Use window.location to get the full base URL for video streaming
          const baseUrl = `${window.location.protocol}//${window.location.host}`;
          message.videoPath = `${baseUrl}/api/download-video/${message.executionId}`;
          console.log('🎬 Video URL configured:', {
            protocol: window.location.protocol,
            host: window.location.host,
            baseUrl: baseUrl,
            executionId: message.executionId,
            finalUrl: message.videoPath,
            videoExists: !!status.video_path
          });
        } else {
          console.warn('⚠️ No video_path in status:', status);
        }

        message.generatedFiles = [
          status.report_path,
          status.script_path,
          status.video_path
        ].filter((path): path is string => !!path && path.trim() !== '');

        // LOAD SUMMARY
        console.log('📊 Execution completed and summary available, loading summary:', {
          summary_available: status.summary_available,
          ticket_id: message.ticketDetails.ticket_id
        });

        this.loadTestSummary(message);
        this.saveChatToHistory();
      }
    },
    error: (error) => {
      console.error('❌ Polling error:', error);
      message.isRunning = false;
      message.executionStatus = 'failed';
      message.executionMessage = '❌ Error monitoring test execution. Please check the logs.';
      message.text = message.executionMessage;
      this.saveChatToHistory();
    }
  });
}

// Add this method to debug
logFeedbackState(message: ChatMessage) {
  console.log('📝 Feedback state:', {
    hasSummary: !!message.summary,
    executionStatus: message.executionStatus,
    isRunning: message.isRunning,
    correctedSelector: message.correctedSelector,
    feedbackSubmitted: message.feedbackSubmitted
  });
}

// Add this method after hasFailedStep()
shouldShowFeedbackSection(message: ChatMessage): boolean {
  // Check if summary exists and has either steps or step_results
  if (!message.summary) {
    return false;
  }

  // Check for steps array
  const hasSteps = message.summary.steps && Array.isArray(message.summary.steps) && message.summary.steps.length > 0;
  
  // Check overall status is FAILED
  const isFailed = message.summary.summary?.overall_status === 'FAILED';
  
  // Check feedback not already submitted
  const notSubmitted = !message.feedbackSubmitted;

  return hasSteps && isFailed && notSubmitted;
}

  private loadTestSummary(message: ChatMessage) {
  console.log('🔍 loadTestSummary called with:', {
    has_ticketDetails: !!message.ticketDetails,
    ticket_id: message.ticketDetails?.ticket_id
  });

  if (!message.ticketDetails?.ticket_id) {
    console.error('❌ No ticket ID available for summary');
    return;
  }

  const ticketId = message.ticketDetails.ticket_id;
  console.log(`📊 Fetching summary for ticket: ${ticketId}`);

  this.apiService.getTestSummary(ticketId).subscribe({
    next: (summary) => {
      console.log('✅ Summary loaded successfully:', summary);
      // Defensive: ensure steps is always an array
      if (!Array.isArray(summary.steps)) {
        summary.steps = [];
      }
      message.summary = summary;
      message.showSummary = true;
      // Initialize feedback properties
      message.correctedSelector = '';
      message.feedbackSubmitted = false;
      this.saveChatToHistory();
    },
    error: (error) => {
      console.error('❌ Could not load summary:', {
        status: error.status,
        message: error.message,
        detail: error.error?.detail
      });
    }
  });
}


  toggleSummary(message: ChatMessage) {
    if (message.summary) {
      message.showSummary = !message.showSummary;
      this.saveChatToHistory();
    }
  }

  getStatusColorClass(status: string): string {
    switch (status) {
      case 'PASSED':
        return 'status-passed';
      case 'FAILED':
        return 'status-failed';
      default:
        return 'status-unknown';
    }
  }


submitSelectorFeedback(step: any, message: ChatMessage) {
  if (!message.ticketDetails?.ticket_id) {
    alert('Ticket ID is missing. Cannot submit feedback.');
    return;
  }
  
  // Clean the step text before sending
  const cleanStepText = step.description.split("Failed to process")[0].trim();
  
  this.selectorFeedbackService.submitFeedback({
    ticket_id: message.ticketDetails?.ticket_id,
    step_number: step.step_num,
    step_text: cleanStepText,
    corrected_selector: step.correctedSelector,
    module: message.ticketDetails.module,
    action_type: step.action_type,
    status: step.status
  }).subscribe({
    next: () => {
      // 🔥 CRITICAL FIX: Mark THIS STEP as submitted
      step.feedbackSubmitted = true;
      // 🔥 FIX 2: ALSO mark it in summary.step_results (CRITICAL)
      if (message.summary?.steps) {
        const summaryStep = message.summary.steps.find(
          (s: any) => s.step_number === step.step_num
        );
        
        if (summaryStep) {
          summaryStep.feedbackSubmitted = true;
          summaryStep.correctedSelector = step.correctedSelector;
          console.log(`✅ Marked step ${step.step_num} as submitted in summary.step_results`);
        } else {
          console.warn(`⚠️ Could not find step ${step.step_num} in summary.step_results`);
        }
      }
      
      // Also update message-level selector (optional, for overall feedback UI)
      message.correctedSelector = step.correctedSelector;
      
      // Save to localStorage
      this.saveChatToHistory();
      
      console.log('✅ Feedback submitted for step:', step.step_num);
    },
    error: (err) => {
      console.error('❌ Failed to submit feedback:', err);
      alert('Failed to submit feedback. Please try again.');
    }
  });
}



hasFailed(message: ChatMessage): boolean {
  return message.summary?.steps?.some(step => step.status === 'FAILED') || false;
}

submitConsolidatedFeedbackAndRerun(message: ChatMessage) {
  if (!message.ticketDetails?.ticket_id) {
    alert('Ticket ID is missing.');
    return;
  }
  // 🔥 SAFER CHECK - Handle undefined case
  const feedbackText = message.consolidatedFeedback?.trim() || '';
if (!feedbackText) {
    alert('Please enter your feedback.');
    return;
  }
  // if (!message.consolidatedFeedback || !message.consolidatedFeedback.trim()) {
  //   alert('Please enter your feedback.');
  //   return;
  // }

  // console.log('📝 Submitting consolidated feedback:', message.consolidatedFeedback);
  console.log('📝 Submitting consolidated feedback:', feedbackText);

  // Mark as running
  message.feedbackSubmitted = true;
  message.isRunning = true;
  message.executionProgress = 0;
  message.executionStatus = undefined;
  message.showSummary = false;

  // 🔥 FIX: Get steps array safely - only use 'steps' as per TestSummary type
  const stepsArray = message.summary?.steps || [];
  
  if (!stepsArray || stepsArray.length === 0) {
    console.error('❌ No steps found in summary');
    message.isRunning = false;
    message.feedbackSubmitted = false;
    alert('No test steps found. Cannot process feedback.');
    return;
  }

  // 🔥 STEP 1: Parse feedback lines and submit each to /api/selector-feedback
  // const feedbackLines = message.consolidatedFeedback.trim().split('\n').filter(line => line.trim());
  const feedbackLines = feedbackText.split('\n').filter(line => line.trim());
  
  const feedbackPromises: Promise<any>[] = [];

  for (const line of feedbackLines) {
    // Parse line: "Step 3 is false positive, should use [data-attribute='default_Measurement01']"
    const match = line.match(/step\s+(\d+)/i);
    if (!match) {
      console.warn(`⚠️ Could not parse step number from: "${line}"`);
      continue;
    }

    const stepNumber = parseInt(match[1]);

    // Extract selector (text between quotes or brackets)
    const selectorMatch = line.match(/\[([^\]]+)\]|["']([^"']+)["']/);
    if (!selectorMatch) {
      console.warn(`⚠️ Could not parse selector from: "${line}"`);
      continue;
    }

    const correctedSelector = selectorMatch[1] || selectorMatch[2];

    // 🔥 FIX: Find the step in summary with safe navigation
    // const step = message.summary?.steps?.find((s: any) => 
    //   (s.step_number === stepNumber) || (s.step_num === stepNumber)
    // );
    const step = stepsArray.find((s: any) => 
      (s.step_number === stepNumber) || (s.step_num === stepNumber)
    );
    
    if (!step) {
      console.warn(`⚠️ Step ${stepNumber} not found in summary`);
      continue;
    }

    // 🔥 FIX: Safely extract step text with multiple fallbacks
    const cleanStepText = 
      step.description?.split("Failed to process")[0].trim() || 
      `Step ${stepNumber}`;

    // 🔥 SUBMIT TO BACKEND (generates embedding and saves to pending/)
    const feedbackPayload = {
      ticket_id: message.ticketDetails?.ticket_id,
      step_number: stepNumber,
      step_text: cleanStepText,
      corrected_selector: correctedSelector,
      module: message.ticketDetails.module || '',
      action_type: step.action_type || 'click',
      status: step.status || 'FAILED'
    };

    console.log(`📤 Submitting feedback for Step ${stepNumber}:`, feedbackPayload);

    // Add to promises array
    feedbackPromises.push(
      this.selectorFeedbackService.submitFeedback(feedbackPayload).toPromise()
    );
  }

  if (feedbackPromises.length === 0) {
    alert('No valid feedback found. Please check format:\nExample: "Step 3: [selector]" or "Step 3 is false positive, should use [selector]"');
    message.isRunning = false;
    message.feedbackSubmitted = false;
    return;
  }

  // 🔥 STEP 2: Wait for ALL feedback submissions to complete
  Promise.all(feedbackPromises)
    .then(() => {
      console.log(`✅ Submitted ${feedbackPromises.length} feedback entries to backend`);

      // 🔥 STEP 3: NOW trigger rerun with feedback
      this.http.post<any>(`${window.location.origin}/api/rerun-with-feedback`, {
        ticket_id: message.ticketDetails!.ticket_id,
        // feedback_text: (message.consolidatedFeedback ?? '').trim()
        feedback_text: message.consolidatedFeedback!.trim()
      }).subscribe({
        next: (res) => {
          console.log('✅ Feedback submitted, rerun started:', res);

          if (res.execution_id) {
            message.executionId = res.execution_id;
            message.executionMessage = '🔄 Processing feedback and rerunning test...';

            // Start polling
            this.pollExecutionStatus(res.execution_id, message);
          } else {
            console.error('❌ No execution_id returned');
            message.isRunning = false;
            message.feedbackSubmitted = false;
            alert('Rerun failed: No execution ID received');
          }
        },
        error: (err) => {
          console.error('❌ Failed to start rerun:', err);
          message.isRunning = false;
          message.feedbackSubmitted = false;
          alert('Failed to submit feedback. Please try again.');
        }
      });
    })
    .catch((err) => {
      console.error('❌ Failed to submit feedback to backend:', err);
      message.isRunning = false;
      message.feedbackSubmitted = false;
      alert('Failed to submit feedback. Please check console for details.');
    });
}

anyFeedbackSubmitted(steps: any[]): boolean {
  return steps.some(s => s.feedbackSubmitted);
}


rerunTestWithFeedback(message: any) {

  const steps = message?.summary?.steps || message?.summary?.step_results;

  if (!steps || steps.length === 0) {
    console.error('❌ No steps found to collect feedback', message);
    return;
  }

  const feedbackSelectors = steps
    .filter((step: any) => step.feedbackSubmitted && step.correctedSelector)
    .map((step: any) => {
      const stepNo = step.step_num ?? step.step_number ?? 'unknown';
      return `Step ${stepNo}: ${step.correctedSelector}`;
    });

  if (feedbackSelectors.length === 0) {
    console.error('❌ No corrected selectors found');
    return;
  }

  const feedbackText = feedbackSelectors.join('\n');

  // ✅ FIX: Resolve ticket_id safely
  const ticketId =
    message.ticket_id ||
    message?.summary?.ticket_id ||
    message?.execution?.ticket_id;

  if (!ticketId) {
    console.error('❌ ticket_id not found in message object', message);
    return;
  }

  console.log('🔁 Sending feedback:', {
    ticket_id: ticketId,
    feedback_text: feedbackText
  });

  this.http.post(`${window.location.origin}/api/rerun-with-feedback`, {
    ticket_id: ticketId,
    feedback_text: feedbackText
  }).subscribe({
    next: (res) => {
      console.log('✅ Rerun started', res);
    },
    error: (err) => {
      console.error('❌ Rerun failed', err);
    }
  });
}











// rerunTestWithFeedback(message: any): void {

//   // 1️⃣ Safety checks
//   if (!message?.summary?.step_results || !Array.isArray(message.summary.step_results)) {
//     console.error('❌ No step results found to collect feedback');
//     return;
//   }

//   // 2️⃣ Collect corrected selectors entered by user
//   const feedbackSelectors = message.summary.step_results
//     .filter((step: any) => step.feedbackSubmitted && step.correctedSelector?.trim())
//     .map((step: any) =>
//       `Step ${step.step_number}: ${step.correctedSelector.trim()}`
//     );

//   // 3️⃣ Join into single feedback text
//   const feedbackText = feedbackSelectors.join('\n');

//   if (!feedbackText) {
//     console.error('❌ Feedback text is empty. Rerun aborted.');
//     return;
//   }

//   // 4️⃣ Debug log (VERY IMPORTANT)
//   console.log('🔁 Sending feedback to backend:', feedbackText);

//   // 5️⃣ Update UI state BEFORE calling API
//   message.isRunning = true;
//   message.executionStatus = null;
//   message.executionProgress = 0;
//   message.showSummary = false;
//   message.summary = null;

//   // 6️⃣ Call backend rerun API
//   this.http.post<any>('http://localhost:8000/api/rerun-with-feedback', {
//     ticket_id: message.ticketDetails?.ticket_id || message.ticket_id,
//     feedback_text: feedbackText
//   }).subscribe({
//     next: (res) => {
//       console.log('✅ Rerun started:', res);

//       // 7️⃣ Save new execution ID returned by backend
//       if (res.execution_id) {
//         message.executionId = res.execution_id;

//         // 8️⃣ Start polling NEW execution
//         this.pollExecutionStatus(message.executionId, message);
//       } else {
//         console.error('❌ Backend did not return execution_id');
//         message.isRunning = false;
//       }
//     },
//     error: (err) => {
//       console.error('❌ Failed to start rerun:', err);
//       message.isRunning = false;
//     }
//   });
// }


submitOverallSelectorFeedback(message: ChatMessage) {
  if (!message.ticketDetails?.ticket_id) {
  alert('Ticket ID is missing. Cannot submit feedback.');
  return;
}
  // Send feedback for the whole test, not per step
  this.selectorFeedbackService.submitFeedback({
    ticket_id: message.ticketDetails.ticket_id,
    corrected_selector: this.overallCorrectedSelector,
    // ...other fields as needed
  }).subscribe(() => {
    this.overallFeedbackSubmitted = true;
  });
}

  getAgentBadgeClass(agent: string): string {
    switch (agent) {
      case 'L1':
        return 'agent-l1';
      case 'L2':
        return 'agent-l2';
      case 'L3':
        return 'agent-l3';
      default:
        return 'agent-default';
    }
  }

submitOverallFeedback(message: any) {
  if (!message.ticketDetails?.ticket_id) {
    alert('Ticket ID is missing.');
    return;
  }
  if (!message.userFeedback || !message.userFeedback.trim()) {
    alert('Please enter your feedback.');
    return;
  }

  // Send feedback to backend
  this.http.post<any>(`${window.location.origin}/api/rerun-with-feedback`, {
    ticket_id: message.ticketDetails.ticket_id,
    feedback_text: message.userFeedback.trim()
  }).subscribe({
    next: (res) => {
      message.feedbackSubmitted = true;
      message.isRunning = true;
      message.executionProgress = 0;
      message.executionStatus = null;
      message.executionMessage = '🔄 Rerunning test with your feedback...';
      if (res.execution_id) {
        message.executionId = res.execution_id;
        this.pollExecutionStatus(res.execution_id, message);
      }
    },
    error: (err) => {
      alert('Failed to submit feedback. Please try again.');
    }
  });
}

  private fetchTicketAndGenerate(ticketId: string) {
  this.isLoadingJira = true;
  this.jiraTicket = null;

  // 🔥 STEP 1: Fetch ticket from Jira API
  this.apiService.getJiraTicket(ticketId).subscribe({
    next: (data: any) => {
      console.log('✅ Parsed Jira ticket:', data);
      this.jiraTicket = data;

      // 🔥 STEP 2: Check for existing scripts after getting Jira ticket
      this.apiService.listScripts(ticketId).subscribe({
        next: (scriptsResponse) => {
          this.isLoadingJira = false;
          this.loading = false;

          const hasScripts = scriptsResponse.scripts_count > 0;

          // Create bot message with both Jira data AND script info
          const botMessage: ChatMessage = {
            type: 'bot',
            text: `✅ Found JIRA ticket: ${data.ticket_id}\n\n📋 Title: ${data.title}\n📦 Module: ${data.module || 'N/A'}\n\n🔢 Test Steps: ${data.steps?.length || 0}\n\n${hasScripts ? '🚀 Script found! Starting test rerun...' : 'Do you want to run the automated test?'}`,
            timestamp: new Date(),
            ticketDetails: {
              ticket_id: data.ticket_id,
              title: data.title,
              description: data.raw_description,
              module: data.module,
              steps: data.steps
            } as TicketDetail,
            waitingForConfirmation: !hasScripts,  // 🔥 Only wait if NO scripts exist
            // 🔥 ADD SCRIPT INFORMATION
            hasExistingScripts: hasScripts,
            scriptsCount: scriptsResponse.scripts_count
          };

          this.currentChatMessages.push(botMessage);
          this.saveChatToHistory();

          // 🔥 AUTO-TRIGGER BASED ON SCRIPT AVAILABILITY
          if (hasScripts) {
            console.log('✅ Scripts found! Auto-triggering rerun...');
            setTimeout(() => {
              this.rerunTest(data.ticket_id);
            }, 1000);  // Small delay for UX
          } else {
            console.log('📝 No scripts found! Auto-triggering generate and run...');
            setTimeout(() => {
              this.runTest(data.ticket_id);
            }, 1000);  // Small delay for UX
          }
        },
        error: (scriptError) => {
          // 🔥 GRACEFUL FALLBACK: If script check fails, continue without script info
          console.warn('⚠️ Could not check scripts, continuing without script info:', scriptError);
          
          this.isLoadingJira = false;
          this.loading = false;

          const botMessage: ChatMessage = {
            type: 'bot',
            text: `✅ Found JIRA ticket: ${data.ticket_id}\n\n📋 Title: ${data.title}\n📦 Module: ${data.module || 'N/A'}\n\n🔢 Test Steps: ${data.steps?.length || 0}\n\n🚀 Starting test generation and run...`,
            timestamp: new Date(),
            ticketDetails: {
              ticket_id: data.ticket_id,
              title: data.title,
              description: data.raw_description,
              module: data.module,
              steps: data.steps
            } as TicketDetail,
            waitingForConfirmation: false,  // 🔥 No confirmation needed - auto-trigger
            // 🔥 DEFAULT TO NO SCRIPTS IF CHECK FAILS
            hasExistingScripts: false,
            scriptsCount: 0
          };

          this.currentChatMessages.push(botMessage);
          this.saveChatToHistory();

          // 🔥 AUTO-TRIGGER GENERATE AND RUN (since we couldn't check for scripts)
          console.log('📝 Script check failed! Auto-triggering generate and run...');
          setTimeout(() => {
            this.runTest(data.ticket_id);
          }, 1000);  // Small delay for UX
        }
      });
    },
    error: (error) => {
      console.error('❌ Error fetching Jira ticket:', error);
      this.isLoadingJira = false;
      this.loading = false;

      // UPDATED ERROR MESSAGE FOR JIRA BOARD
      const botMessage: ChatMessage = {
        type: 'bot',
        text: `❌ Error: Could not find ticket "${ticketId}" on Jira board.\n
        Please make sure:\n
        1. The ticket ID is correct (e.g., RBPLCD-1234)\n
        2. The ticket exists in your Jira board\n
        3. You have access permissions to view this ticket`,
        timestamp: new Date()
      };

      this.currentChatMessages.push(botMessage);
      this.saveChatToHistory();
    }
  });
}

  // 3. ADD THIS NEW METHOD after fetchTicketAndGenerate
  showMessage(message: string, type: string = 'success') {
    console.log(`[${type}] ${message}`);
  }


  onRunTestClick(message: ChatMessage) {
    if (message.ticketDetails) {
      message.waitingForConfirmation = false;
      const userConfirmMessage: ChatMessage = {
        type: 'user',
        text: '✅ Generate and Run Testcase',
        timestamp: new Date(),
        ticketDetails: message.ticketDetails  // ✅ ADD THIS - Preserve ticket details
      };
      this.currentChatMessages.push(userConfirmMessage);
      this.runTest(message.ticketDetails.ticket_id);
    }
  }

  onRerunTestClick(message: ChatMessage) {
  if (message.ticketDetails) {
    message.waitingForConfirmation = false;

    const userConfirmMessage: ChatMessage = {
      type: 'user',
      text: '🔄 Rerun Testcase',
      timestamp: new Date(),
      ticketDetails: message.ticketDetails  // ✅ ADD THIS - Preserve ticket details
    };
    this.currentChatMessages.push(userConfirmMessage);

    // Always call the API - let the backend handle script validation
    this.rerunTest(message.ticketDetails.ticket_id);
  }
}

  private runTest(ticketId: string) {
    // ✅ Find the original message with ticket details
  const originalMessage = this.currentChatMessages.find(
    msg => msg.ticketDetails?.ticket_id === ticketId
  );
    const botMessage: ChatMessage = {
      type: 'bot',
      text: `🚀 Starting test generation and execution for ${ticketId}...\n\nInitializing automation framework...`,
      timestamp: new Date(),
      isRunning: true,
      executionProgress: 0,
      ticketDetails: originalMessage?.ticketDetails
    };
    this.currentChatMessages.push(botMessage);
    this.saveChatToHistory();
    this.apiService.runTest(ticketId).subscribe({
      next: (response) => {
        botMessage.executionId = response.execution_id;
        botMessage.text = `✅ Test execution started!\n\n🆔 Execution ID: ${response.execution_id}\n\n⏳ Status: ${response.message}\n\n🔄 Monitoring progress...`;
        this.saveChatToHistory();
        this.pollExecutionStatus(response.execution_id, botMessage);
      },
      error: (error) => {
        botMessage.isRunning = false;
        botMessage.executionStatus = 'failed';
        botMessage.executionMessage = `❌ Failed to start test: ${error.error?.detail || error.message}`;
        botMessage.text = botMessage.executionMessage;
        this.saveChatToHistory();
      }
    });
  }

  private rerunTest(ticketId: string) {
    // ✅ Find the original message with ticket details
  const originalMessage = this.currentChatMessages.find(
    msg => msg.ticketDetails?.ticket_id === ticketId
  );
    const botMessage: ChatMessage = {
      type: 'bot',
      text: `🔄 Rerunning test for ${ticketId} using existing script...\n\nExecuting saved test script...`,
      timestamp: new Date(),
      isRunning: true,
      executionProgress: 0,
      ticketDetails: originalMessage?.ticketDetails
      
    };
    this.currentChatMessages.push(botMessage);
    this.saveChatToHistory();
    this.apiService.rerunTest(ticketId).subscribe({
      next: (response) => {
        botMessage.executionId = response.execution_id;
        botMessage.text = `✅ Test rerun started!\n\n🆔 Execution ID: ${response.execution_id}\n📜 Using script: ${response.script_path?.split('/').pop() || 'existing script'}\n\n⏳ Status: ${response.message}\n\n🔄 Monitoring progress...`;
        this.saveChatToHistory();
        this.pollExecutionStatus(response.execution_id, botMessage);
      },
      error: (error) => {
        const errorDetail = error.error?.detail || error.message;
        botMessage.isRunning = false;
        botMessage.executionStatus = 'failed';
        if (errorDetail.includes('No generated script found') || errorDetail.includes('script') || errorDetail.includes('Scripts folder not found')) {
          botMessage.executionMessage = `⚠️ No test script found for this ticket.\n\nPlease use "Generate and Run Testcase" first to create the test script.`;
        } else {
          botMessage.executionMessage = `❌ Failed to rerun test: ${errorDetail}`;
        }
        botMessage.text = botMessage.executionMessage;
        this.saveChatToHistory();
      }
    });
  }

  downloadReport(message: ChatMessage) {
    if (!message.executionId) {
      alert('❌ Execution ID not found');
      return;
    }
    this.apiService.downloadReport(message.executionId).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `test_report_${message.ticketDetails?.ticket_id || message.executionId}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      },
      error: (error) => {
        alert('❌ Failed to download report. Please check if the report was generated.');
      }
    });
  }

  downloadScript(message: ChatMessage) {
    if (!message.executionId) {
      alert('❌ Execution ID not found');
      return;
    }
    this.apiService.downloadScript(message.executionId).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `playwright_script_${message.ticketDetails?.ticket_id || message.executionId}.py`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      },
      error: (error) => {
        alert('❌ Failed to download script. Please check if the script was generated.');
      }
    });
  }

  downloadVideo(message: ChatMessage) {
    if (!message.executionId) {
      alert('❌ Execution ID not found');
      return;
    }
    this.apiService.downloadVideo(message.executionId).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `test_video_${message.ticketDetails?.ticket_id || message.executionId}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      },
      error: (error) => {
        alert('❌ Failed to download video. Please check if the video was recorded.');
      }
    });
  }

  // 🔥 NEW: Video event handlers
  onVideoLoadStart(event: any) {
    console.log('🎬 Video: loadstart event');
    this.isVideoLoaded = false;
  }

  onVideoCanPlay(event: any) {
    console.log('🎬 Video: canplay event - video loaded successfully!');
    this.isVideoLoaded = true;
  }

  onVideoError(event: any) {
    const video = event.target;
    console.error('❌ Video: error event', {
      error: video.error?.code,
      message: video.error?.message,
      src: video.src,
      currentSrc: video.currentSrc
    });
    this.isVideoLoaded = false;
    alert('❌ Failed to load video. Check browser console for details.');
  }

  hasPlaywrightScript(): boolean {
    return this.currentChatMessages.some(message =>
      message.type === 'bot' && message.reportGenerated
    );
  }

  onEnterPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.submitMessage();
    }
  }

  saveChatToHistory() {
    if (this.currentChatMessages.length > 0) {
      const firstUserMessage = this.currentChatMessages.find(msg => msg.type === 'user');
      const title = firstUserMessage ?
        firstUserMessage.text.substring(0, 100) + (firstUserMessage.text.length > 100 ? '...' : '') :
        'New Chat';
      if (this.currentChatId) {
        const existingChatIndex = this.chatHistories.findIndex(chat => chat.id === this.currentChatId);
        if (existingChatIndex >= 0) {
          const [updatedChat] = this.chatHistories.splice(existingChatIndex, 1);
          updatedChat.messages = [...this.currentChatMessages];
          updatedChat.title = title;
          updatedChat.date = new Date();
          this.chatHistories.unshift(updatedChat);
        } else {
          this.currentChatId = null;
          this.saveChatToHistory();
          return;
        }
      } else {
        this.currentChatId = Date.now().toString();
        const newChat: ChatSession = {
          id: this.currentChatId,
          title: title,
          date: new Date(),
          messages: [...this.currentChatMessages]
        };
        this.chatHistories.unshift(newChat);
      }
      while (this.chatHistories.length > this.MAX_HISTORY_COUNT) {
        this.chatHistories.pop();
      }
      this.saveToLocalStorage();
    }
    localStorage.setItem('testGenCurrentChatId', this.currentChatId || '');
  }

  loadChatHistories() {
    try {
      const stored = localStorage.getItem('testGenChatHistories');
      if (stored) {
        let loadedHistories = JSON.parse(stored);
        loadedHistories.forEach((chat: ChatSession) => {
          chat.date = new Date(chat.date);
          chat.messages.forEach(msg => {
            msg.timestamp = new Date(msg.timestamp);
          });
        });
        loadedHistories.sort((a: ChatSession, b: ChatSession) =>
          b.date.getTime() - a.date.getTime()
        );
        if (loadedHistories.length > this.MAX_HISTORY_COUNT) {
          loadedHistories = loadedHistories.slice(0, this.MAX_HISTORY_COUNT);
        }
        this.chatHistories = loadedHistories;
        if (loadedHistories.length !== JSON.parse(stored).length) {
          this.saveToLocalStorage();
        }
      }
    } catch (error) {
      this.chatHistories = [];
    }
  }

  private saveToLocalStorage() {
    try {
      localStorage.setItem('testGenChatHistories', JSON.stringify(this.chatHistories));
    } catch (error) {}
  }

  selectChat(index: number) {
    this.currentChatMessages = [...this.chatHistories[index].messages];
    this.currentChatId = this.chatHistories[index].id;
    this.activeTab = 'tab1';
  }

  viewChat(index: number, event: Event) {
    event.stopPropagation();
    this.aborting = true;
    setTimeout(() => {
      this.selectChat(index);
      this.aborting = false;
    }, 500);
  }

  startEdit(index: number, event: Event) {
    event.stopPropagation();
    this.backupTitle = this.chatHistories[index].title;
    this.editingIndex = index;
  }

  // saveEdit(index: number, event: Event) {
  //   event.stopPropagation();
  //   const target = event.target as HTMLElement;
  //   const newTitle = target.innerText.trim();

  //   if (newTitle && newTitle !== this.backupTitle) {
  //     this.chatHistories[index].title = newTitle;
  //     this.saveToLocalStorage();
  //   } else if (newTitle === '') {
  //     this.chatHistories[index].title = this.backupTitle;
  //     target.innerText = this.backupTitle;
  //   }

  //   this.editingIndex = null;
  //   this.backupTitle = '';
  // }



  // cancelEdit(index: number, event: Event) {
  //   event.stopPropagation();
  //   const target = event.target as HTMLElement;
  //   target.innerText = this.backupTitle;
  //   this.chatHistories[index].title = this.backupTitle;
  //   this.editingIndex = null;
  //   this.backupTitle = '';
  // }

  cancelEdit(index: number, event: Event, titleElement?: HTMLElement) {
  event.stopPropagation();
  
  // Get the actual title element
  const target = titleElement || (event.target as HTMLElement);
  
  target.innerText = this.backupTitle;
  this.chatHistories[index].title = this.backupTitle;
  this.editingIndex = null;
  this.backupTitle = '';
}

  showDialog(index: number, dialog: HTMLDialogElement, event: Event) {
    event.stopPropagation();
    this.selectedIndex = index;
    dialog.showModal();
  }

  closeDialog1(dialog: HTMLDialogElement, event: Event) {
    event.stopPropagation();
    dialog.close();
    this.selectedIndex = null;
  }
 

saveEdit(index: number, event: Event, titleElement?: HTMLElement) {
  event.stopPropagation();
  
  // Get the actual title element - either from parameter or from event target
  let target: HTMLElement;
  
  if (titleElement) {
    // Called from button click with template reference
    target = titleElement;
  } else {
    // Called from blur/enter on the span itself
    target = event.target as HTMLElement;
  }
  
  const newTitle = target.innerText.trim();
  
  console.log('📝 Saving edit:', {
    index,
    newTitle,
    backupTitle: this.backupTitle,
    eventType: event.type,
    targetType: target.tagName
  });

  if (!newTitle) {
    // Restore backup if empty
    this.chatHistories[index].title = this.backupTitle;
    target.innerText = this.backupTitle;
    this.editingIndex = null;
    this.backupTitle = '';
    return;
  }

  if (newTitle === this.backupTitle) {
    // No change
    this.editingIndex = null;
    this.backupTitle = '';
    return;
  }

  // Save to backend
  const sessionId = this.chatHistories[index].id;

  this.apiService.updateChatTitle(sessionId, newTitle).subscribe({
    next: (response) => {
      console.log('✅ Title updated successfully:', response);

      // Update local state
      this.chatHistories[index].title = newTitle;
      this.chatHistories[index].date = new Date(response.updated_at);

      // Update localStorage
      this.saveToLocalStorage();

      // Clear editing state
      this.editingIndex = null;
      this.backupTitle = '';
    },
    error: (error) => {
      console.error('❌ Failed to update title:', error);

      // Revert on error
      this.chatHistories[index].title = this.backupTitle;
      target.innerText = this.backupTitle;

      alert('Failed to update title. Please try again.');

      this.editingIndex = null;
      this.backupTitle = '';
    }
  });
}




// Also update the onConfirm method to call backend delete

onConfirm(dialog: HTMLDialogElement, event: Event) {
  event.stopPropagation();

  if (this.selectedIndex !== null) {
    const sessionToDelete = this.chatHistories[this.selectedIndex];
    const sessionId = sessionToDelete.id;

    this.deletingSessionId = sessionId;

    // Call backend API to delete
    this.apiService.deleteChatSession(sessionId).subscribe({
      next: (response) => {
        console.log('✅ Session deleted from backend:', response);

        // Remove from local array
        this.chatHistories.splice(this.selectedIndex!, 1);

        // Clear current chat if it was the deleted one
        if (this.currentChatId === sessionId) {
          this.currentChatMessages = [];
          this.currentChatId = null;
        }

        // Update localStorage
        this.saveToLocalStorage();

        // Reset state
        this.deletingSessionId = null;
        dialog.close();
        this.selectedIndex = null;
      },
      error: (error) => {
        console.error('❌ Failed to delete session:', error);
        alert('Failed to delete chat. Please try again.');

        this.deletingSessionId = null;
        dialog.close();
        this.selectedIndex = null;
      }
    });
  }
}
  startNewChat() {
    if (this.pollingSubscription) {
      this.pollingSubscription.unsubscribe();
    }

  // Reset all chat-related state
  this.currentChatId = null;
  this.currentChatMessages = [];
  this.userMessage = '';
  this.currentMessage = '';
  this.loading = false;
  this.aborting = false;

  // 🆕 ADD THESE LINES TO CLEAR JIRA TICKET STATE:
  this.jiraTicket = null;
  this.isLoadingJira = false;
  this.jiraTicketId = '';
  this.jiraDetails = null;
  this.error = '';

  this.activeTab = 'tab1';

  // Scroll to top
  setTimeout(() => {
    if (this.chatContainer) {
      this.chatContainer.nativeElement.scrollTop = 0;
    }
  }, 100);

  console.log('✅ Started new chat session');
}

  switchTab(tab: 'tab1' | 'tab2') {
  this.activeTab = tab;
  if (tab === 'tab1') {
    this.startNewChat();
  }
  }

  hasFailedStep(message: any): boolean {
  return !!message.summary?.steps?.some((step: any) => step.status === 'FAILED');
}

//   switchTab(tab: 'tab1' | 'tab2') {
//   if (this.activeTab !== tab) {
//     this.activeTab = tab;
//     if (tab === 'tab1' && this.currentChatMessages.length === 0) {
//       // Only start a new chat if there are no current messages
//       this.startNewChat();
//     }
//   }
// }

//   switchTab(tab: 'tab1' | 'tab2') {
//   if (this.activeTab !== tab) {
//     this.activeTab = tab;
//     if (tab === 'tab1') {
//       this.startNewChat();
//     }
//   }
// }
// get isHistoryTab(): boolean {
//   return this.activeTabIndex === 1;
// }
  onLike(message: ChatMessage) {
    if (message.disliked) {
      message.disliked = false;
    }
    message.liked = !message.liked;
    this.saveChatToHistory();
  }

  onDislike(message: ChatMessage) {
    if (message.liked) {
      message.liked = false;
    }
    message.disliked = !message.disliked;
    this.saveChatToHistory();
  }

  onCopy(message: ChatMessage) {
    if (message.text?.trim()) {
      navigator.clipboard.writeText(message.text).then(() => {
        message.copied = true;
        setTimeout(() => {
          message.copied = false;
          this.saveChatToHistory();
        }, 2000);
      }).catch(err => {});
    }
  }
}