import { Component, ElementRef, ViewChild, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, TestSummary, TicketDetail } from '../services/api.service';
import { interval, Subscription } from 'rxjs';
import { switchMap, takeWhile } from 'rxjs/operators';
import { MatIconModule } from '@angular/material/icon';
import { MsalService } from '@azure/msal-angular';

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
}

interface ChatSession {
  id: string;
  title: string;
  date: Date;
  messages: ChatMessage[];
  isEditing?: boolean;
  editTitle?: string;
}

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
  activeTab: string = 'tab1';
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

  private pollingSubscription?: Subscription;

  @ViewChild('chatContainer') chatContainer!: ElementRef;
  @ViewChild('messageTextarea') messageTextarea!: ElementRef;

  currentChatId: string | null = null;
  selectedImages: string[] = [];

  constructor(private apiService: ApiService, private msalService: MsalService) {}

  ngOnInit(): void {
    this.loadChatHistories();
     const account = this.msalService.instance.getActiveAccount();

    if (account && account.name) {
      this.username = this.extractFirstName(account.name);           
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
  this.pollingSubscription = interval(3000).pipe(
    switchMap(() => {
      console.log('📊 Execution completed, checking summary...');
      return this.apiService.getExecutionStatus(executionId);
    }),
    takeWhile((status) => {
      return status.status === 'running' || status.status === 'pending';
    }, true)
  ).subscribe({
    next: (status) => {
      message.executionProgress = status.progress || 0;

      if (status.status === 'running') {
        message.text = `🔄 Test execution in progress...\n\n🆔 Execution ID: ${executionId}\n📊 Progress: ${status.progress}%\n⏱️ Status: ${status.message}`;
      }

      if (status.status === 'completed') {
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
        message.videoPath = status.video_path;
        
        message.generatedFiles = [
          status.report_path,
          status.script_path,
          status.video_path
        ].filter((path): path is string => !!path && path.trim() !== '');

        console.log('📊 Execution completed, checking summary...');
        console.log('Summary available:', status.summary_available);
        console.log('Ticket ID:', message.ticketDetails?.ticket_id);
        console.log('📊 Execution completed, checking summary...');
        console.log('Summary available:', status.summary_available);
        console.log('Ticket ID from status:', status.ticket_id);

        if (status.summary_available && status.ticket_id) {
          console.log('✅ Loading summary for ticket:', status.ticket_id);
          
          if (!message.ticketDetails) {
            message.ticketDetails = { ticket_id: status.ticket_id } as TicketDetail;
          }
          
          this.loadTestSummary(message);
        } else {
          console.warn('⚠️ Summary not available:', {
            summary_available: status.summary_available,
            ticket_id: status.ticket_id,
            has_ticketDetails: !!message.ticketDetails
          });
        }
      

        this.saveChatToHistory();
      } else if (status.status === 'failed') {
        message.isRunning = false;
        message.executionProgress = 100;
        message.executionStatus = 'failed';
        message.executionMessage = `❌ Test execution failed!\n\nError: ${status.message || 'Unknown error'}`;
        message.text = message.executionMessage;
        
        if (status.summary_available && message.ticketDetails?.ticket_id) {
          console.log('⚠️ Loading summary for failed execution...');
          this.loadTestSummary(message);
        }
        
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

private loadTestSummary(message: ChatMessage) {
  if (!message.ticketDetails?.ticket_id) {
    console.error('❌ No ticket ID available for summary');
    return;
  }

  const ticketId = message.ticketDetails.ticket_id;
  console.log(`📊 Fetching summary for ticket: ${ticketId}`);

  this.apiService.getTestSummary(ticketId).subscribe({
    next: (summary) => {
      console.log('✅ Summary loaded successfully:', summary);
      message.summary = summary;
      message.showSummary = true;
      this.saveChatToHistory();
    },
    error: (error) => {
      console.error('❌ Could not load summary:', error);
      console.error('Error details:', error.error);
      console.error('Status:', error.status);
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
  private fetchTicketAndGenerate(ticketId: string) {
    this.apiService.getTicketDetails(ticketId).subscribe({
      next: (ticketDetails) => {
        this.apiService.listScripts(ticketId).subscribe({
          next: (scriptsResponse) => {
            this.loading = false;

            const hasScripts = scriptsResponse.scripts_count > 0;

            const botMessage: ChatMessage = {
              type: 'bot',
              text: `✅ Found JIRA ticket: ${ticketDetails.ticket_id}\n\n📋 Title: ${ticketDetails.title}\n📦 Module: ${ticketDetails.module || 'N/A'}\n\n🔢 Test Steps Found: ${ticketDetails.steps?.length || 0}\n\nDo you want to run the automated test?`,
              timestamp: new Date(),
              ticketDetails: ticketDetails,
              waitingForConfirmation: true,
              hasExistingScripts: hasScripts,
              scriptsCount: scriptsResponse.scripts_count
            };

            this.currentChatMessages.push(botMessage);
            this.saveChatToHistory();
          },
          error: (scriptError) => {
            this.loading = false;

            const botMessage: ChatMessage = {
              type: 'bot',
              text: `✅ Found JIRA ticket: ${ticketDetails.ticket_id}\n\n📋 Title: ${ticketDetails.title}\n📦 Module: ${ticketDetails.module || 'N/A'}\n\n🔢 Test Steps Found: ${ticketDetails.steps?.length || 0}\n\nDo you want to run the automated test?`,
              timestamp: new Date(),
              ticketDetails: ticketDetails,
              waitingForConfirmation: true,
              hasExistingScripts: false,
              scriptsCount: 0
            };

            this.currentChatMessages.push(botMessage);
            this.saveChatToHistory();
          }
        });
      },
      error: (error) => {
        this.loading = false;

        const botMessage: ChatMessage = {
          type: 'bot',
          text: `❌ Error: Could not find ticket "${ticketId}".\n
          Please make sure:\n
          1. The ticket is uploaded to the system\n
          2. The ticket file exists in Jira_Tickets folder\n
          3. The ticket ID is correct`,
          timestamp: new Date()
        };

        this.currentChatMessages.push(botMessage);
        this.saveChatToHistory();
      }
    });
  }

  onRunTestClick(message: ChatMessage) {
    if (message.ticketDetails) {
      message.waitingForConfirmation = false;

      const userConfirmMessage: ChatMessage = {
        type: 'user',
        text: '✅ Generate and Run Testcase',
        timestamp: new Date()
      };
      this.currentChatMessages.push(userConfirmMessage);
      this.runTest(message.ticketDetails.ticket_id);
    }
  }

  onRerunTestClick(message: ChatMessage) {
    if (message.ticketDetails) {
      if (!message.hasExistingScripts || message.scriptsCount === 0) {
        const warningMessage: ChatMessage = {
          type: 'bot',
          text: '⚠️ No existing test scripts found for this ticket.\n\nPlease use "Generate and Run Testcase" first to create the test script.',
          timestamp: new Date()
        };
        this.currentChatMessages.push(warningMessage);
        this.saveChatToHistory();
        return;
      }

      message.waitingForConfirmation = false;

      const userConfirmMessage: ChatMessage = {
        type: 'user',
        text: '🔄 Rerun Testcase',
        timestamp: new Date()
      };
      this.currentChatMessages.push(userConfirmMessage);
      this.rerunTest(message.ticketDetails.ticket_id);
    }
  }

  private runTest(ticketId: string) {
    const botMessage: ChatMessage = {
      type: 'bot',
      text: `🚀 Starting test generation and execution for ${ticketId}...\n\nInitializing automation framework...`,
      timestamp: new Date(),
      isRunning: true,
      executionProgress: 0
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
    const botMessage: ChatMessage = {
      type: 'bot',
      text: `🔄 Rerunning test for ${ticketId} using existing script...\n\nExecuting saved test script...`,
      timestamp: new Date(),
      isRunning: true,
      executionProgress: 0
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

  cancelEdit(index: number, event: Event) {
    event.stopPropagation();
    const target = event.target as HTMLElement;
    
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

  // onConfirm(dialog: HTMLDialogElement, event: Event) {
  //   event.stopPropagation();
    
  //   if (this.selectedIndex !== null) {
  //     const sessionToDelete = this.chatHistories[this.selectedIndex];
  //     const sessionId = sessionToDelete.id;
      
  //     this.deletingSessionId = sessionId;
      
  //     setTimeout(() => {
  //       this.chatHistories.splice(this.selectedIndex!, 1);
        
  //       if (this.currentChatId === sessionId) {
  //         this.currentChatMessages = [];
  //         this.currentChatId = null;
  //       }
        
  //       this.saveToLocalStorage();
  //       this.deletingSessionId = null;
        
  //       dialog.close();
  //       this.selectedIndex = null;
  //     }, 500);
  //   }
  // }
// Update your saveEdit method in test-generation.component.ts

saveEdit(index: number, event: Event) {
  event.stopPropagation();
  const target = event.target as HTMLElement;
  const newTitle = target.innerText.trim();

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
      
      // Show error message (you can use a toast/snackbar service)
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

    this.currentChatMessages = [];
    this.currentChatId = null;
    this.activeTab = 'tab1';
    this.loading = false;
  }

  switchTab(tab: string) {
    this.activeTab = tab;
    if (tab === 'tab1') {
      this.startNewChat();
    }
  }

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
