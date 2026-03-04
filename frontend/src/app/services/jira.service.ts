import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class JiraService {
  // private apiUrl = 'http://localhost:8000/jira'; // Your FastAPI endpoint
  //private apiUrl = `${window.location.origin}/jira`; // Dynamic URL based on current origin
  private apiUrl = '/api/jira'; // Proxy configuration in Angular
  constructor(private http: HttpClient) {}

  getTicket(ticketId: string) {
    return this.http.get(`${this.apiUrl}/${ticketId}`);
  }
}