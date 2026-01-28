import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class JiraService {
  private apiUrl = 'http://localhost:8000/jira'; // Your FastAPI endpoint

  constructor(private http: HttpClient) {}

  getTicket(ticketId: string) {
    return this.http.get(`${this.apiUrl}/${ticketId}`);
  }
}