import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class JiraService {
  private apiUrl = 'http://localhost:8000/jira';

  constructor(private http: HttpClient) {}

  getTicket(ticketId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/${ticketId}`);
  }
}