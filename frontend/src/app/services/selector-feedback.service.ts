import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class SelectorFeedbackService {
  private apiUrl = 'http://localhost:8000/api/selector-feedback';
  constructor(private http: HttpClient) {}
  submitFeedback(data: any): Observable<any> {
    return this.http.post(this.apiUrl, data);
  }
}