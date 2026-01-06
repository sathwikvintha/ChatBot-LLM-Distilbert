import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiUrl = 'http://127.0.0.1:8000/chat';

  constructor(private http: HttpClient) {}

  askQuestion(question: string, top_k: number = 5): Observable<any> {
    return this.http.post<any>(this.apiUrl, { question, top_k });
  }
}
