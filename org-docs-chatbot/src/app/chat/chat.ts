// Import Angular core and required modules
import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

// -----------------------------
// Component definition
// -----------------------------
@Component({
  selector: 'app-chat', // Component selector used in HTML
  standalone: true, // Standalone component (no NgModule needed)
  imports: [CommonModule, FormsModule, HttpClientModule], // Modules required by this component
  templateUrl: './chat.html', // HTML template file
  styleUrls: ['./chat.scss'], // SCSS stylesheet file
})
export class ChatComponent {
  // -----------------------------
  // Component state variables
  // -----------------------------
  question = ''; // User's input question
  answer = ''; // Answer returned from backend
  citations: Array<{
    source_path?: string;
    snippet?: string;
    page_number?: number;
    section_heading?: string;
  }> = []; // Citations (sources/snippets) returned from backend
  loading = false; // Flag to show loading spinner
  errorMessage = ''; // Error message if backend call fails

  // Inject HttpClient for API calls and ChangeDetectorRef for manual UI updates
  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {}

  // -----------------------------
  // Method: askQuestion
  // -----------------------------
  askQuestion() {
    const q = this.question?.trim(); // Clean up user input
    if (!q) return; // Do nothing if input is empty

    // Reset state before making request
    this.loading = true;
    this.answer = '';
    this.citations = [];
    this.errorMessage = '';

    // Make POST request to FastAPI backend
    this.http.post<any>('http://127.0.0.1:8000/chat', { question: q, top_k: 8 }).subscribe({
      next: (resp) => {
        // On success: update answer and citations
        this.answer = resp?.answer ?? '';
        this.citations = Array.isArray(resp?.citations) ? resp.citations : [];
        this.loading = false;
        this.cdr.detectChanges(); // Trigger UI update
      },
      error: (err) => {
        // On error: log and show message
        console.error('Error contacting backend:', err);
        this.errorMessage = 'Error contacting backend. Please try again.';
        this.answer = '';
        this.citations = [];
        this.loading = false;
        this.cdr.detectChanges(); // Trigger UI update
      },
    });
  }

  // -----------------------------
  // Method: openSource
  // -----------------------------
  openSource(citation: any) {
    const url = citation?.source_path; // Get source path from citation
    if (!url) {
      console.warn('No source_path found for citation:', citation);
      return;
    }
    // Open source file in new browser tab
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  // -----------------------------
  // Method: citationTitle
  // -----------------------------
  citationTitle(c: any) {
    const parts: string[] = [];
    if (c?.source_path) parts.push(c.source_path); // Add source path
    if (c?.section_heading) parts.push(c.section_heading); // Add section heading
    if (c?.page_number) parts.push(`Page ${c.page_number}`); // Add page number
    // Join parts with separator
    return parts.join(' • ');
  }
}
