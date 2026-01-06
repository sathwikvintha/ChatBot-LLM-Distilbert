import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  // ✅ Inline template instead of missing app.html
  template: `
    <main class="app-main">
      <router-outlet></router-outlet>
    </main>
  `,
  styleUrls: ['./app.scss'],
})
export class AppComponent {
  title = 'Org Docs Chatbot';
}
