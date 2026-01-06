import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { ChatComponent } from './chat';

describe('ChatComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatComponent, HttpClientTestingModule, FormsModule],
    }).compileComponents();
  });

  it('should create the component', () => {
    const fixture = TestBed.createComponent(ChatComponent);
    const component = fixture.componentInstance;
    expect(component).toBeTruthy();
  });

  it('should have empty initial state', () => {
    const fixture = TestBed.createComponent(ChatComponent);
    const component = fixture.componentInstance;
    expect(component.question).toBe('');
    expect(component.answer).toBe('');
    expect(component.citations.length).toBe(0);
  });
});
