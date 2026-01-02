import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TestGenerationComponent } from './test-generation.component';

describe('TestGenerationComponent', () => {
  let component: TestGenerationComponent;
  let fixture: ComponentFixture<TestGenerationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestGenerationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TestGenerationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
