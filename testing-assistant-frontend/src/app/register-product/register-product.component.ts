import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';

interface Project {
  id: string;
  name: string;
}

interface Tester {
  id: string;
  name: string;
  email: string;
}

interface UploadedFile {
  name: string;
  type: string;
  size: number;
  uploadDate: Date;
}

@Component({
  selector: 'app-register-product',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatSelectModule,
    MatFormFieldModule,
    MatButtonModule,
    MatIconModule,
    MatInputModule
  ],
  templateUrl: './register-product.component.html',
  styleUrls: ['./register-product.component.scss']
})
export class RegisterProductComponent implements OnInit {
  projects: Project[] = [];
  selectedProject: string = '';
  
  availableTesters: Tester[] = [
    { id: '1', name: 'John Doe', email: 'john@example.com' },
    { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
    { id: '3', name: 'Mike Johnson', email: 'mike@example.com' },
    { id: '4', name: 'Sarah Williams', email: 'sarah@example.com' }
  ];
  
  assignedTesters: Tester[] = [];
  selectedTesterToAdd: string = '';
  
  uploadedFiles: UploadedFile[] = [];
  
  configFile: File | null = null;
  jiraTokenFile: File | null = null;
  zipFile: File | null = null;

  ngOnInit() {
    this.loadProjects();
  }

  loadProjects() {
    this.projects = [
      { id: '1', name: 'PLCD' },
      { id: '2', name: 'ITERL' },
      { id: '3', name: 'Manufacturing Suite' },
      { id: '4', name: 'Quality Management' }
    ];
  }

  onProjectChange() {
    this.loadProjectTesters();
    this.loadProjectFiles();
  }

  loadProjectTesters() {
    this.assignedTesters = [
      { id: '1', name: 'John Doe', email: 'john@example.com' }
    ];
  }

  loadProjectFiles() {
    this.uploadedFiles = [];
  }

  addTester() {
    if (this.selectedTesterToAdd) {
      const tester = this.availableTesters.find(t => t.id === this.selectedTesterToAdd);
      if (tester && !this.assignedTesters.find(t => t.id === tester.id)) {
        this.assignedTesters.push(tester);
        this.selectedTesterToAdd = '';
      }
    }
  }

  removeTester(testerId: string) {
    this.assignedTesters = this.assignedTesters.filter(t => t.id !== testerId);
  }

  onFileSelected(event: any, fileType: string) {
    const file = event.target.files[0];
    if (file) {
      switch(fileType) {
        case 'config':
          this.configFile = file;
          break;
        case 'jira':
          this.jiraTokenFile = file;
          break;
        case 'zip':
          this.zipFile = file;
          break;
      }
      
      this.uploadedFiles.push({
        name: file.name,
        type: fileType,
        size: file.size,
        uploadDate: new Date()
      });
    }
  }

  removeFile(fileName: string) {
    this.uploadedFiles = this.uploadedFiles.filter(f => f.name !== fileName);
    
    if (this.configFile?.name === fileName) this.configFile = null;
    if (this.jiraTokenFile?.name === fileName) this.jiraTokenFile = null;
    if (this.zipFile?.name === fileName) this.zipFile = null;
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }

  saveRegistration() {
    const registration = {
      projectId: this.selectedProject,
      testers: this.assignedTesters.map(t => t.id),
      files: {
        config: this.configFile?.name,
        jiraToken: this.jiraTokenFile?.name,
        zip: this.zipFile?.name
      }
    };
    
    console.log('Saving registration:', registration);
    alert('Registration saved successfully!');
  }

  getAvailableTestersToAdd(): Tester[] {
    return this.availableTesters.filter(
      t => !this.assignedTesters.find(at => at.id === t.id)
    );
  }
}