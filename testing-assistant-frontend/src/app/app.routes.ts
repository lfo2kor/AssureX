import { Routes } from '@angular/router';
import { RegisterProductComponent } from './register-product/register-product.component';
import { TestGenerationComponent } from './test-generation/test-generation.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { OnboardProductComponent } from './onboard/onboard.component';
import { MsalGuard } from '@azure/msal-angular';

export const routes: Routes = [
  { 
    path: '', 
    redirectTo: 'testing-assistant', 
    pathMatch: 'full' 
  },
  { 
    path: 'dashboard', 
    component: DashboardComponent,
    canActivate: [MsalGuard],
    // Remove canActivate temporarily
  },
  { 
    path: 'onboard-product', 
    component: OnboardProductComponent,
    canActivate: [MsalGuard]
  },
  { 
    path: 'testing-assistant', 
    component: TestGenerationComponent,
    canActivate: [MsalGuard]
  },
  { 
    path: 'register-product', 
    component: RegisterProductComponent,
    canActivate: [MsalGuard],
  }
];