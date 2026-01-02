import { Component, Inject, OnInit } from '@angular/core';
import { Router, RouterOutlet } from '@angular/router';
import { SidebarComponent } from './sidebar/sidebar.component';
import { SidebarNavItem } from './models/sidebar-nav-item.model';
import { AuthService, UserRole } from './auth.service';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';

import {
  MsalService,
  MsalModule,
  MsalBroadcastService,
  MSAL_GUARD_CONFIG,
  MsalGuardConfiguration,
} from '@azure/msal-angular';
import {
  AuthenticationResult,
  InteractionStatus,
  PopupRequest,
  RedirectRequest,
  EventMessage,
  EventType,
} from '@azure/msal-browser';
import { Subject } from 'rxjs';
import { filter, takeUntil } from 'rxjs/operators';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, SidebarComponent, CommonModule, MatIconModule,MsalModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'Testing Assistant';
  isIframe = false;
  loginDisplay = false;
  private readonly _destroying$ = new Subject<void>();
  applicationTitle = 'Testing Assistant';
  currentUser = '';
  sidebarLinks: SidebarNavItem[] = [];

  constructor(
    private authService: AuthService,
    @Inject(MSAL_GUARD_CONFIG) private msalGuardConfig: MsalGuardConfiguration,
    private msalService: MsalService,
    private msalBroadcastService: MsalBroadcastService,
    private router: Router
  ) {
    console.log('AppComponent constructor');
  }

  ngOnInit() {
    console.log('AppComponent ngOnInit');
    this.msalService.handleRedirectObservable().subscribe();
    
    this.isIframe = window !== window.parent && !window.opener; // Remove this line to use Angular Universal
    this.authService.setLoginDisplay();
    this.msalService.instance.enableAccountStorageEvents(); // Optional - This will enable ACCOUNT_ADDED and ACCOUNT_REMOVED events emitted when a user logs in or out of another tab or window
    this.msalBroadcastService.msalSubject$
      .pipe(
        filter(
          (msg: EventMessage) =>
            msg.eventType === EventType.ACCOUNT_ADDED ||
            msg.eventType === EventType.ACCOUNT_REMOVED
        )
      )
      .subscribe((result: EventMessage) => {
        if (this.msalService.instance.getAllAccounts().length === 0) {
          window.location.pathname = '/';
        } else {
          this.setLoginDisplay();
        }
      });

    this.msalBroadcastService.inProgress$
      .pipe(
        filter(
          (status: InteractionStatus) => status === InteractionStatus.None
        ),
        takeUntil(this._destroying$)
      )
      .subscribe(() => {
        this.setLoginDisplay();
        this.checkAndSetActiveAccount();
      });
    
    // if (!this.authService.getCurrentUser()) {
    //   this.authService.login('Tester1', 'tester');
    // }

    // this.authService.currentUser$.subscribe(user => {
    //   console.log('User changed:', user);
    //   if (user) {
    //     this.currentUser = user.username;
    //     this.applicationTitle = this.getTitleForRole(user.role);
    //     this.updateSidebarLinks(user.role);
    //   }
    // });
  }
    setLoginDisplay() {
    this.loginDisplay = this.msalService.instance.getAllAccounts().length > 0;
  }

  checkAndSetActiveAccount() {

    let activeAccount = this.msalService.instance.getActiveAccount();

    if (
      !activeAccount &&
      this.msalService.instance.getAllAccounts().length > 0
    ) {
      let accounts = this.msalService.instance.getAllAccounts();
      this.msalService.instance.setActiveAccount(accounts[0]);
    }
  }

  loginRedirect() {
    if (this.msalGuardConfig.authRequest) {
      this.msalService.loginRedirect({
        ...this.msalGuardConfig.authRequest,
      } as RedirectRequest);
    } else {
      this.authService.loginRedirect();
    }
  }

  loginPopup() {
    if (this.msalGuardConfig.authRequest) {
      this.msalService
        .loginPopup({ ...this.msalGuardConfig.authRequest } as PopupRequest)
        .subscribe((response: AuthenticationResult) => {
          this.msalService.instance.setActiveAccount(response.account);
        });
    } else {
      this.msalService
        .loginPopup()
        .subscribe((response: AuthenticationResult) => {
          this.msalService.instance.setActiveAccount(response.account);
        });
    }
  }

  logout(popup?: boolean) {
    if (popup) {
      this.msalService.logoutPopup({
        mainWindowRedirectUri: '/',
      });
    } else {
      this.msalService.logoutRedirect();
    }
  }

  ngOnDestroy(): void {
    this._destroying$.next(undefined);
    this._destroying$.complete();
  }

  getTitleForRole(role: UserRole): string {
    switch(role) {
      case 'admin': return 'Test Assistant Admin View';
      case 'po': return 'Product Owner View';
      case 'tester': return 'Testing Assistant User';
      default: return 'Testing Assistant';
    }
  }

  getDefaultRouteForRole(role: UserRole): string {
    switch(role) {
      case 'admin': return '/dashboard';
      case 'po': return '/register-product';
      case 'tester': return '/testing-assistant';
      default: return '/testing-assistant';
    }
  }

  switchRole(role: UserRole) {
    console.log('Switching to role:', role);
    const username = role === 'admin' ? 'Admin User' : 
                     role === 'po' ? 'PO User' : 'Tester1';
    
    // Save the role and navigate
    // this.authService.loginRedirect()
    const route = this.getDefaultRouteForRole(role);
    
    // Navigate then reload
    this.router.navigate([route]).then(() => {
      window.location.reload();
    });
  }

  private updateSidebarLinks(role: UserRole) {
    console.log('Updating sidebar for role:', role);
    
    if (role === 'admin') {
      this.sidebarLinks = [
        { id: 'dashboard', title: 'Dashboard', url: '/dashboard', icon: 'dashboard', active: false },
        { id: 'onboard-product', title: 'Onboard New Product', url: '/onboard-product', icon: 'add_circle', active: false }
      ];
    } else if (role === 'po') {
      this.sidebarLinks = [
        { id: 'register-product', title: 'Register Product', url: '/register-product', icon: 'app_registration', active: false }
      ];
    } else {
      this.sidebarLinks = [
        { id: 'testing-assistant', title: 'Testing Assistant', url: '/testing-assistant', icon: 'science', active: false }
      ];
    }
    
    console.log('Sidebar links updated:', this.sidebarLinks);
  }
}