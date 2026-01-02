import { Component, Input, signal, computed, HostListener, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatBadgeModule } from '@angular/material/badge';
import { SidebarNavItem } from '../models/sidebar-nav-item.model';
import { AuthService } from '../auth.service';
import { Subject, takeUntil } from 'rxjs';


@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [
    CommonModule,
    MatSidenavModule,
    MatListModule,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
    MatBadgeModule
  ],
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent implements OnInit{
  @Input() applicationTitle: string = 'Application';
  @Input() currentUser: string = 'User';
  username = '';
  ntid: string = '';
  department: string = '';
  @Input() sidebarLinks: SidebarNavItem[] = [];
  private destroy$ = new Subject<void>();

  private _isOpen = signal(true);
  private _isMobile = signal(false);

  isOpen = this._isOpen.asReadonly();
  
  sideNavMode = computed(() => this._isMobile() ? 'over' : 'side');
  isInMobileMode = computed(() => this._isMobile());

  constructor(private router: Router, private authService: AuthService) {
    // this.getuserId()
    this.checkScreenSize();
  }

  ngOnInit(): void {
    this.authService.userId$
    .pipe(takeUntil(this.destroy$))
    .subscribe((ntid) => {

      if (ntid) {
        const userId = this.authService.getUserId() || '';

        this.ntid = ntid.toUpperCase();
        this.username = userId.split('(')[0].trim();
        this.department = userId.match(/\(([^)]+)\)/)?.[1] || '';
      }
    });
  }

  @HostListener('window:resize', ['$event'])
  onResize() {
    this.checkScreenSize();
  }

  private checkScreenSize() {
    this._isMobile.set(window.innerWidth < 768);
    if (this._isMobile()) {
      this._isOpen.set(false);
    }
  }

  toggleSideNavigationBar() {
    this._isOpen.set(!this._isOpen());
  }
  

  // getuserId(){
  //   this.username = this.authService.getUserId();
  // }


  openSidebar() {
    this._isOpen.set(true);
  }

  toggleExpanded(item: SidebarNavItem) {
    if (item.items && item.items.length > 0) {
      if (!this._isOpen()) {
        this._isOpen.set(true);
      }
      item.expanded = !item.expanded;
    }
  }

  navigateToLink(item: SidebarNavItem) {
    if (item.url) {
      this.sidebarLinks.forEach(link => {
        link.active = false;
        if (link.items) {
          link.items.forEach(child => child.active = false);
        }
      });
      item.active = true;
      
      this.router.navigate([item.url]);
      
      if (this._isMobile()) {
        this._isOpen.set(false);
      }
    }
  }

  logout() {
    this.authService.logout();
    // this.router.navigate(['/testing-assistant']);
  }
  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

}