import { Observable } from 'rxjs';

export interface SidebarNavItem {
  id: string;
  title: string;
  url?: string;
  icon?: string;
  svgIcon?: string;
  iconFont?: string;
  iconBadgeText?: Observable<string>;
  iconBadgeColor?: Observable<string>;
  active?: boolean;
  expanded?: boolean;
  items?: SidebarNavItem[];
  routerLinkActiveOptions?: { exact: boolean };
}