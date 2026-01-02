import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-onboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './onboard.component.html',
  styleUrls: ['./onboard.component.scss']
})
export class OnboardProductComponent {
  productName = '';
  ntid = '';
  products: Array<{name: string, ntid: string}> = [];

  addProduct() {
    if (this.productName && this.ntid) {
      this.products.push({
        name: this.productName,
        ntid: this.ntid
      });
      this.productName = '';
      this.ntid = '';
    }
  }
}