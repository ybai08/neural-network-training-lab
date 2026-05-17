// Renders a filter bank as a horizontal strip of single-channel kernels,
// OR for multi-input-channel conv layers, as one row per input channel.
//
// Conv1 has 4 filters × 1 input channel × 3×3 → 4 mini-grids in a row.
// Conv2 has 8 filters × 4 input channels × 3×3 → 8 columns × 4 rows of mini-grids.
//
// Storage matches convnet.ts: K[(oc · IC + ic) · KH + ky) · KW + kx].

import { cellColor } from './matrix-view';

export interface FilterBankViewOptions {
  cellSize?: number;
  gap?: number;
  showOutputLabel?: boolean;
}

export class FilterBankView {
  readonly element: HTMLElement;
  private cellSize: number;
  private gap: number;
  private showOutputLabel: boolean;

  constructor(opts: FilterBankViewOptions = {}) {
    this.cellSize = opts.cellSize ?? 14;
    this.gap = opts.gap ?? 8;
    this.showOutputLabel = opts.showOutputLabel ?? true;

    this.element = document.createElement('div');
    this.element.className = 'filter-bank';
    this.element.style.display = 'flex';
    this.element.style.flexDirection = 'column';
    this.element.style.gap = `${this.gap / 2}px`;
  }

  /** Update with a flat 4D kernel tensor of shape (outC × inC × kH × kW). */
  update(kernels: Float64Array | null, outC: number, inC: number, kH: number, kW: number): void {
    if (!kernels) { this.element.innerHTML = ''; return; }
    // Shared scale so colour intensity is comparable across filters.
    let scale = 1e-6;
    for (let i = 0; i < kernels.length; i++) {
      const a = Math.abs(kernels[i]);
      if (a > scale) scale = a;
    }

    // Layout: outC columns (each output filter gets a column);
    //         inC rows (one row per input channel).
    // For inC = 1 (Conv1) this collapses to a single horizontal strip.
    const gridW = kW * this.cellSize;
    const gridH = kH * this.cellSize;
    const html: string[] = [];
    html.push(`<div class="fb-row">`);
    if (this.showOutputLabel) {
      // Header row: filter indices.
      html.push(`<div class="fb-corner"></div>`);
      for (let oc = 0; oc < outC; oc++) html.push(`<div class="fb-collabel" style="width:${gridW}px;">${oc}</div>`);
    }
    html.push(`</div>`);

    for (let ic = 0; ic < inC; ic++) {
      html.push(`<div class="fb-row">`);
      if (this.showOutputLabel) html.push(`<div class="fb-rowlabel">${inC > 1 ? `in ${ic}` : ''}</div>`);
      for (let oc = 0; oc < outC; oc++) {
        const off = (oc * inC + ic) * kH * kW;
        html.push(`<div class="fb-grid" style="width:${gridW}px;height:${gridH}px;">`);
        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            const v = kernels[off + ky * kW + kx];
            html.push(`<div class="fb-cell" style="left:${kx * this.cellSize}px;top:${ky * this.cellSize}px;width:${this.cellSize}px;height:${this.cellSize}px;background:${cellColor(v, scale)};"></div>`);
          }
        }
        html.push(`</div>`);
      }
      html.push(`</div>`);
    }
    this.element.innerHTML = html.join('');
  }
}
