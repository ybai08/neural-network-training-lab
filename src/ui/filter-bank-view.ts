// Renders a filter bank as a horizontal strip of single-channel kernels,
// OR for multi-input-channel conv layers, as one row per input channel.
//
// Conv1 has 4 filters × 1 input channel × 3×3 → 4 mini-grids in a row.
// Conv2 has 8 filters × 4 input channels × 3×3 → 8 columns × 4 rows of mini-grids.
//
// Storage matches convnet.ts: K[(oc · IC + ic) · KH + ky) · KW + kx].

import { cellColor, formatCellNum, formatNum } from './matrix-view';

export interface FilterBankViewOptions {
  cellSize?: number;
  gap?: number;
  showOutputLabel?: boolean;
  inputLabelPrefix?: string;
}

export class FilterBankView {
  readonly element: HTMLElement;
  private cellSize: number;
  private gap: number;
  private showOutputLabel: boolean;
  private inputLabelPrefix: string;
  private highlighted: FilterBankCell | null = null;

  constructor(opts: FilterBankViewOptions = {}) {
    this.cellSize = opts.cellSize ?? 14;
    this.gap = opts.gap ?? 8;
    this.showOutputLabel = opts.showOutputLabel ?? true;
    this.inputLabelPrefix = opts.inputLabelPrefix ?? 'in';

    this.element = document.createElement('div');
    this.element.className = 'filter-bank';
    this.element.style.display = 'flex';
    this.element.style.flexDirection = 'column';
    this.element.style.gap = `${this.gap / 2}px`;
    this.element.addEventListener('click', (e) => {
      const t = e.target as HTMLElement;
      const index = t.dataset.index;
      if (index == null) return;
      this.element.dispatchEvent(new CustomEvent('cell-click', {
        detail: {
          index: +index,
          outChannel: +(t.dataset.outChannel ?? 0),
          inChannel: +(t.dataset.inChannel ?? 0),
          ky: +(t.dataset.ky ?? 0),
          kx: +(t.dataset.kx ?? 0),
        },
        bubbles: true,
      }));
    });
  }

  setOptions(opts: FilterBankViewOptions): void {
    if (opts.cellSize != null) this.cellSize = opts.cellSize;
    if (opts.gap != null) {
      this.gap = opts.gap;
      this.element.style.gap = `${this.gap / 2}px`;
    }
    if (opts.showOutputLabel !== undefined) this.showOutputLabel = opts.showOutputLabel;
    if (opts.inputLabelPrefix !== undefined) this.inputLabelPrefix = opts.inputLabelPrefix;
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
      if (this.showOutputLabel) html.push(`<div class="fb-rowlabel">${inC > 1 ? `${this.inputLabelPrefix} ${ic}` : ''}</div>`);
      for (let oc = 0; oc < outC; oc++) {
        const off = (oc * inC + ic) * kH * kW;
        html.push(`<div class="fb-grid" style="width:${gridW}px;height:${gridH}px;">`);
        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            const v = kernels[off + ky * kW + kx];
            const index = off + ky * kW + kx;
            const isHl = sameFilterCell(this.highlighted, { outChannel: oc, inChannel: ic, ky, kx });
            const text = this.cellSize >= 18 ? formatCellNum(v, this.cellSize) : '';
            html.push(
              `<div class="fb-cell mv-cell${isHl ? ' mv-cell-hl' : ''}" ` +
              `data-index="${index}" data-out-channel="${oc}" data-in-channel="${ic}" data-ky="${ky}" data-kx="${kx}" ` +
              `style="left:${kx * this.cellSize}px;top:${ky * this.cellSize}px;` +
              `width:${this.cellSize}px;height:${this.cellSize}px;background:${cellColor(v, scale)};" title="${formatNum(v)}">${text}</div>`,
            );
          }
        }
        html.push(`</div>`);
      }
      html.push(`</div>`);
    }
    this.element.innerHTML = html.join('');
  }

  /** Outline one filter weight cell in yellow. Pass null to clear. */
  setHighlight(cell: FilterBankCell | null): void {
    if (sameFilterCell(cell, this.highlighted)) return;
    this.highlighted = cell;
    this.element.querySelectorAll<HTMLElement>('.fb-cell.mv-cell-hl').forEach(el => el.classList.remove('mv-cell-hl'));
    if (cell) {
      const selector =
        `.fb-cell[data-out-channel="${cell.outChannel}"][data-in-channel="${cell.inChannel}"]` +
        `[data-ky="${cell.ky}"][data-kx="${cell.kx}"]`;
      const el = this.element.querySelector<HTMLElement>(selector);
      if (el) el.classList.add('mv-cell-hl');
    }
  }
}

export interface FilterBankCell {
  outChannel: number;
  inChannel: number;
  ky: number;
  kx: number;
}

function sameFilterCell(a: FilterBankCell | null, b: FilterBankCell | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.outChannel === b.outChannel && a.inChannel === b.inChannel && a.ky === b.ky && a.kx === b.kx;
}
