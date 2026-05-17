// Single-column DOM heatmap, used for y / z[2] / a[2] / σ'(z[2]) / a[2]−y / δ[2]
// and for the bias vectors b[2] / Δb[2]. Click events bubble up as
// CustomEvent('cell-click', { detail: { index } }).

import { cellColor, formatNum } from './matrix-view';

export interface VectorViewOptions {
  cellWidth?: number;
  cellHeight?: number;
  title?: string;
  fixedScale?: number;
}

export class VectorView {
  readonly element: HTMLElement;
  private cellWidth: number;
  private cellHeight: number;
  private title: string | undefined;
  private fixedScale: number | undefined;
  private highlighted: number | null = null;
  private size = 0;

  constructor(opts: VectorViewOptions = {}) {
    this.cellWidth = opts.cellWidth ?? 56;
    this.cellHeight = opts.cellHeight ?? 22;
    this.title = opts.title;
    this.fixedScale = opts.fixedScale;
    this.element = document.createElement('div');
    this.element.className = 'vector-view';
    this.element.addEventListener('click', (e) => {
      const t = e.target as HTMLElement;
      const i = t.dataset.index;
      if (i != null) {
        this.element.dispatchEvent(new CustomEvent('cell-click', {
          detail: { index: +i }, bubbles: true,
        }));
      }
    });
  }

  setOptions(opts: VectorViewOptions): void {
    if (opts.cellWidth != null) this.cellWidth = opts.cellWidth;
    if (opts.cellHeight != null) this.cellHeight = opts.cellHeight;
    if (opts.title !== undefined) this.title = opts.title;
    if (opts.fixedScale !== undefined) this.fixedScale = opts.fixedScale;
  }

  update(data: Float64Array | null, size?: number): void {
    if (size != null) this.size = size;
    if (!data) {
      this.element.innerHTML = `<div class="vv-grid vv-empty" style="width:${this.cellWidth}px;height:${(this.title ? 14 : 0) + this.size * this.cellHeight}px;"></div>`;
      return;
    }
    this.size = data.length;
    const scale = this.fixedScale ?? Math.max(1e-6, maxAbs(data));
    const titleH = this.title ? 14 : 0;
    const html: string[] = [];
    html.push(`<div class="vv-grid" style="position:relative;width:${this.cellWidth}px;height:${titleH + data.length * this.cellHeight}px;">`);
    if (this.title) html.push(`<div class="vv-title" style="height:${titleH}px;width:${this.cellWidth}px;">${this.title}</div>`);
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const color = cellColor(v, scale);
      const isHl = this.highlighted === i;
      html.push(
        `<div class="vv-cell${isHl ? ' vv-cell-hl' : ''}" data-index="${i}" ` +
        `style="top:${titleH + i * this.cellHeight}px;width:${this.cellWidth}px;height:${this.cellHeight}px;background:${color};">${formatNum(v)}</div>`,
      );
    }
    html.push('</div>');
    this.element.innerHTML = html.join('');
  }

  setHighlight(index: number | null): void {
    if (this.highlighted === index) return;
    this.highlighted = index;
    this.element.querySelectorAll<HTMLElement>('.vv-cell.vv-cell-hl').forEach(el => el.classList.remove('vv-cell-hl'));
    if (index != null) {
      const el = this.element.querySelector<HTMLElement>(`.vv-cell[data-index="${index}"]`);
      if (el) el.classList.add('vv-cell-hl');
    }
  }
}

function maxAbs(d: Float64Array): number {
  let m = 0;
  for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > m) m = a; }
  return m;
}
