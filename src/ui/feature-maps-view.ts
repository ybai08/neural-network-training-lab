// Renders a (channels × H × W) tensor as a horizontal strip of small heatmaps,
// one per channel. Used for every Conv / Pool / ReLU output in the CNN page.
//
// Storage convention matches convnet.ts: tensor[c · H · W + y · W + x].

import { cellColor } from './matrix-view';

export interface FeatureMapsViewOptions {
  cellSize?: number;    // pixel side of each individual cell
  gap?: number;         // horizontal gap between feature maps
  showIndices?: boolean;
}

export class FeatureMapsView {
  readonly element: HTMLElement;
  private cellSize: number;
  private gap: number;
  private showIndices: boolean;

  constructor(opts: FeatureMapsViewOptions = {}) {
    this.cellSize = opts.cellSize ?? 12;
    this.gap = opts.gap ?? 8;
    this.showIndices = opts.showIndices ?? true;

    this.element = document.createElement('div');
    this.element.className = 'feature-maps';
    this.element.style.display = 'flex';
    this.element.style.flexWrap = 'wrap';
    this.element.style.gap = `${this.gap}px`;
  }

  /** Update with a flat channel-major Float64Array of shape (channels × H × W). */
  update(data: Float64Array | null, channels: number, H: number, W: number): void {
    if (!data) {
      this.element.innerHTML = '';
      return;
    }
    // Single shared scale across all channels — using max|v| over the whole
    // tensor — so colour intensity is comparable between feature maps.
    let scale = 1e-6;
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]);
      if (a > scale) scale = a;
    }

    const html: string[] = [];
    for (let c = 0; c < channels; c++) {
      const stride = c * H * W;
      html.push(`<div class="fm-block">`);
      if (this.showIndices) html.push(`<div class="fm-label">${c}</div>`);
      html.push(`<div class="fm-grid" style="width:${W * this.cellSize}px;height:${H * this.cellSize}px;">`);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = data[stride + y * W + x];
          html.push(`<div class="fm-cell" style="left:${x * this.cellSize}px;top:${y * this.cellSize}px;width:${this.cellSize}px;height:${this.cellSize}px;background:${cellColor(v, scale)};"></div>`);
        }
      }
      html.push(`</div></div>`);
    }
    this.element.innerHTML = html.join('');
  }
}
