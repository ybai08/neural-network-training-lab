// Tiny per-epoch line chart. SVG polyline — zero deps, zero canvas state to
// maintain, scales cleanly. Used for Test-set Loss and Accuracy curves.

export class LineChart {
  readonly element: HTMLElement;
  private label: HTMLElement;
  private valueEl: HTMLElement;
  private svg: SVGSVGElement;
  private path: SVGPolylineElement;
  private markers: SVGGElement;
  private placeholder: HTMLElement;
  private values: number[] = [];
  private color: string;
  private fixedRange: { min: number; max: number } | null;

  constructor(opts: { title: string; color: string; fixedRange?: { min: number; max: number } }) {
    this.color = opts.color;
    this.fixedRange = opts.fixedRange ?? null;

    this.element = document.createElement('div');
    this.element.className = 'chart';

    const header = document.createElement('div');
    header.className = 'chart-header';
    this.label = document.createElement('span');
    this.label.innerHTML = opts.title;
    renderMath(this.label);
    this.valueEl = document.createElement('span');
    this.valueEl.className = 'chart-value';
    header.appendChild(this.label);
    header.appendChild(this.valueEl);
    this.element.appendChild(header);

    this.placeholder = document.createElement('div');
    this.placeholder.className = 'chart-placeholder';
    this.placeholder.textContent = 'Graph starts after 50 training samples.';

    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('preserveAspectRatio', 'none');
    this.svg.style.width = '100%';
    this.svg.style.height = 'calc(100% - 22px)';
    this.svg.style.display = 'block';
    this.path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline') as SVGPolylineElement;
    this.path.setAttribute('fill', 'none');
    this.path.setAttribute('stroke', this.color);
    this.path.setAttribute('stroke-width', '2');
    this.svg.appendChild(this.path);
    this.markers = document.createElementNS('http://www.w3.org/2000/svg', 'g') as SVGGElement;
    this.svg.appendChild(this.markers);
    this.element.appendChild(this.placeholder);
    this.element.appendChild(this.svg);

    new ResizeObserver(() => this.redraw()).observe(this.element);
  }

  push(pointLabel: string, value: number, displayValue: string): void {
    this.values.push(value);
    this.valueEl.textContent = `${pointLabel}: ${displayValue}`;
    this.placeholder.textContent = this.values.length < 2 ? 'First checkpoint plotted. One more checkpoint will show the trend.' : '';
    this.redraw();
  }

  clear(): void {
    this.values = [];
    this.valueEl.textContent = '';
    this.placeholder.textContent = 'Graph starts after 50 training samples.';
    this.path.setAttribute('points', '');
    this.markers.replaceChildren();
  }

  private redraw(): void {
    const w = this.svg.clientWidth;
    const h = this.svg.clientHeight;
    if (w <= 0 || h <= 0 || this.values.length === 0) {
      this.path.setAttribute('points', '');
      this.markers.replaceChildren();
      return;
    }
    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    let yMin: number, yMax: number;
    if (this.fixedRange) {
      yMin = this.fixedRange.min;
      yMax = this.fixedRange.max;
    } else {
      yMin = Number.POSITIVE_INFINITY;
      yMax = Number.NEGATIVE_INFINITY;
      for (const v of this.values) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
      const pad = Math.max(1e-6, (yMax - yMin) * 0.1);
      yMin -= pad; yMax += pad;
    }
    const range = Math.max(1e-9, yMax - yMin);
    const xStep = this.values.length <= 1 ? 0 : w / (this.values.length - 1);
    const pts: string[] = [];
    const dots: SVGCircleElement[] = [];
    for (let i = 0; i < this.values.length; i++) {
      const x = this.values.length <= 1 ? w / 2 : i * xStep;
      const y = h - ((this.values[i] - yMin) / range) * h;
      pts.push(`${x},${y}`);
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle') as SVGCircleElement;
      dot.setAttribute('cx', `${x}`);
      dot.setAttribute('cy', `${y}`);
      dot.setAttribute('r', this.values.length <= 1 || i === this.values.length - 1 ? '3.5' : '2.2');
      dot.setAttribute('fill', this.color);
      dots.push(dot);
    }
    this.path.setAttribute('points', pts.join(' '));
    this.markers.replaceChildren(...dots);
  }
}
import { renderMath } from './katex-util';
