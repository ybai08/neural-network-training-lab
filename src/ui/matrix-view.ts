// DOM-based heatmap view of a row-major Float64Array shaped (rows × cols).
// Each cell is a <div> with a background-color computed from sign+magnitude.
// Click events bubble up as CustomEvent('cell-click', { detail: {row, col} }).
//
// Why DOM over Canvas: the matrices we display are small (≤ 300 cells), and
// DOM gives us hover/tooltips and click hit-testing for free.

export interface MatrixViewOptions {
  cellWidth?: number;
  cellHeight?: number;
  rowLabelPrefix?: string;
  colLabelPrefix?: string;
  colLabelEvery?: number;
  /** Fixed denominator for the colour-alpha calc; leave undefined to auto-scale to max|v|. */
  fixedScale?: number;
}

export class MatrixView {
  readonly element: HTMLElement;
  private rows = 0;
  private cols = 0;
  private cellWidth: number;
  private cellHeight: number;
  private rowLabelPrefix: string | undefined;
  private colLabelPrefix: string | undefined;
  private colLabelEvery: number;
  private fixedScale: number | undefined;
  private highlighted: { row: number; col: number } | null = null;

  constructor(opts: MatrixViewOptions = {}) {
    this.cellWidth = opts.cellWidth ?? 36;
    this.cellHeight = opts.cellHeight ?? 22;
    this.rowLabelPrefix = opts.rowLabelPrefix;
    this.colLabelPrefix = opts.colLabelPrefix;
    this.colLabelEvery = opts.colLabelEvery ?? 1;
    this.fixedScale = opts.fixedScale;

    this.element = document.createElement('div');
    this.element.className = 'matrix-view';
    this.element.addEventListener('click', (e) => {
      const t = e.target as HTMLElement;
      const r = t.dataset.row, c = t.dataset.col;
      if (r != null && c != null) {
        this.element.dispatchEvent(new CustomEvent('cell-click', {
          detail: { row: +r, col: +c }, bubbles: true,
        }));
      }
    });
  }

  setOptions(opts: MatrixViewOptions): void {
    if (opts.cellWidth != null) this.cellWidth = opts.cellWidth;
    if (opts.cellHeight != null) this.cellHeight = opts.cellHeight;
    if (opts.rowLabelPrefix !== undefined) this.rowLabelPrefix = opts.rowLabelPrefix;
    if (opts.colLabelPrefix !== undefined) this.colLabelPrefix = opts.colLabelPrefix;
    if (opts.colLabelEvery != null) this.colLabelEvery = Math.max(1, Math.floor(opts.colLabelEvery));
    if (opts.fixedScale !== undefined) this.fixedScale = opts.fixedScale;
  }

  /** Replace contents. Pass null/undefined to clear. */
  update(data: Float64Array | null, rows: number, cols: number): void {
    this.rows = rows;
    this.cols = cols;
    if (!data) {
      // Render an empty placeholder grid so the parent layout doesn't reflow when data arrives.
      this.renderPlaceholder();
      return;
    }

    const scale = this.fixedScale ?? Math.max(1e-6, maxAbs(data));
    const showText = this.cellWidth >= 18 && this.cellHeight >= 14;

    // Build a single string of HTML for fewer round-trips through layout.
    const labelW = 20, labelH = 14;
    const html: string[] = [];
    html.push(`<div class="mv-grid" style="position:relative;width:${labelW + cols * this.cellWidth}px;height:${labelH + rows * this.cellHeight}px;">`);
    // Column labels.
    const colLabelW = Math.max(this.cellWidth, 22);
    for (let c = 0; c < cols; c++) {
      if (c % this.colLabelEvery !== 0) continue;
      const lbl = this.colLabelPrefix != null ? `${this.colLabelPrefix}${c}` : `${c}`;
      const left = labelW + c * this.cellWidth - (colLabelW - this.cellWidth) / 2;
      html.push(`<div class="mv-collabel" style="left:${left}px;width:${colLabelW}px;height:${labelH}px;">${lbl}</div>`);
    }
    // Row labels + cells.
    for (let r = 0; r < rows; r++) {
      const rlbl = this.rowLabelPrefix != null ? `${this.rowLabelPrefix}${r}` : `${r}`;
      html.push(`<div class="mv-rowlabel" style="top:${labelH + r * this.cellHeight}px;width:${labelW}px;height:${this.cellHeight}px;">${rlbl}</div>`);
      for (let c = 0; c < cols; c++) {
        const v = data[r * cols + c];
        const color = cellColor(v, scale);
        const isHl = this.highlighted && this.highlighted.row === r && this.highlighted.col === c;
        const text = showText ? formatCellNum(v, this.cellWidth) : '';
        html.push(
          `<div class="mv-cell${isHl ? ' mv-cell-hl' : ''}" data-row="${r}" data-col="${c}" ` +
          `style="left:${labelW + c * this.cellWidth}px;top:${labelH + r * this.cellHeight}px;` +
          `width:${this.cellWidth}px;height:${this.cellHeight}px;background:${color};" title="${formatNum(v)}">${text}</div>`,
        );
      }
    }
    html.push('</div>');
    this.element.innerHTML = html.join('');
  }

  /** Outline (row, col) in yellow. Pass null to clear. */
  setHighlight(cell: { row: number; col: number } | null): void {
    if (sameCell(cell, this.highlighted)) return;
    this.highlighted = cell;
    // Cheap: just toggle the class on the affected nodes.
    this.element.querySelectorAll<HTMLElement>('.mv-cell.mv-cell-hl').forEach(el => el.classList.remove('mv-cell-hl'));
    if (cell) {
      const el = this.element.querySelector<HTMLElement>(`.mv-cell[data-row="${cell.row}"][data-col="${cell.col}"]`);
      if (el) el.classList.add('mv-cell-hl');
    }
  }

  private renderPlaceholder(): void {
    const labelW = 20, labelH = 14;
    this.element.innerHTML =
      `<div class="mv-grid mv-empty" style="width:${labelW + this.cols * this.cellWidth}px;height:${labelH + this.rows * this.cellHeight}px;"></div>`;
  }
}

// ── helpers ────────────────────────────────────────────────────────────────

function maxAbs(d: Float64Array): number {
  let m = 0;
  for (let i = 0; i < d.length; i++) {
    const a = Math.abs(d[i]);
    if (a > m) m = a;
  }
  return m;
}

/** Green for positive, red for negative; alpha tracks |v|/scale. */
export function cellColor(v: number, scale: number): string {
  const alpha = clamp(Math.abs(v) / scale, 0.05, 0.9);
  return v >= 0
    ? `rgba(91, 186, 111, ${alpha.toFixed(3)})`
    : `rgba(255, 107, 107, ${alpha.toFixed(3)})`;
}

/** Same tight number formatting MatrixView used in the WPF version. */
export function formatNum(v: number): string {
  if (Math.abs(v) >= 100) return v >= 0 ? `+${v.toFixed(0)}` : `${v.toFixed(0)}`;
  if (Math.abs(v) >= 10)  return v >= 0 ? `+${v.toFixed(1)}` : `${v.toFixed(1)}`;
  if (v > 0) return `+${v.toFixed(2)}`;
  if (v < 0) return v.toFixed(2);
  return ' 0.00';
}

export function formatCellNum(v: number, cellWidth: number): string {
  if (cellWidth >= 24) return formatNum(v);
  if (Math.abs(v) >= 100) return v >= 0 ? `+${v.toFixed(0)}` : `${v.toFixed(0)}`;
  if (Math.abs(v) >= 10) return v >= 0 ? `+${v.toFixed(0)}` : `${v.toFixed(0)}`;
  if (v > 0) return `+${v.toFixed(1)}`;
  if (v < 0) return v.toFixed(1);
  return '0.0';
}

function clamp(v: number, lo: number, hi: number): number { return v < lo ? lo : v > hi ? hi : v; }
function sameCell(a: { row: number; col: number } | null, b: { row: number; col: number } | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.row === b.row && a.col === b.col;
}
