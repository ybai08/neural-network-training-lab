// Canvas 2D port of NetworkDiagram.cs. Draws layer columns with neurons and
// inter-layer connections, all phase-aware (neurons coloured by activation
// or δ depending on phase; connections recoloured by gradient at Phase 4,
// or by contribution to the focused hidden neuron at Phase 5).
//
// Hit-testing is done in DOM coordinates by re-running the same geometry the
// renderer uses, so a click on a line emits 'weight-click' and a click on a
// hidden neuron emits 'hidden-click'.

import { Phase } from '../types';

const FULL_DRAW_MAX_COUNT = 200;
const TRUNCATED_VISIBLE_COUNT = 28;
const TRUNCATED_VISIBLE_HALF = TRUNCATED_VISIBLE_COUNT / 2;
const MAX_NEURON_RADIUS = 9;
const MIN_NEURON_RADIUS = 1.5;
const COLUMN_MARGIN = 60;
const TOP_BOTTOM_MARGIN = 16;

interface Position { y: number; actualIndex: number; }

export interface DiagramState {
  weights: Float64Array[];          // [W[1], W[2]]
  sizes: readonly number[];         // [196, 30, 10]
  trueLabel: number;                // -1 if unknown
  predictedLabel: number;
  phase: Phase;
  hiddenActivation: Float64Array | null;
  outputActivation: Float64Array | null;
  hiddenDelta: Float64Array | null;
  outputDelta: Float64Array | null;
  focusedHiddenNeuron: number | null;
  /** (weightLayerIndex, row, col) of the line drawn with a thick yellow highlight. */
  highlightedWeight: { layerIdx: number; row: number; col: number } | null;
}

export class NetworkDiagram {
  readonly element: HTMLElement;
  private canvas: HTMLCanvasElement;
  private dpr = window.devicePixelRatio || 1;
  private state: DiagramState | null = null;
  private cachedPositions: Position[][] | null = null;
  private cachedX: number[] | null = null;

  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'diagram-wrap';
    // The diagram MUST be strictly contained — otherwise its canvas (which has
    // no intrinsic CSS size) can blow up and draw across the entire viewport,
    // covering everything below. position:relative + overflow:hidden gives the
    // absolute-positioned canvas a containing block AND prevents bleed.
    this.element.style.position = 'relative';
    this.element.style.overflow = 'hidden';
    this.element.style.minWidth = '0';
    this.element.style.minHeight = '0';
    this.element.style.width = '100%';
    this.element.style.height = '100%';
    this.element.style.background = 'var(--bg-cell)';
    this.element.style.border = '1px solid var(--border)';

    this.canvas = document.createElement('canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.inset = '0';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.element.appendChild(this.canvas);

    this.canvas.addEventListener('click', (e) => this.onClick(e));

    // Re-render on resize. ResizeObserver fires whenever the layout gives the
    // canvas a different CSS size — much cleaner than polling.
    const ro = new ResizeObserver(() => this.draw());
    ro.observe(this.element);
  }

  setState(state: DiagramState): void {
    this.state = state;
    this.draw();
  }

  clear(): void {
    this.state = null;
    this.cachedPositions = null;
    this.cachedX = null;
    this.draw();
  }

  private draw(): void {
    const w = this.element.clientWidth;
    const h = this.element.clientHeight;
    if (w <= 0 || h <= 0) return;
    // Re-bind backing store size to logical size × DPR so lines stay crisp on hi-DPI.
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    const ctx = this.canvas.getContext('2d')!;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    ctx.fillStyle = '#0a1825';
    ctx.fillRect(0, 0, w, h);

    const s = this.state;
    if (!s || !s.sizes || s.sizes.length < 2) {
      ctx.fillStyle = 'rgba(224,232,240,0.6)';
      ctx.font = '14px "Segoe UI"';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('(no network loaded — press Start)', w / 2, h / 2);
      return;
    }

    const numLayers = s.sizes.length;
    const positions: Position[][] = new Array(numLayers);
    const layerX: number[] = new Array(numLayers);
    for (let i = 0; i < numLayers; i++) {
      layerX[i] = COLUMN_MARGIN + i * (w - 2 * COLUMN_MARGIN) / (numLayers - 1);
      positions[i] = buildLayerPositions(s.sizes[i], h);
    }
    this.cachedPositions = positions;
    this.cachedX = layerX;

    // Connections.
    for (let i = 0; i < numLayers - 1; i++) {
      this.drawConnections(ctx, positions[i], positions[i + 1], s.weights[i], s.sizes[i], layerX[i], layerX[i + 1], i);
    }

    // Highlighted weight overlay on top of the connection layer.
    if (s.highlightedWeight) {
      const hw = s.highlightedWeight;
      const from = positions[hw.layerIdx];
      const to = positions[hw.layerIdx + 1];
      const fi = findVisible(from, hw.col);
      const ti = findVisible(to, hw.row);
      if (fi >= 0 && ti >= 0) {
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(layerX[hw.layerIdx], from[fi].y);
        ctx.lineTo(layerX[hw.layerIdx + 1], to[ti].y);
        ctx.stroke();
      }
    }

    // Neurons + per-layer labels.
    for (let i = 0; i < numLayers; i++) {
      this.drawNeurons(ctx, positions[i], layerX[i], s.sizes[i], i, i === numLayers - 1);
    }
    ctx.fillStyle = 'rgba(224,232,240,0.6)';
    ctx.font = '11px "Segoe UI"';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < numLayers; i++) {
      const label = i === 0 ? `input (${s.sizes[i]})`
        : i === numLayers - 1 ? `output (${s.sizes[i]})`
        : `hidden (${s.sizes[i]})`;
      ctx.fillText(label, layerX[i], TOP_BOTTOM_MARGIN / 2 + 8);
    }
  }

  private drawConnections(
    ctx: CanvasRenderingContext2D,
    from: Position[], to: Position[],
    wFlat: Float64Array, fromSize: number,
    fromX: number, toX: number,
    weightLayerIndex: number,
  ): void {
    const s = this.state!;
    const isOutputWeightLayer = weightLayerIndex === s.weights.length - 1;
    const layerAlpha = isOutputWeightLayer ? 1 : 0.14;
    const previousAlpha = ctx.globalAlpha;
    ctx.globalAlpha = layerAlpha;

    // Phase 4 (Gradient): recolour by direction & magnitude of the weight update.
    // Phase 5 (HiddenDelta) with focused hidden neuron: spotlight 10 fan-out lines.
    const gradientOverlay = isOutputWeightLayer && s.phase >= Phase.Gradient && s.outputDelta && s.hiddenActivation;
    const hiddenFocusOverlay = isOutputWeightLayer && s.phase >= Phase.HiddenDelta && s.focusedHiddenNeuron != null && s.outputDelta;

    const cols = fromSize;  // wFlat is shape (to.actualSize × from.actualSize); cols = fromSize
    for (let j = 0; j < to.length; j++) {
      const actualJ = to[j].actualIndex;
      for (let i = 0; i < from.length; i++) {
        const actualI = from[i].actualIndex;
        const weight = wFlat[actualJ * cols + actualI];

        let width = 1;
        let color: string;

        if (hiddenFocusOverlay && actualI === s.focusedHiddenNeuron!) {
          // Lines from focused hidden neuron: width ∝ |w·δ|, colour by sign of contribution.
          const contribution = weight * s.outputDelta![actualJ];
          const mag = Math.abs(contribution);
          const alpha = clamp(mag * 3, 0.25, 1);
          width = clamp(1 + mag * 25, 1, 6);
          color = contribution >= 0
            ? `rgba(255,107,107,${alpha.toFixed(3)})`
            : `rgba(107,181,255,${alpha.toFixed(3)})`;
        } else if (hiddenFocusOverlay) {
          if (Math.abs(weight) < 0.02) continue;
          const alpha = clamp(Math.abs(weight) / 6, 0.03, 0.25);
          color = weight >= 0
            ? `rgba(91,186,111,${alpha.toFixed(3)})`
            : `rgba(255,107,107,${alpha.toFixed(3)})`;
        } else if (gradientOverlay) {
          const grad = s.outputDelta![actualJ] * s.hiddenActivation![actualI];
          const mag = Math.abs(grad);
          const alpha = clamp(mag * 4, 0.05, 0.95);
          width = clamp(1 + mag * 20, 1, 5);
          color = grad <= 0
            ? `rgba(91,186,111,${alpha.toFixed(3)})`
            : `rgba(255,107,107,${alpha.toFixed(3)})`;
        } else {
          if (Math.abs(weight) < 0.02) continue;
          const alpha = clamp(Math.abs(weight) / 2, 0.05, 0.85);
          color = weight >= 0
            ? `rgba(91,186,111,${alpha.toFixed(3)})`
            : `rgba(255,107,107,${alpha.toFixed(3)})`;
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(fromX, from[i].y);
        ctx.lineTo(toX, to[j].y);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = previousAlpha;
  }

  private drawNeurons(
    ctx: CanvasRenderingContext2D,
    positions: Position[], x: number, layerSize: number, layerIndex: number, isOutput: boolean,
  ): void {
    const s = this.state!;
    const r = computeNeuronRadius(positions);
    const isInput = layerIndex === 0;
    const isHidden = !isOutput && layerIndex > 0;
    const previousAlpha = ctx.globalAlpha;
    if (isInput) ctx.globalAlpha = 0.22;

    // Decide colouring source.
    let colourByActivation: Float64Array | null = null;
    let colourByDelta: Float64Array | null = null;
    if (isHidden) {
      if (s.phase >= Phase.HiddenDelta && s.hiddenDelta) colourByDelta = s.hiddenDelta;
      else colourByActivation = s.hiddenActivation;
    } else if (isOutput) {
      if (s.phase >= Phase.OutputDelta && s.outputDelta) colourByDelta = s.outputDelta;
      else colourByActivation = s.outputActivation;
    }

    // Ellipsis if column was truncated.
    if (positions.length < layerSize) {
      const topEnd = TRUNCATED_VISIBLE_HALF - 1;
      const dotsTop = positions[topEnd].y + r + 6;
      const dotsBottom = positions[topEnd + 1].y - r - 6;
      const midY = (dotsTop + dotsBottom) / 2;
      ctx.fillStyle = 'rgba(224,232,240,0.6)';
      for (let k = -1; k <= 1; k++) {
        ctx.beginPath();
        ctx.arc(x, midY + k * 6, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (let i = 0; i < positions.length; i++) {
      const actual = positions[i].actualIndex;
      let strokeColor = 'rgba(224,232,240,0.8)';
      let strokeWidth = 1.5;
      if (isOutput && actual === s.trueLabel) { strokeColor = '#ffd700'; strokeWidth = 2.5; }
      else if (isHidden && s.focusedHiddenNeuron === actual) { strokeColor = '#ffd700'; strokeWidth = 2.5; }

      // Dark base so transparent fills don't bleed the canvas behind through.
      ctx.fillStyle = '#102030';
      ctx.beginPath();
      ctx.arc(x, positions[i].y, r, 0, Math.PI * 2);
      ctx.fill();

      let tint: string | null = null;
      if (colourByDelta) {
        const d = colourByDelta[actual];
        const alpha = clamp(Math.abs(d) * 5, 0.05, 0.95);
        tint = d >= 0
          ? `rgba(255,107,107,${alpha.toFixed(3)})`
          : `rgba(107,181,255,${alpha.toFixed(3)})`;
      } else if (colourByActivation) {
        const v = clamp(colourByActivation[actual], 0, 1);
        tint = `rgba(91,186,111,${Math.max(0.05, Math.min(0.95, v)).toFixed(3)})`;
      }
      if (tint) {
        ctx.fillStyle = tint;
        ctx.beginPath();
        ctx.arc(x, positions[i].y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = strokeWidth;
      ctx.beginPath();
      ctx.arc(x, positions[i].y, r, 0, Math.PI * 2);
      ctx.stroke();

      if (isOutput) {
        const isPred = actual === s.predictedLabel;
        const isTrue = actual === s.trueLabel;
        ctx.fillStyle = isPred ? (isTrue ? '#5bba6f' : '#ff6b6b') : 'rgba(224,232,240,0.6)';
        ctx.font = '11px "Segoe UI"';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${actual}`, x + r + 14, positions[i].y);
      }
    }
    ctx.globalAlpha = previousAlpha;
  }

  private onClick(e: MouseEvent): void {
    if (!this.cachedPositions || !this.cachedX || !this.state) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // 1. Look for a hidden-neuron hit (output is also clickable but we only
    //    use hidden clicks for now; the diagram-line hit-test catches W₂ clicks).
    const numLayers = this.state.sizes.length;
    for (let li = 0; li < numLayers; li++) {
      const positions = this.cachedPositions[li];
      const x = this.cachedX[li];
      const r = computeNeuronRadius(positions);
      for (let i = 0; i < positions.length; i++) {
        const dy = py - positions[i].y;
        const dx = px - x;
        if (dx * dx + dy * dy <= (r + 3) * (r + 3)) {
          if (li === 1) {
            this.element.dispatchEvent(new CustomEvent('hidden-click', {
              detail: { index: positions[i].actualIndex }, bubbles: true,
            }));
            return;
          }
        }
      }
    }

    // 2. Closest connection-line hit, within 5 px.
    const threshold = 5;
    let bestDist = threshold;
    let best: { layerIdx: number; row: number; col: number } | null = null;
    for (let li = 1; li < this.cachedPositions.length - 1; li++) {
      const from = this.cachedPositions[li];
      const to = this.cachedPositions[li + 1];
      const fromX = this.cachedX[li];
      const toX = this.cachedX[li + 1];
      for (let j = 0; j < to.length; j++) {
        for (let i = 0; i < from.length; i++) {
          const d = distToSegment(px, py, fromX, from[i].y, toX, to[j].y);
          if (d < bestDist) {
            bestDist = d;
            best = { layerIdx: li, row: to[j].actualIndex, col: from[i].actualIndex };
          }
        }
      }
    }
    if (best) {
      this.element.dispatchEvent(new CustomEvent('weight-click', { detail: best, bubbles: true }));
    }
  }
}

// ── helpers ────────────────────────────────────────────────────────────────

function buildLayerPositions(total: number, canvasH: number): Position[] {
  const topY = TOP_BOTTOM_MARGIN + 18;
  const bottomY = canvasH - TOP_BOTTOM_MARGIN;
  const span = bottomY - topY;
  if (total <= FULL_DRAW_MAX_COUNT) {
    const out: Position[] = new Array(total);
    const step = total === 1 ? 0 : span / (total - 1);
    for (let i = 0; i < total; i++) out[i] = { y: topY + i * step, actualIndex: i };
    return out;
  }
  const out: Position[] = new Array(TRUNCATED_VISIBLE_COUNT);
  const gapFraction = 0.18;
  const halfSpan = span * (1 - gapFraction) / 2;
  const stepHalf = halfSpan / (TRUNCATED_VISIBLE_HALF - 1);
  for (let i = 0; i < TRUNCATED_VISIBLE_HALF; i++) out[i] = { y: topY + i * stepHalf, actualIndex: i };
  const bottomStart = bottomY - halfSpan;
  for (let i = 0; i < TRUNCATED_VISIBLE_HALF; i++) {
    out[TRUNCATED_VISIBLE_HALF + i] = { y: bottomStart + i * stepHalf, actualIndex: total - TRUNCATED_VISIBLE_HALF + i };
  }
  return out;
}

function computeNeuronRadius(positions: Position[]): number {
  if (positions.length <= 1) return MAX_NEURON_RADIUS;
  const spacing = Math.abs(positions[1].y - positions[0].y);
  return clamp(spacing * 0.4, MIN_NEURON_RADIUS, MAX_NEURON_RADIUS);
}

function findVisible(positions: Position[], actualIndex: number): number {
  for (let i = 0; i < positions.length; i++) if (positions[i].actualIndex === actualIndex) return i;
  return -1;
}

function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-9) {
    const ex = px - ax, ey = py - ay;
    return Math.sqrt(ex * ex + ey * ey);
  }
  const t = clamp(((px - ax) * dx + (py - ay) * dy) / lenSq, 0, 1);
  const qx = px - (ax + t * dx);
  const qy = py - (ay + t * dy);
  return Math.sqrt(qx * qx + qy * qy);
}

function clamp(v: number, lo: number, hi: number): number { return v < lo ? lo : v > hi ? hi : v; }
