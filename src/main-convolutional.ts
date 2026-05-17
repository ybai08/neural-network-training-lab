// Convolutional teaching page. Spawns convnet.worker.ts (a separate CNN-specific
// worker) and renders the network as a horizontal pipeline of "layer blocks",
// each showing the layer's role + output shape + visualisation of its current
// activations or filter weights.
//
// Architectural narrative (top → bottom of the page):
//   1. Arch banner — text summary of the whole pipeline + parameter counts.
//   2. Sample image + final prediction sit on the left, training charts on the right.
//   3. Layer pipeline (scrolls horizontally if needed):
//      Input → Conv1 filters + outputs → Pool1 outputs → Conv2 filters + outputs → FC weights + output.
//   4. Controls always docked at the bottom.

import './style.css';
import type { MainToWorker, ConvWorkerToMain, ConvNetSnapshot } from './types';
import { LineChart } from './ui/charts';
import { FeatureMapsView } from './ui/feature-maps-view';
import { FilterBankView } from './ui/filter-bank-view';
import { MatrixView } from './ui/matrix-view';
import { VectorView } from './ui/vector-view';

// Architecture constants — must match convnet.ts.
const INPUT_SIDE = 14;
const SAMPLE_DISPLAY = 224;
const CONV1_FILTERS = 4;
const CONV1_OUT = 12;
const POOL1_OUT = 6;
const CONV2_FILTERS = 8;
const CONV2_OUT = 4;
const FC_INPUT = CONV2_OUT * CONV2_OUT * CONV2_FILTERS; // 128
const OUTPUTS = 10;

// Parameter counts per layer (so we can show them in the arch banner).
const PARAMS_CONV1 = CONV1_FILTERS * (1 * 3 * 3 + 1);                       // 40
const PARAMS_CONV2 = CONV2_FILTERS * (CONV1_FILTERS * 3 * 3 + 1);          // 296
const PARAMS_FC = OUTPUTS * FC_INPUT + OUTPUTS;                             // 1290
const PARAMS_TOTAL = PARAMS_CONV1 + PARAMS_CONV2 + PARAMS_FC;               // 1626

// ─── State ────────────────────────────────────────────────────────────────

let stepMode = true;
let latestSnapshot: ConvNetSnapshot | null = null;

// ─── Worker ───────────────────────────────────────────────────────────────

const worker = new Worker(new URL('./convnet.worker.ts', import.meta.url), { type: 'module' });
function post(msg: MainToWorker): void { worker.postMessage(msg); }

// ─── UI components ────────────────────────────────────────────────────────

const conv1Filters = new FilterBankView({ cellSize: 18 });
const conv1Pre = new FeatureMapsView({ cellSize: 12 });
const conv1Relu = new FeatureMapsView({ cellSize: 12 });
const pool1View = new FeatureMapsView({ cellSize: 18 });
const conv2Filters = new FilterBankView({ cellSize: 14 });
const conv2Pre = new FeatureMapsView({ cellSize: 18 });
const conv2Relu = new FeatureMapsView({ cellSize: 18 });
const fcWeights = new MatrixView({ cellWidth: 8, cellHeight: 18 });
const outputVec = new VectorView({ cellWidth: 52, fixedScale: 1.0 });
const targetVec = new VectorView({ cellWidth: 52, fixedScale: 1.0 });

const lossChart = new LineChart({ title: 'Test-set loss (½‖a − y‖²)', color: '#5BBA6F' });
const accuracyChart = new LineChart({ title: 'Test-set accuracy', color: '#FFB347', fixedRange: { min: 0, max: 1 } });

// Sample image canvas (14×14 → 224 px square).
const sampleCanvas = document.createElement('canvas');
sampleCanvas.width = INPUT_SIDE;
sampleCanvas.height = INPUT_SIDE;
sampleCanvas.style.width = `${SAMPLE_DISPLAY}px`;
sampleCanvas.style.height = `${SAMPLE_DISPLAY}px`;
sampleCanvas.style.imageRendering = 'pixelated';
sampleCanvas.style.background = '#000';
sampleCanvas.style.border = '1px solid var(--border)';

// ─── Layout ───────────────────────────────────────────────────────────────

buildLayout();

function buildLayout(): void {
  const app = document.getElementById('app')!;
  app.style.gridTemplateRows = 'auto 1fr auto';

  // ── HEADER: status + nav + architecture banner ───────────────────────────
  const header = document.createElement('div');
  header.style.display = 'flex';
  header.style.flexDirection = 'column';
  header.style.gap = '8px';

  const statusRow = document.createElement('div');
  statusRow.style.display = 'flex';
  statusRow.style.justifyContent = 'space-between';
  statusRow.style.alignItems = 'center';
  statusRow.style.gap = '12px';
  const statusText = document.createElement('span');
  statusText.id = 'status-text';
  statusText.style.fontSize = '14px';
  statusText.textContent = 'Ready. Press Start to begin training.';
  statusText.style.flex = '1';
  const nav = document.createElement('nav');
  nav.className = 'page-nav';
  nav.innerHTML = '<a href="/">Regular (fully connected)</a><a href="/convolutional/" class="current">Convolutional</a>';
  const miniBatchText = document.createElement('span');
  miniBatchText.id = 'minibatch-text';
  miniBatchText.style.fontSize = '14px';
  miniBatchText.style.fontWeight = '700';
  statusRow.appendChild(statusText);
  statusRow.appendChild(nav);
  statusRow.appendChild(miniBatchText);
  header.appendChild(statusRow);

  // Architecture summary — visually shows the pipeline at a glance.
  header.appendChild(buildArchBanner());
  app.appendChild(header);

  // ── MAIN: top row (sample + charts), bottom row (layer pipeline) ────────
  const main = document.createElement('div');
  main.id = 'main';

  // Top row: sample image on the left, final prediction in the middle, charts on the right.
  const topRow = document.createElement('div');
  topRow.className = 'row';
  topRow.style.display = 'grid';
  topRow.style.gridTemplateColumns = `${SAMPLE_DISPLAY + 20}px auto 1fr`;
  topRow.style.gap = '14px';
  topRow.style.flex = '0 0 auto';

  // Sample column.
  const sampleCol = document.createElement('div');
  sampleCol.style.display = 'flex';
  sampleCol.style.flexDirection = 'column';
  sampleCol.style.gap = '6px';
  const sampleHdr = document.createElement('div');
  sampleHdr.textContent = 'Input — 14×14 grayscale';
  sampleHdr.style.fontWeight = '700';
  sampleCol.appendChild(sampleHdr);
  sampleCol.appendChild(sampleCanvas);
  const labelBox = document.createElement('div');
  labelBox.style.fontSize = '13px';
  labelBox.innerHTML = `
    <div>True label: <b id="true-label">—</b></div>
    <div>Network says: <b id="predicted-label">—</b></div>
    <div id="sample-loss" style="color:var(--gold);font-weight:700;margin-top:4px;"></div>`;
  sampleCol.appendChild(labelBox);
  topRow.appendChild(sampleCol);

  // Prediction column — output vector + target vector side-by-side.
  const predCol = document.createElement('div');
  predCol.className = 'layer-block';
  predCol.style.height = `${SAMPLE_DISPLAY + 50}px`;
  predCol.innerHTML = '<div class="layer-title">Output a (10)  vs  target y</div><div class="layer-dim">σ(z), one per digit class</div>';
  const predRow = document.createElement('div');
  predRow.style.display = 'flex';
  predRow.style.gap = '12px';
  const aWrap = document.createElement('div');
  aWrap.innerHTML = '<div class="layer-section-label">a</div>';
  aWrap.appendChild(outputVec.element);
  const yWrap = document.createElement('div');
  yWrap.innerHTML = '<div class="layer-section-label">y</div>';
  yWrap.appendChild(targetVec.element);
  predRow.appendChild(aWrap);
  predRow.appendChild(yWrap);
  predCol.appendChild(predRow);
  topRow.appendChild(predCol);

  // Charts column.
  const chartsCol = document.createElement('div');
  chartsCol.style.display = 'grid';
  chartsCol.style.gridTemplateRows = '1fr 1fr';
  chartsCol.style.gap = '8px';
  chartsCol.appendChild(lossChart.element);
  chartsCol.appendChild(accuracyChart.element);
  topRow.appendChild(chartsCol);

  main.appendChild(topRow);

  // ── Layer pipeline (horizontally scrollable if needed) ──────────────────
  const pipeline = document.createElement('div');
  pipeline.className = 'row';
  pipeline.style.display = 'flex';
  pipeline.style.gap = '8px';
  pipeline.style.alignItems = 'stretch';
  pipeline.style.overflowX = 'auto';
  pipeline.style.flex = '0 0 auto';
  pipeline.style.padding = '4px 0';

  pipeline.appendChild(buildBlock({
    title: 'Conv1',
    dim: '3×3 filters · 4 outputs · valid · stride 1',
    sections: [
      { label: `Filters (${CONV1_FILTERS} × 1 × 3 × 3 + ${CONV1_FILTERS} biases = ${PARAMS_CONV1} params)`, view: conv1Filters.element },
      { label: `Pre-ReLU output  →  12×12×${CONV1_FILTERS}`, view: conv1Pre.element },
      { label: `After ReLU       →  12×12×${CONV1_FILTERS}`, view: conv1Relu.element },
    ],
  }));
  pipeline.appendChild(arrow());

  pipeline.appendChild(buildBlock({
    title: 'MaxPool1',
    dim: '2×2 max · stride 2 · 0 params',
    sections: [
      { label: `Output  →  6×6×${CONV1_FILTERS}`, view: pool1View.element },
    ],
  }));
  pipeline.appendChild(arrow());

  pipeline.appendChild(buildBlock({
    title: 'Conv2',
    dim: '3×3 filters · 8 outputs · valid · stride 1',
    sections: [
      { label: `Filters (${CONV2_FILTERS} × ${CONV1_FILTERS} × 3 × 3 + ${CONV2_FILTERS} biases = ${PARAMS_CONV2} params)`, view: conv2Filters.element },
      { label: `Pre-ReLU output  →  4×4×${CONV2_FILTERS}`, view: conv2Pre.element },
      { label: `After ReLU       →  4×4×${CONV2_FILTERS}`, view: conv2Relu.element },
    ],
  }));
  pipeline.appendChild(arrow());

  pipeline.appendChild(buildBlock({
    title: 'Flatten + FC',
    dim: `128 → 10 · ${PARAMS_FC} params`,
    sections: [
      { label: `Weights W  (10 × 128)`, view: fcWeights.element },
    ],
  }));

  main.appendChild(pipeline);
  app.appendChild(main);

  // ── FOOTER: controls ────────────────────────────────────────────────────
  const controls = document.createElement('div');
  controls.className = 'controls';
  const startBtn = button('Start', () => onStart());
  const stopBtn = button('Stop', () => post({ type: 'stop' }));
  stopBtn.disabled = true;

  const stepLabel = document.createElement('label');
  stepLabel.style.display = 'flex';
  stepLabel.style.alignItems = 'center';
  stepLabel.style.gap = '6px';
  const stepCb = document.createElement('input');
  stepCb.type = 'checkbox';
  stepCb.checked = true;
  stepCb.addEventListener('change', () => {
    stepMode = stepCb.checked;
    post({ type: 'setStepMode', on: stepMode });
    nextBtn.disabled = !(stepMode && !stopBtn.disabled);
  });
  const stepText = document.createElement('span');
  stepText.textContent = 'Step mode';
  stepLabel.appendChild(stepCb);
  stepLabel.appendChild(stepText);

  const nextBtn = button('Next sample', () => post({ type: 'stepNext' }));
  nextBtn.disabled = true;

  controls.appendChild(startBtn);
  controls.appendChild(stopBtn);
  controls.appendChild(stepLabel);
  controls.appendChild(nextBtn);

  const spacer = document.createElement('div');
  spacer.className = 'spacer';
  controls.appendChild(spacer);

  const throttleLabel = document.createElement('span');
  throttleLabel.textContent = 'Throttle (auto mode ms):';
  controls.appendChild(throttleLabel);
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '2000';
  slider.value = '200';
  slider.style.width = '200px';
  const throttleVal = document.createElement('span');
  throttleVal.textContent = '200 ms';
  throttleVal.style.minWidth = '56px';
  slider.addEventListener('input', () => {
    throttleVal.textContent = `${slider.value} ms`;
    post({ type: 'setThrottleMs', ms: +slider.value });
  });
  controls.appendChild(slider);
  controls.appendChild(throttleVal);

  app.appendChild(controls);

  (window as any).__nodes = {
    statusText, miniBatchText, startBtn, stopBtn, nextBtn,
    sampleLossEl: document.getElementById('sample-loss')!,
    trueLabelEl: document.getElementById('true-label')!,
    predictedLabelEl: document.getElementById('predicted-label')!,
  };

  function onStart(): void {
    lossChart.clear();
    accuracyChart.clear();
    post({ type: 'init', learningRate: 0.1, miniBatchSize: 10, epochs: 30, seed: 42, stepMode });
    post({ type: 'setThrottleMs', ms: +slider.value });
    post({ type: 'start' });
    startBtn.disabled = true;
    stopBtn.disabled = false;
    nextBtn.disabled = !stepMode;
  }
}

// ─── Architecture summary banner ──────────────────────────────────────────

function buildArchBanner(): HTMLElement {
  const banner = document.createElement('div');
  banner.className = 'arch-banner';
  // The pipeline is a sequence of (step, shape) pairs joined by → arrows.
  const steps: { step: string; shape: string }[] = [
    { step: 'Input',          shape: '14×14×1' },
    { step: 'Conv 3×3, 4',    shape: '12×12×4' },
    { step: 'ReLU',           shape: '12×12×4' },
    { step: 'MaxPool 2×2',    shape: '6×6×4' },
    { step: 'Conv 3×3, 8',    shape: '4×4×8' },
    { step: 'ReLU',           shape: '4×4×8' },
    { step: 'Flatten',        shape: '128' },
    { step: 'FC',             shape: '10' },
    { step: 'Sigmoid',        shape: '10' },
  ];
  const cells: string[] = [];
  steps.forEach((s, i) => {
    cells.push(`<span class="arch-step">${s.step}</span>`);
    cells.push(`<span class="arch-shape">[${s.shape}]</span>`);
    if (i < steps.length - 1) cells.push(`<span class="arch-arrow">→</span>`);
  });
  cells.push(`<span class="arch-arrow">·</span>`);
  cells.push(`<span class="arch-params">total params ≈ ${PARAMS_TOTAL}</span>`);
  banner.innerHTML = cells.join('');
  return banner;
}

// ─── Layer-block builder ──────────────────────────────────────────────────

function buildBlock(opts: {
  title: string;
  dim: string;
  sections: { label: string; view: HTMLElement }[];
}): HTMLElement {
  const block = document.createElement('div');
  block.className = 'layer-block';
  const t = document.createElement('div');
  t.className = 'layer-title';
  t.textContent = opts.title;
  block.appendChild(t);
  const d = document.createElement('div');
  d.className = 'layer-dim';
  d.textContent = opts.dim;
  block.appendChild(d);
  for (const sec of opts.sections) {
    const lbl = document.createElement('div');
    lbl.className = 'layer-section-label';
    lbl.textContent = sec.label;
    block.appendChild(lbl);
    block.appendChild(sec.view);
  }
  return block;
}

function arrow(): HTMLElement {
  const a = document.createElement('div');
  a.className = 'layer-arrow';
  a.textContent = '→';
  return a;
}

// ─── Worker message handler ───────────────────────────────────────────────

worker.onmessage = (ev: MessageEvent<ConvWorkerToMain>) => {
  const msg = ev.data;
  const nodes = (window as any).__nodes;
  switch (msg.type) {
    case 'status':
      nodes.statusText.textContent = msg.text;
      break;
    case 'sample': {
      latestSnapshot = msg.snapshot;
      const oneBased = msg.snapshot.miniBatchSampleIndex + 1;
      const isLast = oneBased === msg.snapshot.miniBatchSize;
      nodes.miniBatchText.textContent = isLast
        ? `Sample ${oneBased} of ${msg.snapshot.miniBatchSize}  →  weights about to be APPLIED`
        : `Sample ${oneBased} of ${msg.snapshot.miniBatchSize}  →  accumulating gradients (weights unchanged)`;
      nodes.miniBatchText.style.color = isLast ? 'var(--gold)' : 'var(--text-dim)';
      nodes.trueLabelEl.textContent = `${msg.snapshot.trueLabel}`;
      nodes.predictedLabelEl.textContent = `${msg.snapshot.predictedLabel}`;
      nodes.predictedLabelEl.style.color =
        msg.snapshot.predictedLabel === msg.snapshot.trueLabel ? 'var(--green)' : 'var(--red)';
      drawSampleImage(msg.snapshot.pixels, msg.snapshot.inputSide);

      // Sample-level loss for the running scalar readout.
      let c = 0;
      for (let k = 0; k < OUTPUTS; k++) {
        const d = msg.snapshot.output[k] - msg.snapshot.target[k];
        c += d * d;
      }
      c *= 0.5;
      nodes.sampleLossEl.textContent = `C (this sample) = (1/2) · Σ (a[k] − y[k])²  =  ${c.toFixed(4)}`;

      renderSnapshot(msg.snapshot);
      break;
    }
    case 'miniBatchApplied':
      // After weights update, refresh filter visualisations to show the new state.
      if (latestSnapshot) renderSnapshot(latestSnapshot);
      break;
    case 'epoch': {
      const label = msg.snapshot.label ?? `epoch ${msg.snapshot.epoch + 1}`;
      lossChart.push(label, msg.snapshot.meanTestLoss, msg.snapshot.meanTestLoss.toFixed(4));
      const acc = msg.snapshot.testTotal > 0 ? msg.snapshot.testCorrect / msg.snapshot.testTotal : 0;
      accuracyChart.push(label, acc, `${msg.snapshot.testCorrect}/${msg.snapshot.testTotal} (${(acc * 100).toFixed(2)}%)`);
      break;
    }
    case 'done':
      nodes.startBtn.disabled = false;
      nodes.stopBtn.disabled = true;
      nodes.nextBtn.disabled = true;
      nodes.statusText.textContent = 'Training complete.';
      break;
    case 'error':
      nodes.statusText.textContent = `Error: ${msg.message}`;
      nodes.statusText.style.color = 'var(--red)';
      break;
  }
};

// ─── Rendering ────────────────────────────────────────────────────────────

function renderSnapshot(snap: ConvNetSnapshot): void {
  conv1Filters.update(snap.conv1Kernels, CONV1_FILTERS, 1, 3, 3);
  conv1Pre.update(snap.conv1Pre, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  conv1Relu.update(snap.conv1Relu, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  pool1View.update(snap.pool1Out, CONV1_FILTERS, POOL1_OUT, POOL1_OUT);
  conv2Filters.update(snap.conv2Kernels, CONV2_FILTERS, CONV1_FILTERS, 3, 3);
  conv2Pre.update(snap.conv2Pre, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  conv2Relu.update(snap.conv2Relu, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  fcWeights.update(snap.fcWeights, OUTPUTS, FC_INPUT);
  outputVec.update(snap.output, OUTPUTS);
  targetVec.update(snap.target, OUTPUTS);
}

function drawSampleImage(pixels: Float64Array, side: number): void {
  const ctx = sampleCanvas.getContext('2d')!;
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
    img.data[i * 4 + 0] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(img, 0, 0);
}

function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}
