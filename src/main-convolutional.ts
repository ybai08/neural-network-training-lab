// Convolutional teaching page. Spawns convnet.worker.ts (a separate CNN-specific
// worker) and renders the network as a step-by-step lesson surface:
// sample + architecture map up top, live layer details in the learning area,
// output inspection on the right, and charts at the lower-left.

import './style.css';
import type { MainToWorker, ConvWorkerToMain, ConvNetSnapshot } from './types';
import { LineChart } from './ui/charts';
import { FeatureMapsView } from './ui/feature-maps-view';
import { FilterBankView, type FilterBankCell } from './ui/filter-bank-view';
import { MatrixView } from './ui/matrix-view';
import { renderMath } from './ui/katex-util';
import { VectorView } from './ui/vector-view';

// Architecture constants — must match convnet.ts.
const INPUT_SIDE = 14;
const SAMPLE_DISPLAY = 200;
const CONV1_FILTERS = 4;
const CONV1_OUT = 12;
const POOL1_OUT = 6;
const CONV2_FILTERS = 8;
const CONV2_OUT = 4;
const FC_INPUT = CONV2_OUT * CONV2_OUT * CONV2_FILTERS; // 128
const OUTPUTS = 10;

const MAP_FULL_DRAW_MAX_COUNT = 200;
const MAP_MAX_NEURON_RADIUS = 9;
const MAP_MIN_NEURON_RADIUS = 1.5;
const MAP_COLUMN_MARGIN = 60;
const MAP_TOP_BOTTOM_MARGIN = 16;
const REGULAR_NN_LINE_WEIGHT_SCALE = 2;
const CNN_DENSE_LINE_WEIGHT_SCALE = REGULAR_NN_LINE_WEIGHT_SCALE / Math.sqrt(FC_INPUT);

// ─── State ────────────────────────────────────────────────────────────────

let stepMode = true;
let learningRate = 0.1;
let latestSnapshot: ConvNetSnapshot | null = null;
let setActiveCnnStep: ((index: number, scroll?: boolean) => void) | null = null;
let setCnnStepsEnabled: ((on: boolean) => void) | null = null;
let cnnMapElement: HTMLElement | null = null;
let cnnMapCanvas: HTMLCanvasElement | null = null;
let cnnMapSnapshot: ConvNetSnapshot | null = null;
let cnnStepsEnabled = false;

function appHref(route: '' | 'convolutional/'): string {
  const base = import.meta.env.BASE_URL;
  if (!base || base === './') return route ? `/${route}` : '/';
  return `${base}${route}`;
}

type MapPosition = { y: number; actualIndex: number };
type MapCell = { x: number; y: number; actualIndex: number };
type FeatureMapBox = { x: number; y: number; size: number; actualIndex: number; gridSide: number; cells: MapCell[] };
type CnnMapLineRef =
  | { kind: 'conv1'; outChannel: number; ky: number; kx: number; outRow: number; outCol: number }
  | { kind: 'conv2'; outChannel: number; inChannel: number; ky: number; kx: number; outRow: number; outCol: number }
  | { kind: 'fc'; row: number; col: number };
type CnnParamSelection =
  | ({ kind: 'convWeight'; layer: 1 | 2 } & FilterBankCell & { line?: CnnMapLineRef })
  | { kind: 'convBias'; layer: 1 | 2; index: number }
  | { kind: 'denseWeight'; row: number; col: number; line?: CnnMapLineRef }
  | { kind: 'denseBias'; index: number };
let focusedCnnParam: CnnParamSelection | null = null;

const CNN_MAP_COLUMNS = [
  { label: 'input (196)', step: null },
  { label: 'conv1 (4)', step: 1 },
  { label: 'pool (4)', step: 2 },
  { label: 'conv2 (8)', step: 3 },
  { label: 'output (10)', step: 4 },
] as const;

const CNN_STEPS = [
  {
    stepIndex: 1,
    label: '1. Conv1',
    targetId: 'cnn-step-conv1',
    detail:
      'Conv1 finds local patterns in the input image $x$ using $4$ learned $3\\times3$ filters. Each filter scans the digit with shared weights, so it can detect the same pattern in different locations:\n' +
      '$$z_{1,f,i,j}=b_{1,f}+\\sum_{u=0}^{2}\\sum_{v=0}^{2}W_{1,f,u,v}\\,x_{i+u,j+v} \\qquad a_{1,f,i,j}=\\mathrm{ReLU}(z_{1,f,i,j})$$' +
      '$$\\mathrm{ReLU}(z)=\\max(0,z)$$' +
      '$f$ selects a filter, and $(i,j)$ selects a location. $W_1$ and $b_1$ are learned; $z_1$ is the raw filter score. ReLU is an activation function like sigmoid, but it does not bound values to $[0,1]$: it maps negative scores to $0$ and keeps positive scores.',
  },
  {
    stepIndex: 2,
    label: '2. Compress',
    targetId: 'cnn-step-pool',
    detail:
      'Compression shrinks each Conv1 map from $12\\times12$ to $6\\times6$ with no weights or biases. Each $2\\times2$ patch keeps its strongest activation:\n' +
      '$$p_{1,f,i,j}=\\max_{0\\le u,v<2} a_{1,f,2i+u,2j+v}$$' +
      'This keeps the strongest nearby feature evidence and reduces sensitivity to small shifts. During backpropagation, the gradient goes back to the cell that was kept.',
  },
  {
    stepIndex: 3,
    label: '3. Conv2',
    targetId: 'cnn-step-conv2',
    detail:
      'Conv2 scans the compressed feature maps instead of the original pixels. Its filters combine Conv1 patterns into more useful digit parts:\n' +
      '$$z_{2,g,i,j}=b_{2,g}+\\sum_{f=0}^{3}\\sum_{u=0}^{2}\\sum_{v=0}^{2}W_{2,g,f,u,v}\\,p_{1,f,i+u,j+v} \\qquad a_{2,g,i,j}=\\mathrm{ReLU}(z_{2,g,i,j})$$' +
      '$g$ selects one of the $8$ Conv2 filters. $W_2$ and $b_2$ are learned; $p_1$ is the compressed output from Step 2.',
  },
  {
    stepIndex: 4,
    label: '4. Classify',
    targetId: 'cnn-step-classifier',
    detail:
      'Flatten the final feature maps $a_2$ into a $128$-number vector $q$. A dense layer produces the $10$ output scores:\n' +
      '$$z_3=W_3q+b_3 \\qquad a_3=\\sigma(z_3) \\qquad C=\\frac12\\sum_k(a_{3,k}-y_k)^2$$' +
      '$a_3=\\hat{y}$ is the network\'s 10-number prediction; $y$ is the target vector. Backpropagation starts with $\\delta_{3,k}=(a_{3,k}-y_k)\\sigma\'(z_{3,k})$, then sends that error backward through the dense layer, Conv2, compression, and Conv1.',
  },
] as const;

const CNN_LOCKED_EXPLAINER =
  'Press Start to load the first MNIST training sample. Once a sample is available, the CNN steps unlock so you can inspect the input, convolution filters, pooling, and final classifier.';

// ─── Worker ───────────────────────────────────────────────────────────────

let worker = createWorker();
function post(msg: MainToWorker): void { worker.postMessage(msg); }
function createWorker(): Worker {
  const nextWorker = new Worker(new URL('./convnet.worker.ts', import.meta.url), { type: 'module' });
  nextWorker.onmessage = handleWorkerMessage;
  return nextWorker;
}

// ─── UI components ────────────────────────────────────────────────────────

const conv1Filters = new FilterBankView({ cellSize: 20, gap: 4 });
const conv1BiasVec = new VectorView({ cellWidth: 42, cellHeight: 18, title: 'b' });
const conv1Pre = new FeatureMapsView({ cellSize: 6, gap: 5 });
const conv1Relu = new FeatureMapsView({ cellSize: 6, gap: 5 });
const pool1View = new FeatureMapsView({ cellSize: 13, gap: 6 });
const conv2Filters = new FilterBankView({ cellSize: 18, gap: 4, inputLabelPrefix: 'pool' });
const conv2BiasVec = new VectorView({ cellWidth: 42, cellHeight: 18, title: 'b' });
const conv2Pre = new FeatureMapsView({ cellSize: 7, gap: 5 });
const conv2Relu = new FeatureMapsView({ cellSize: 7, gap: 5 });
const fcWeights = new MatrixView({ cellWidth: 4, cellHeight: 9, colLabelEvery: 10 });
const fcBiasVec = new VectorView({ cellWidth: 42, cellHeight: 14, title: 'b' });
const targetVec = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 1.0 });
const fcOutVec = new VectorView({ cellWidth: 34, cellHeight: 18 });
const outputVec = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 1.0 });
const fcSigmoidPrimeVec = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 0.25 });
const fcErrorVec = new VectorView({ cellWidth: 34, cellHeight: 18 });
const fcDeltaVec = new VectorView({ cellWidth: 34, cellHeight: 18 });

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
  document.body.classList.add('cnn-page', 'cnn-steps-locked');
  app.style.gridTemplateRows = 'auto 1fr auto';

  // ── HEADER: lesson title + nav + status + architecture banner ────────────
  const header = document.createElement('div');
  header.className = 'lesson-header';

  const titleRow = document.createElement('div');
  titleRow.className = 'lesson-title-row';
  const titleBlock = document.createElement('div');
  titleBlock.innerHTML = `
    <h1>Neural Network Training Lab</h1>
    <div class="lesson-source">
      Based on videos 1-4 of
      <a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" target="_blank" rel="noopener noreferrer">
        3Blue1Brown's neural networks series
      </a>
      <span class="source-separator">Data:</span>
      <a href="https://github.com/mkolod/MNIST" target="_blank" rel="noopener noreferrer">
        MNIST dataset GitHub mirror
      </a>
    </div>`;
  titleRow.appendChild(titleBlock);

  const headerActions = document.createElement('div');
  headerActions.className = 'lesson-actions';
  const nav = document.createElement('nav');
  nav.className = 'page-nav';
  nav.innerHTML = `<a href="${appHref('')}">Regular NN</a><a href="${appHref('convolutional/')}" class="current">CNN</a>`;
  headerActions.appendChild(nav);
  const trainingInfo = document.createElement('span');
  trainingInfo.className = 'info-tip compact-info site-info';
  trainingInfo.tabIndex = 0;
  trainingInfo.setAttribute('aria-label', 'About this CNN training demo');
  trainingInfo.innerHTML = `?
    <span class="info-tip-content">
      This CNN demo trains on MNIST digits: $60{,}000$ training samples over $30$ epochs.<br>
      Mini-batch size is fixed at $10$ here, so each update averages gradients from $10$ examples.<br>
      The learning rate $\\eta$ below controls how far the CNN weights move after each update.
    </span>`;
  headerActions.appendChild(trainingInfo);
  renderMath(trainingInfo);
  titleRow.appendChild(headerActions);
  header.appendChild(titleRow);

  const statusText = document.createElement('span');
  statusText.id = 'status-text';
  statusText.textContent = 'Ready. Press Start to begin training.';
  const miniBatchText = document.createElement('span');
  miniBatchText.id = 'minibatch-text';
  miniBatchText.className = 'status-pill';
  miniBatchText.textContent = 'Update not started';

  const stepGuide = buildCnnStepGuide();
  header.appendChild(stepGuide.strip);
  header.appendChild(stepGuide.explainer);
  app.appendChild(header);

  // ── MAIN: lesson-style CNN workspace ────────────────────────────────────
  const main = document.createElement('div');
  main.id = 'main';

  // Row 1: sample + compact architecture map, matching the Regular NN stage.
  const primaryStage = document.createElement('div');
  primaryStage.className = 'row primary-stage cnn-primary-stage';
  primaryStage.style.display = 'grid';
  primaryStage.style.gridTemplateColumns = `${SAMPLE_DISPLAY + 28}px 1fr`;
  primaryStage.style.gap = '12px';
  primaryStage.style.flex = '0 0 290px';

  const sampleCol = document.createElement('div');
  sampleCol.id = 'cnn-step-input';
  sampleCol.className = 'sample-card';
  const sampleLabel = document.createElement('div');
  sampleLabel.className = 'sample-title';
  sampleLabel.innerHTML = '<span>Training sample</span><small>Resolution: 14×14 (196 inputs)</small>';
  sampleCol.appendChild(sampleLabel);
  sampleCol.appendChild(sampleCanvas);
  const labelBox = document.createElement('div');
  labelBox.id = 'label-box';
  labelBox.className = 'prediction-readout';
  labelBox.innerHTML = `
    <div><span>True</span><b id="true-label">—</b></div>
    <div><span>Guess</span><b id="predicted-label">—</b></div>`;
  sampleCol.appendChild(labelBox);
  primaryStage.appendChild(sampleCol);

  const mapPanel = document.createElement('div');
  mapPanel.className = 'diagram-panel cnn-map-panel';
  const mapHeader = document.createElement('div');
  mapHeader.className = 'panel-header diagram-header';
  mapHeader.innerHTML = '<span>Network map</span>';
  mapPanel.appendChild(mapHeader);
  mapPanel.appendChild(buildCnnNetworkMap());
  primaryStage.appendChild(mapPanel);
  main.appendChild(primaryStage);

  const detailsBody = document.createElement('div');
  detailsBody.className = 'details-body';
  const cnnWorkspace = document.createElement('div');
  cnnWorkspace.className = 'row cnn-workspace';
  const cnnLeft = document.createElement('div');
  cnnLeft.className = 'cnn-left-stack';

  const stepRows = document.createElement('div');
  stepRows.className = 'cnn-step-rows';
  const conv1Params = buildCnnParamsView(conv1Filters.element, conv1BiasVec.element);
  const conv2Params = buildCnnParamsView(conv2Filters.element, conv2BiasVec.element);
  const outputParams = buildCnnParamsView(fcWeights.element, fcBiasVec.element);
  setupConvParamsAutoSize(conv1Params, conv1Filters, conv1BiasVec, CONV1_FILTERS, 3, 16, 31);
  setupConvParamsAutoSize(conv2Params, conv2Filters, conv2BiasVec, CONV2_FILTERS, 3, 10, 18, 31);
  setupDenseParamsAutoSize(outputParams);
  wireCnnParameterInteractions();
  stepRows.appendChild(buildCnnStepRow({
    id: 'cnn-step-conv1',
    stepIndex: 1,
    title: 'Conv1 learning: scan the input with $W_1$, add $b_1$, then apply ReLU',
    sections: [
      { label: '$W_1,b_1$ (Conv1 parameters)', view: conv1Params, className: 'cnn-params-section' },
      { label: '$z_1 = W_1x + b_1$', view: conv1Pre.element, className: 'cnn-activation-section' },
      { label: '$a_1 = \\mathrm{ReLU}(z_1)$', view: conv1Relu.element, className: 'cnn-activation-section' },
    ],
  }));
  stepRows.appendChild(buildCnnStepRow({
    id: 'cnn-step-conv2',
    stepIndex: 3,
    title: 'Conv2 learning: combine pooled maps with $W_2$, add $b_2$, then apply ReLU',
    sections: [
      { label: '$W_2,b_2$ (Conv2 parameters)', view: conv2Params, className: 'cnn-params-section' },
      { label: '$z_2 = W_2p_1 + b_2$', view: conv2Pre.element, className: 'cnn-activation-section' },
      { label: '$a_2 = \\mathrm{ReLU}(z_2)$', view: conv2Relu.element, className: 'cnn-activation-section' },
    ],
  }));

  const classifierOutput = document.createElement('div');
  classifierOutput.className = 'cnn-output-in-row';
  const sampleLoss = document.createElement('div');
  sampleLoss.id = 'sample-loss';
  sampleLoss.className = 'output-cost-equation cnn-sample-loss';
  sampleLoss.innerHTML = '$C=\\tfrac{1}{2}\\sum_k(a_{3,k}-y_k)^2$';
  renderMath(sampleLoss);
  classifierOutput.appendChild(sampleLoss);
  const outputStateGrid = document.createElement('div');
  outputStateGrid.className = 'output-vector-grid cnn-output-vectors';
  const cnnVectorHeaders = ['$y$', '$z_3$', '$a_3$', "$\\sigma'(z_3)$", '$a_3-y$', '$\\delta_3$'];
  for (const h of cnnVectorHeaders) {
    const hd = document.createElement('div');
    hd.className = 'output-vector-heading';
    hd.innerHTML = h;
    renderMath(hd);
    outputStateGrid.appendChild(hd);
  }
  for (const v of [targetVec, fcOutVec, outputVec, fcSigmoidPrimeVec, fcErrorVec, fcDeltaVec]) {
    outputStateGrid.appendChild(v.element);
  }
  classifierOutput.appendChild(outputStateGrid);

  const classifierRow = buildCnnStepRow({
    id: 'cnn-step-classifier',
    stepIndex: 4,
    title: 'Output layer learning: compare prediction to target, form $\\delta_3$, then compute the gradient for $W_3$',
    sections: [
      { label: '$W_3,b_3$ (dense parameters)', view: outputParams, className: 'cnn-classifier-params-section' },
      { label: '', view: classifierOutput, className: 'cnn-classifier-output-section' },
    ],
  });

  cnnLeft.appendChild(stepRows);
  cnnLeft.appendChild(classifierRow);

  const charts = document.createElement('div');
  charts.className = 'charts-row cnn-charts-row';
  charts.appendChild(lossChart.element);
  charts.appendChild(accuracyChart.element);
  cnnLeft.appendChild(charts);

  cnnWorkspace.appendChild(cnnLeft);
  detailsBody.appendChild(cnnWorkspace);
  main.appendChild(detailsBody);
  app.appendChild(main);
  resetCnnLessonDisplay();

  // ── FOOTER: controls ────────────────────────────────────────────────────
  const controls = document.createElement('div');
  controls.className = 'controls teaching-controls';

  const startBtn = button('Start', () => onStart());
  startBtn.className = 'primary-action';
  const stopBtn = button('Stop', () => post({ type: 'stop' }));
  stopBtn.disabled = true;

  const runGroup = document.createElement('div');
  runGroup.className = 'control-group';
  runGroup.appendChild(startBtn);
  runGroup.appendChild(stopBtn);

  const nextBtn = button('Next sample', () => post({ type: 'stepNext' }));
  nextBtn.className = 'next-action';
  nextBtn.disabled = true;
  runGroup.appendChild(nextBtn);
  controls.appendChild(runGroup);

  const modeGroup = document.createElement('div');
  modeGroup.className = 'control-group';
  const stepLabel = document.createElement('label');
  stepLabel.className = 'toggle-label';
  const stepCb = document.createElement('input');
  stepCb.type = 'checkbox';
  stepCb.checked = true;
  stepCb.addEventListener('change', () => {
    stepMode = stepCb.checked;
    post({ type: 'setStepMode', on: stepMode });
    nextBtn.disabled = !(stepMode && !stopBtn.disabled);
  });
  const stepText = document.createElement('span');
  stepText.textContent = 'Walk through samples';
  stepLabel.appendChild(stepCb);
  stepLabel.appendChild(stepText);
  modeGroup.appendChild(stepLabel);
  controls.appendChild(modeGroup);

  const speedGroup = document.createElement('div');
  speedGroup.className = 'control-group control-range';
  const throttleLabel = document.createElement('span');
  throttleLabel.textContent = 'Auto speed';
  speedGroup.appendChild(throttleLabel);
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
  speedGroup.appendChild(slider);
  speedGroup.appendChild(throttleVal);
  controls.appendChild(speedGroup);

  const settingsGroup = document.createElement('div');
  settingsGroup.className = 'control-group control-settings';
  const lrLabel = document.createElement('label');
  lrLabel.textContent = 'Learning rate (η)';
  const lrInput = document.createElement('input');
  lrInput.type = 'number';
  lrInput.min = '0.1';
  lrInput.max = '10';
  lrInput.step = '0.1';
  lrInput.value = `${learningRate}`;
  lrInput.addEventListener('change', () => {
    const next = Number(lrInput.value);
    if (Number.isFinite(next) && next > 0) learningRate = next;
    lrInput.value = `${learningRate}`;
  });
  lrLabel.appendChild(lrInput);
  settingsGroup.appendChild(lrLabel);
  controls.appendChild(settingsGroup);

  app.appendChild(controls);

  (window as any).__nodes = {
    statusText, miniBatchText, startBtn, stopBtn, nextBtn, lrInput,
    sampleLossEl: document.getElementById('sample-loss')!,
    trueLabelEl: document.getElementById('true-label')!,
    predictedLabelEl: document.getElementById('predicted-label')!,
  };

  function onStart(): void {
    resetCnnLessonDisplay();
    lossChart.clear();
    accuracyChart.clear();
    const nextLearningRate = Number(lrInput.value);
    if (Number.isFinite(nextLearningRate) && nextLearningRate > 0) learningRate = nextLearningRate;
    lrInput.value = `${learningRate}`;
    post({ type: 'init', learningRate, miniBatchSize: 10, epochs: 30, seed: 42, stepMode });
    post({ type: 'setThrottleMs', ms: +slider.value });
    post({ type: 'start' });
    startBtn.disabled = true;
    stopBtn.disabled = false;
    nextBtn.disabled = !stepMode;
    lrInput.disabled = true;
  }
}

// ─── Visual CNN map ───────────────────────────────────────────────────────

function buildCnnNetworkMap(): HTMLElement {
  const map = document.createElement('div');
  map.className = 'cnn-network-map';
  map.setAttribute('role', 'img');
  map.setAttribute('aria-label', 'CNN network map from original input pixels to output digit neurons');
  const canvas = document.createElement('canvas');
  canvas.className = 'cnn-map-canvas';
  map.appendChild(canvas);
  cnnMapElement = map;
  cnnMapCanvas = canvas;
  canvas.addEventListener('click', onCnnMapClick);
  const ro = new ResizeObserver(() => drawCnnNetworkMap());
  ro.observe(map);
  requestAnimationFrame(() => drawCnnNetworkMap());
  return map;
}

function drawCnnNetworkMap(): void {
  const element = cnnMapElement;
  const canvas = cnnMapCanvas;
  if (!element || !canvas) return;
  const w = element.clientWidth;
  const h = element.clientHeight;
  if (w <= 0 || h <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a1825';
  ctx.fillRect(0, 0, w, h);

  if (!cnnMapSnapshot) {
    ctx.fillStyle = 'rgba(224,232,240,0.6)';
    ctx.font = '14px "Segoe UI"';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('(no network loaded - press Start)', w / 2, h / 2);
    return;
  }

  const layerX = cnnMapColumnX(w);
  const inputPositions = buildCnnMapLayerPositions(INPUT_SIDE * INPUT_SIDE, h);
  const outputPositions = buildCnnMapLayerPositions(OUTPUTS, h);
  const conv1Boxes = buildFeatureMapBoxes(layerX[1], CONV1_FILTERS, CONV1_OUT, h);
  const poolBoxes = buildFeatureMapBoxes(layerX[2], CONV1_FILTERS, POOL1_OUT, h);
  const conv2Boxes = buildFeatureMapBoxes(layerX[3], CONV2_FILTERS, CONV2_OUT, h);

  drawFeatureMapBackdrops(ctx, conv1Boxes);
  drawFeatureMapBackdrops(ctx, poolBoxes);
  drawFeatureMapBackdrops(ctx, conv2Boxes);

  drawConv1WeightConnections(ctx, inputPositions, layerX[0], conv1Boxes);
  drawConv2WeightConnections(ctx, poolBoxes, conv2Boxes);
  drawFeatureMapToOutputConnections(ctx, conv2Boxes, outputPositions, layerX[4]);
  drawFocusedCnnMapLine(ctx, layerX, inputPositions, conv1Boxes, poolBoxes, conv2Boxes, outputPositions);

  drawCnnMapNeurons(ctx, inputPositions, layerX[0], true, false);
  drawFeatureMapDetails(ctx, conv1Boxes);
  drawFeatureMapDetails(ctx, poolBoxes);
  drawFeatureMapDetails(ctx, conv2Boxes);
  drawCnnMapNeurons(ctx, outputPositions, layerX[4], false, true);

  ctx.fillStyle = 'rgba(224,232,240,0.6)';
  ctx.font = '11px "Segoe UI"';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < CNN_MAP_COLUMNS.length; i++) {
    ctx.fillText(CNN_MAP_COLUMNS[i].label, layerX[i], MAP_TOP_BOTTOM_MARGIN / 2 + 8);
  }
}

function cnnMapColumnX(width: number): number[] {
  const left = MAP_COLUMN_MARGIN;
  const right = width - MAP_COLUMN_MARGIN;
  const span = right - left;
  return [
    left,
    left + span * 0.25,
    left + span * 0.38,
    left + span * 0.68,
    right,
  ];
}

function buildCnnMapLayerPositions(total: number, canvasH: number): MapPosition[] {
  const topY = MAP_TOP_BOTTOM_MARGIN + 18;
  const bottomY = canvasH - MAP_TOP_BOTTOM_MARGIN;
  const span = bottomY - topY;
  if (total <= MAP_FULL_DRAW_MAX_COUNT) {
    const out: MapPosition[] = new Array(total);
    const step = total === 1 ? 0 : span / (total - 1);
    for (let i = 0; i < total; i++) out[i] = { y: topY + i * step, actualIndex: i };
    return out;
  }
  return [];
}

function buildFeatureMapBoxes(x: number, count: number, gridSide: number, canvasH: number): FeatureMapBox[] {
  const topY = MAP_TOP_BOTTOM_MARGIN + 18;
  const bottomY = canvasH - MAP_TOP_BOTTOM_MARGIN;
  const span = bottomY - topY;
  const gap = count > 4 ? 4 : 8;
  const maxSize = count > 4 ? 26 : 44;
  const side = clamp((span - gap * (count - 1)) / count, 16, maxSize);
  const totalHeight = count * side + (count - 1) * gap;
  const startY = topY + (span - totalHeight) / 2;
  const boxes: FeatureMapBox[] = [];
  for (let i = 0; i < count; i++) {
    const y = startY + i * (side + gap) + side / 2;
    boxes.push({ x, y, size: side, actualIndex: i, gridSide, cells: buildFeatureMapCells(x, y, side, gridSide, i) });
  }
  return boxes;
}

function buildFeatureMapCells(x: number, y: number, size: number, gridSide: number, channel: number): MapCell[] {
  const padding = gridSide >= 10 ? Math.max(3, size * 0.1) : Math.max(3, size * 0.16);
  const step = gridSide <= 1 ? 0 : (size - padding * 2) / (gridSide - 1);
  const startX = x - size / 2 + padding;
  const startY = y - size / 2 + padding;
  const cells: MapCell[] = [];
  for (let row = 0; row < gridSide; row++) {
    for (let col = 0; col < gridSide; col++) {
      const cellIndex = row * gridSide + col;
      cells.push({
        x: startX + col * step,
        y: startY + row * step,
        actualIndex: channel * gridSide * gridSide + cellIndex,
      });
    }
  }
  return cells;
}

function drawConv1WeightConnections(
  ctx: CanvasRenderingContext2D,
  from: readonly MapPosition[],
  fromX: number,
  to: readonly FeatureMapBox[],
): void {
  for (const box of to) {
    for (let row = 0; row < box.gridSide; row++) {
      for (let col = 0; col < box.gridSide; col++) {
        const target = featureMapCellAt(box, row, col);
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const inputIndex = (row + ky) * INPUT_SIDE + (col + kx);
            const source = from[inputIndex];
            const weight = cnnMapSnapshot ? cnnMapWeight(0, ky * 3 + kx, box.actualIndex, cnnMapSnapshot) : null;
            drawCnnWeightLine(ctx, fromX, source.y, target.x, target.y, weight);
          }
        }
      }
    }
  }
}

function drawConv2WeightConnections(ctx: CanvasRenderingContext2D, from: readonly FeatureMapBox[], to: readonly FeatureMapBox[]): void {
  for (const toBox of to) {
    for (let row = 0; row < toBox.gridSide; row++) {
      for (let col = 0; col < toBox.gridSide; col++) {
        const target = featureMapCellAt(toBox, row, col);
        for (const fromBox of from) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const source = featureMapCellAt(fromBox, row + ky, col + kx);
              const weight = cnnMapSnapshot ? cnnMapWeight(2, fromBox.actualIndex * 9 + ky * 3 + kx, toBox.actualIndex, cnnMapSnapshot) : null;
              drawCnnWeightLine(ctx, source.x, source.y, target.x, target.y, weight);
            }
          }
        }
      }
    }
  }
}

function drawFeatureMapToOutputConnections(
  ctx: CanvasRenderingContext2D,
  from: readonly FeatureMapBox[],
  to: readonly MapPosition[],
  toX: number,
): void {
  for (const toPoint of to) {
    for (const box of from) {
      for (const cell of box.cells) {
        const weight = cnnMapSnapshot ? cnnMapWeight(3, cell.actualIndex, toPoint.actualIndex, cnnMapSnapshot) : null;
        drawCnnWeightLine(ctx, cell.x, cell.y, toX, toPoint.y, weight, CNN_DENSE_LINE_WEIGHT_SCALE);
      }
    }
  }
}

function drawCnnWeightLine(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  weight: number | null,
  visualScale = REGULAR_NN_LINE_WEIGHT_SCALE,
): void {
  ctx.strokeStyle = cnnMapStrokeForWeight(weight, visualScale);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function featureMapCellAt(box: FeatureMapBox, row: number, col: number): MapCell {
  const safeRow = Math.max(0, Math.min(box.gridSide - 1, row));
  const safeCol = Math.max(0, Math.min(box.gridSide - 1, col));
  return box.cells[safeRow * box.gridSide + safeCol];
}

function drawCnnMapNeurons(
  ctx: CanvasRenderingContext2D,
  positions: readonly MapPosition[],
  x: number,
  isInput: boolean,
  isOutput: boolean,
): void {
  const r = computeCnnMapNeuronRadius(positions);
  const previousAlpha = ctx.globalAlpha;
  if (isInput) ctx.globalAlpha = 0.22;
  for (const position of positions) {
    const actual = position.actualIndex;
    let strokeColor = 'rgba(224,232,240,0.8)';
    let strokeWidth = 1.5;
    if (isOutput && cnnMapSnapshot && actual === cnnMapSnapshot.trueLabel) {
      strokeColor = '#ffd700';
      strokeWidth = 2.5;
    }

    ctx.fillStyle = '#102030';
    ctx.beginPath();
    ctx.arc(x, position.y, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.beginPath();
    ctx.arc(x, position.y, r, 0, Math.PI * 2);
    ctx.stroke();

    if (isOutput) {
      const isPred = cnnMapSnapshot?.predictedLabel === actual;
      const isTrue = cnnMapSnapshot?.trueLabel === actual;
      ctx.fillStyle = isPred ? (isTrue ? '#5bba6f' : '#ff6b6b') : 'rgba(224,232,240,0.6)';
      ctx.font = '11px "Segoe UI"';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${actual}`, x + r + 14, position.y);
    }
  }
  ctx.globalAlpha = previousAlpha;
}

function drawFeatureMapBackdrops(ctx: CanvasRenderingContext2D, boxes: readonly FeatureMapBox[]): void {
  for (const box of boxes) {
    ctx.fillStyle = '#102030';
    ctx.fillRect(box.x - box.size / 2, box.y - box.size / 2, box.size, box.size);
  }
}

function drawFeatureMapDetails(ctx: CanvasRenderingContext2D, boxes: readonly FeatureMapBox[]): void {
  for (const box of boxes) {
    ctx.strokeStyle = 'rgba(224,232,240,0.8)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(box.x - box.size / 2, box.y - box.size / 2, box.size, box.size);

    const dotRadius = box.gridSide >= 10 ? 0.65 : box.gridSide >= 6 ? 0.85 : 1.05;
    for (const cell of box.cells) {
      ctx.fillStyle = '#102030';
      ctx.beginPath();
      ctx.arc(cell.x, cell.y, dotRadius, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = 'rgba(224,232,240,0.8)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.arc(cell.x, cell.y, dotRadius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function computeCnnMapNeuronRadius(positions: readonly MapPosition[]): number {
  if (positions.length <= 1) return MAP_MAX_NEURON_RADIUS;
  const spacing = Math.abs(positions[1].y - positions[0].y);
  return clamp(spacing * 0.4, MAP_MIN_NEURON_RADIUS, MAP_MAX_NEURON_RADIUS);
}

function cnnMapStrokeForWeight(weight: number | null, visualScale = REGULAR_NN_LINE_WEIGHT_SCALE): string {
  if (weight == null || !Number.isFinite(weight)) return 'rgba(156,193,224,0.18)';
  const alpha = clamp(Math.abs(weight) / visualScale, 0.05, 0.85);
  return weight >= 0
    ? `rgba(91,186,111,${alpha.toFixed(3)})`
    : `rgba(255,107,107,${alpha.toFixed(3)})`;
}

function cnnMapWeight(layerIndex: number, fromIndex: number, toIndex: number, snapshot: ConvNetSnapshot): number | null {
  switch (layerIndex) {
    case 0:
      return snapshot.conv1Kernels[toIndex * 9 + fromIndex] ?? null;
    case 2: {
      const inputChannel = Math.floor(fromIndex / 9);
      const kernelCell = fromIndex % 9;
      return snapshot.conv2Kernels[(toIndex * CONV1_FILTERS + inputChannel) * 9 + kernelCell] ?? null;
    }
    case 3:
      return snapshot.fcWeights[toIndex * FC_INPUT + fromIndex] ?? null;
    default:
      return null;
  }
}

function drawFocusedCnnMapLine(
  ctx: CanvasRenderingContext2D,
  layerX: readonly number[],
  inputPositions: readonly MapPosition[],
  conv1Boxes: readonly FeatureMapBox[],
  poolBoxes: readonly FeatureMapBox[],
  conv2Boxes: readonly FeatureMapBox[],
  outputPositions: readonly MapPosition[],
): void {
  const line = mapLineForSelection(focusedCnnParam);
  if (!line) return;

  const oldAlpha = ctx.globalAlpha;
  const oldStroke = ctx.strokeStyle;
  const oldWidth = ctx.lineWidth;
  const oldCap = ctx.lineCap;
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';

  const draw = (x1: number, y1: number, x2: number, y2: number) => {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };

  if (line.kind === 'conv1') {
    const box = conv1Boxes[line.outChannel];
    const source = inputPositions[(line.outRow + line.ky) * INPUT_SIDE + line.outCol + line.kx];
    if (box && source) {
      const target = featureMapCellAt(box, line.outRow, line.outCol);
      draw(layerX[0], source.y, target.x, target.y);
    }
  } else if (line.kind === 'conv2') {
    const fromBox = poolBoxes[line.inChannel];
    const toBox = conv2Boxes[line.outChannel];
    if (fromBox && toBox) {
      const source = featureMapCellAt(fromBox, line.outRow + line.ky, line.outCol + line.kx);
      const target = featureMapCellAt(toBox, line.outRow, line.outCol);
      draw(source.x, source.y, target.x, target.y);
    }
  } else {
    const from = findFeatureMapCellByActualIndex(conv2Boxes, line.col);
    const to = findMapPositionByActualIndex(outputPositions, line.row);
    if (from && to) draw(from.x, from.y, layerX[4], to.y);
  }

  ctx.globalAlpha = oldAlpha;
  ctx.strokeStyle = oldStroke;
  ctx.lineWidth = oldWidth;
  ctx.lineCap = oldCap;
}

function mapLineForSelection(selection: CnnParamSelection | null): CnnMapLineRef | null {
  if (!selection) return null;
  if (selection.kind === 'convWeight') {
    if (selection.line) return selection.line;
    if (selection.layer === 1) {
      return {
        kind: 'conv1',
        outChannel: selection.outChannel,
        ky: selection.ky,
        kx: selection.kx,
        outRow: Math.floor(CONV1_OUT / 2),
        outCol: Math.floor(CONV1_OUT / 2),
      };
    }
    return {
      kind: 'conv2',
      outChannel: selection.outChannel,
      inChannel: selection.inChannel,
      ky: selection.ky,
      kx: selection.kx,
      outRow: Math.floor(CONV2_OUT / 2),
      outCol: Math.floor(CONV2_OUT / 2),
    };
  }
  if (selection.kind === 'denseWeight') {
    return selection.line ?? { kind: 'fc', row: selection.row, col: selection.col };
  }
  return null;
}

function hitTestCnnMapWeightLine(px: number, py: number, width: number, height: number): CnnParamSelection | null {
  if (!cnnMapSnapshot) return null;

  const layerX = cnnMapColumnX(width);
  const inputPositions = buildCnnMapLayerPositions(INPUT_SIDE * INPUT_SIDE, height);
  const outputPositions = buildCnnMapLayerPositions(OUTPUTS, height);
  const conv1Boxes = buildFeatureMapBoxes(layerX[1], CONV1_FILTERS, CONV1_OUT, height);
  const poolBoxes = buildFeatureMapBoxes(layerX[2], CONV1_FILTERS, POOL1_OUT, height);
  const conv2Boxes = buildFeatureMapBoxes(layerX[3], CONV2_FILTERS, CONV2_OUT, height);

  let bestDist = 5;
  let best: CnnParamSelection | null = null;
  const consider = (x1: number, y1: number, x2: number, y2: number, selection: CnnParamSelection) => {
    const distance = distToSegment(px, py, x1, y1, x2, y2);
    if (distance < bestDist) {
      bestDist = distance;
      best = selection;
    }
  };

  for (const box of conv1Boxes) {
    for (let row = 0; row < box.gridSide; row++) {
      for (let col = 0; col < box.gridSide; col++) {
        const target = featureMapCellAt(box, row, col);
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const source = inputPositions[(row + ky) * INPUT_SIDE + col + kx];
            if (!source) continue;
            const line: CnnMapLineRef = { kind: 'conv1', outChannel: box.actualIndex, ky, kx, outRow: row, outCol: col };
            consider(layerX[0], source.y, target.x, target.y, {
              kind: 'convWeight',
              layer: 1,
              outChannel: box.actualIndex,
              inChannel: 0,
              ky,
              kx,
              line,
            });
          }
        }
      }
    }
  }

  for (const toBox of conv2Boxes) {
    for (let row = 0; row < toBox.gridSide; row++) {
      for (let col = 0; col < toBox.gridSide; col++) {
        const target = featureMapCellAt(toBox, row, col);
        for (const fromBox of poolBoxes) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const source = featureMapCellAt(fromBox, row + ky, col + kx);
              const line: CnnMapLineRef = {
                kind: 'conv2',
                outChannel: toBox.actualIndex,
                inChannel: fromBox.actualIndex,
                ky,
                kx,
                outRow: row,
                outCol: col,
              };
              consider(source.x, source.y, target.x, target.y, {
                kind: 'convWeight',
                layer: 2,
                outChannel: toBox.actualIndex,
                inChannel: fromBox.actualIndex,
                ky,
                kx,
                line,
              });
            }
          }
        }
      }
    }
  }

  for (const toPoint of outputPositions) {
    for (const box of conv2Boxes) {
      for (const cell of box.cells) {
        const line: CnnMapLineRef = { kind: 'fc', row: toPoint.actualIndex, col: cell.actualIndex };
        consider(cell.x, cell.y, layerX[4], toPoint.y, {
          kind: 'denseWeight',
          row: toPoint.actualIndex,
          col: cell.actualIndex,
          line,
        });
      }
    }
  }

  return best;
}

function findFeatureMapCellByActualIndex(boxes: readonly FeatureMapBox[], actualIndex: number): MapCell | null {
  for (const box of boxes) {
    const found = box.cells.find(cell => cell.actualIndex === actualIndex);
    if (found) return found;
  }
  return null;
}

function findMapPositionByActualIndex(positions: readonly MapPosition[], actualIndex: number): MapPosition | null {
  return positions.find(position => position.actualIndex === actualIndex) ?? null;
}

function distToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function onCnnMapClick(event: MouseEvent): void {
  const element = cnnMapElement;
  if (!element || !cnnStepsEnabled || !cnnMapSnapshot) return;
  const rect = element.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const weightHit = hitTestCnnMapWeightLine(x, y, element.clientWidth, element.clientHeight);
  if (weightHit) {
    selectCnnParam(weightHit, true);
    return;
  }
  const w = element.clientWidth;
  const layerX = cnnMapColumnX(w);
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < layerX.length; i++) {
    const distance = Math.abs(x - layerX[i]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  const stepIndex = CNN_MAP_COLUMNS[bestIndex].step;
  if (bestDistance <= 42 && stepIndex != null) setActiveCnnStep?.(stepIndex, true);
}

function updateCnnMapLineWeights(snapshot: ConvNetSnapshot): void {
  cnnMapSnapshot = snapshot;
  drawCnnNetworkMap();
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function buildCnnStepGuide(): { strip: HTMLElement; explainer: HTMLElement } {
  const strip = document.createElement('div');
  strip.className = 'phase-strip cnn-step-strip';
  const explainer = document.createElement('div');
  explainer.className = 'phase-explainer cnn-step-explainer';
  const chips: HTMLElement[] = [];
  let activeIndex: number | null = null;

  setActiveCnnStep = (index: number, scroll = false) => {
    if (!cnnStepsEnabled) return;
    const step = CNN_STEPS.find(item => item.stepIndex === index);
    if (!step) return;
    activeIndex = index;
    chips.forEach((chip, i) => chip.classList.toggle('active', CNN_STEPS[i].stepIndex === index));
    explainer.innerHTML = `<div class="phase-detail">${step.detail}</div>`;
    renderMath(explainer);
    document.querySelectorAll('.cnn-step-target').forEach(el => el.classList.remove('active-cnn-step'));
    document.querySelectorAll(`[data-cnn-step="${index}"]`).forEach(el => el.classList.add('active-cnn-step'));
    const target = document.getElementById(step.targetId);
    if (target?.classList.contains('cnn-step-target')) target.classList.add('active-cnn-step');
    if (scroll && target) target.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
  };

  setCnnStepsEnabled = (on: boolean) => {
    cnnStepsEnabled = on;
    document.body.classList.toggle('cnn-steps-locked', !on);
    if (!on) {
      activeIndex = null;
      chips.forEach(chip => {
        chip.classList.remove('active', 'done', 'pinned');
        chip.classList.add('disabled');
        chip.setAttribute('aria-disabled', 'true');
      });
      document.querySelectorAll('.cnn-step-target').forEach(el => el.classList.remove('active-cnn-step'));
      explainer.innerHTML = `<div class="phase-detail">${CNN_LOCKED_EXPLAINER}</div>`;
      return;
    }
    chips.forEach((chip, i) => {
      chip.classList.remove('disabled');
      chip.setAttribute('aria-disabled', 'false');
      chip.classList.toggle('active', activeIndex === CNN_STEPS[i].stepIndex);
    });
    if (activeIndex == null) setActiveCnnStep?.(1, false);
  };

  CNN_STEPS.forEach((step) => {
    const chip = document.createElement('div');
    chip.className = 'phase-chip disabled';
    chip.setAttribute('aria-disabled', 'true');
    chip.textContent = step.label;
    chip.addEventListener('click', () => {
      if (!cnnStepsEnabled) return;
      setActiveCnnStep?.(step.stepIndex, true);
    });
    chips.push(chip);
    strip.appendChild(chip);
  });

  explainer.innerHTML = `<div class="phase-detail">${CNN_LOCKED_EXPLAINER}</div>`;
  return { strip, explainer };
}

// ─── CNN step-row builder ─────────────────────────────────────────────────

function buildCnnParamsView(weights: HTMLElement, biases: HTMLElement): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'cnn-params-pair';
  const content = document.createElement('div');
  content.className = 'cnn-params-pair-content';
  content.appendChild(weights);
  content.appendChild(biases);
  wrap.appendChild(content);
  return wrap;
}

function setupConvParamsAutoSize(
  wrap: HTMLElement,
  filters: FilterBankView,
  biases: VectorView,
  outC: number,
  kernelW: number,
  minCell: number,
  maxCell: number,
  labelW = 18,
): void {
  let lastKey = '';
  let raf = 0;
  const schedule = () => {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      raf = 0;
      const availableW = wrap.clientWidth;
      if (availableW <= 0) return;
      let biasW = 42;
      let cell = minCell;
      const pairGap = 6;
      const rowGap = 3;
      for (let i = 0; i < 3; i++) {
        const fixedW = labelW + pairGap + biasW + outC * rowGap;
        cell = clamp((availableW - fixedW) / (outC * kernelW), minCell, maxCell);
        biasW = clamp(Math.round(cell * 2.2), 38, 56);
      }
      const filterCell = Number(cell.toFixed(2));
      const biasCellW = Math.round(biasW);
      const biasCellH = clamp(Math.round(filterCell * 0.86), 15, 22);
      const key = `${filterCell}:${biasCellW}:${biasCellH}`;
      if (key === lastKey) return;
      lastKey = key;
      filters.setOptions({ cellSize: filterCell });
      biases.setOptions({ cellWidth: biasCellW, cellHeight: biasCellH });
      renderCnnParameterViews();
    });
  };
  new ResizeObserver(schedule).observe(wrap);
  schedule();
}

function setupDenseParamsAutoSize(wrap: HTMLElement): void {
  let lastKey = '';
  let raf = 0;
  const schedule = () => {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      raf = 0;
      const availableW = wrap.clientWidth;
      if (availableW <= 0) return;
      const labelW = 20;
      const pairGap = 6;
      const biasW = 42;
      const isMobile = window.matchMedia('(max-width: 980px)').matches;
      const fitCellW = (availableW - labelW - pairGap - biasW) / FC_INPUT;
      const cellW = Number((isMobile ? 8 : clamp(fitCellW, 3.5, 14)).toFixed(2));
      const cellH = cellW;
      const outputCellH = 18;
      const key = `${cellW}:${cellH}:${outputCellH}`;
      if (key === lastKey) return;
      lastKey = key;
      fcWeights.setOptions({ cellWidth: cellW, cellHeight: cellH });
      fcBiasVec.setOptions({ cellWidth: biasW, cellHeight: cellH });
      targetVec.setOptions({ cellHeight: outputCellH });
      fcOutVec.setOptions({ cellHeight: outputCellH });
      outputVec.setOptions({ cellHeight: outputCellH });
      fcSigmoidPrimeVec.setOptions({ cellHeight: outputCellH });
      fcErrorVec.setOptions({ cellHeight: outputCellH });
      fcDeltaVec.setOptions({ cellHeight: outputCellH });
      renderCnnParameterViews();
      renderCnnOutputState(latestSnapshot);
    });
  };
  new ResizeObserver(schedule).observe(wrap);
  schedule();
}

function buildCnnStepRow(opts: {
  id: string;
  stepIndex: number;
  title?: string;
  sections: { label: string; view: HTMLElement; className?: string }[];
}): HTMLElement {
  const row = document.createElement('section');
  row.id = opts.id;
  row.className = 'cnn-step-row cnn-step-target';
  row.dataset.cnnStep = `${opts.stepIndex}`;

  if (opts.title) {
    const title = document.createElement('div');
    title.className = 'panel-header cnn-step-row-header';
    title.innerHTML = opts.title;
    renderMath(title);
    row.appendChild(title);
  }

  const visuals = document.createElement('div');
  visuals.className = 'cnn-step-visuals';
  for (const sec of opts.sections) {
    const card = document.createElement('div');
    card.className = `cnn-step-section ${sec.className ?? ''}`.trim();
    if (sec.label) {
      const lbl = document.createElement('div');
      lbl.className = 'layer-section-label cnn-step-section-label';
      lbl.innerHTML = sec.label;
      renderMath(lbl);
      card.appendChild(lbl);
    }
    const body = document.createElement('div');
    body.className = 'cnn-step-section-body';
    body.appendChild(sec.view);
    card.appendChild(body);
    visuals.appendChild(card);
  }
  row.appendChild(visuals);
  return row;
}

function wireCnnParameterInteractions(): void {
  conv1Filters.element.addEventListener('cell-click', (e) => {
    const d = (e as CustomEvent<FilterBankCell>).detail;
    selectCnnParam({ kind: 'convWeight', layer: 1, ...d });
  });
  conv2Filters.element.addEventListener('cell-click', (e) => {
    const d = (e as CustomEvent<FilterBankCell>).detail;
    selectCnnParam({ kind: 'convWeight', layer: 2, ...d });
  });
  conv1BiasVec.element.addEventListener('cell-click', (e) => {
    const { index } = (e as CustomEvent<{ index: number }>).detail;
    selectCnnParam({ kind: 'convBias', layer: 1, index });
  });
  conv2BiasVec.element.addEventListener('cell-click', (e) => {
    const { index } = (e as CustomEvent<{ index: number }>).detail;
    selectCnnParam({ kind: 'convBias', layer: 2, index });
  });
  fcWeights.element.addEventListener('cell-click', (e) => {
    const { row, col } = (e as CustomEvent<{ row: number; col: number }>).detail;
    selectCnnParam({ kind: 'denseWeight', row, col });
  });
  fcBiasVec.element.addEventListener('cell-click', (e) => {
    const { index } = (e as CustomEvent<{ index: number }>).detail;
    selectCnnParam({ kind: 'denseBias', index });
  });
}

function selectCnnParam(selection: CnnParamSelection, scroll = false): void {
  focusedCnnParam = selection;
  clearCnnParamHighlights();

  if (selection.kind === 'convWeight') {
    const filterCell = {
      outChannel: selection.outChannel,
      inChannel: selection.inChannel,
      ky: selection.ky,
      kx: selection.kx,
    };
    if (selection.layer === 1) conv1Filters.setHighlight(filterCell);
    else conv2Filters.setHighlight(filterCell);
  } else if (selection.kind === 'convBias') {
    if (selection.layer === 1) conv1BiasVec.setHighlight(selection.index);
    else conv2BiasVec.setHighlight(selection.index);
  } else if (selection.kind === 'denseWeight') {
    fcWeights.setHighlight({ row: selection.row, col: selection.col });
    setCnnOutputHighlight(selection.row);
  } else {
    fcBiasVec.setHighlight(selection.index);
    setCnnOutputHighlight(selection.index);
  }

  setActiveCnnStep?.(cnnStepForSelection(selection), scroll);
  drawCnnNetworkMap();
}

function clearCnnParamSelection(): void {
  focusedCnnParam = null;
  clearCnnParamHighlights();
  drawCnnNetworkMap();
}

function clearCnnParamHighlights(): void {
  conv1Filters.setHighlight(null);
  conv2Filters.setHighlight(null);
  conv1BiasVec.setHighlight(null);
  conv2BiasVec.setHighlight(null);
  fcWeights.setHighlight(null);
  fcBiasVec.setHighlight(null);
  setCnnOutputHighlight(null);
}

function setCnnOutputHighlight(index: number | null): void {
  targetVec.setHighlight(index);
  fcOutVec.setHighlight(index);
  outputVec.setHighlight(index);
  fcSigmoidPrimeVec.setHighlight(index);
  fcErrorVec.setHighlight(index);
  fcDeltaVec.setHighlight(index);
}

function cnnStepForSelection(selection: CnnParamSelection): number {
  if (selection.kind === 'convWeight' || selection.kind === 'convBias') {
    return selection.layer === 1 ? 1 : 3;
  }
  return 4;
}

// ─── Worker message handler ───────────────────────────────────────────────

function handleWorkerMessage(ev: MessageEvent<ConvWorkerToMain>): void {
  const msg = ev.data;
  const nodes = (window as any).__nodes;
  switch (msg.type) {
    case 'status':
      nodes.statusText.textContent = msg.text;
      nodes.statusText.style.color = '';
      break;
    case 'sample': {
      latestSnapshot = msg.snapshot;
      if (!cnnStepsEnabled) {
        setCnnStepsEnabled?.(true);
        setActiveCnnStep?.(1, false);
      }
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
      nodes.sampleLossEl.innerHTML = `$C=\\tfrac{1}{2}\\sum_k(a_{3,k}-y_k)^2=$ <span class="cost-result">$\\;${c.toFixed(4)}$</span>`;
      renderMath(nodes.sampleLossEl);

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
      nodes.lrInput.disabled = false;
      nodes.statusText.textContent = 'Training complete.';
      worker.terminate();
      worker = createWorker();
      break;
    case 'error':
      nodes.statusText.textContent = `Error: ${msg.message}`;
      nodes.statusText.style.color = 'var(--red)';
      nodes.startBtn.disabled = false;
      nodes.stopBtn.disabled = true;
      nodes.nextBtn.disabled = true;
      nodes.lrInput.disabled = false;
      worker.terminate();
      worker = createWorker();
      break;
  }
}

// ─── Rendering ────────────────────────────────────────────────────────────

function renderSnapshot(snap: ConvNetSnapshot): void {
  updateCnnMapLineWeights(snap);
  renderCnnParameterViews(snap);
  conv1Pre.update(snap.conv1Pre, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  conv1Relu.update(snap.conv1Relu, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  pool1View.update(snap.pool1Out, CONV1_FILTERS, POOL1_OUT, POOL1_OUT);
  conv2Pre.update(snap.conv2Pre, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  conv2Relu.update(snap.conv2Relu, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  renderCnnOutputState(snap);
}

function renderCnnOutputState(snap: ConvNetSnapshot | null): void {
  if (!snap) {
    targetVec.update(null, OUTPUTS);
    fcOutVec.update(null, OUTPUTS);
    outputVec.update(null, OUTPUTS);
    fcSigmoidPrimeVec.update(null, OUTPUTS);
    fcErrorVec.update(null, OUTPUTS);
    fcDeltaVec.update(null, OUTPUTS);
    return;
  }

  const sigmoidPrime = new Float64Array(OUTPUTS);
  const error = new Float64Array(OUTPUTS);
  const delta = new Float64Array(OUTPUTS);
  for (let k = 0; k < OUTPUTS; k++) {
    const a = snap.output[k];
    sigmoidPrime[k] = a * (1 - a);
    error[k] = a - snap.target[k];
    delta[k] = error[k] * sigmoidPrime[k];
  }
  targetVec.update(snap.target, OUTPUTS);
  fcOutVec.update(snap.fcOut, OUTPUTS);
  outputVec.update(snap.output, OUTPUTS);
  fcSigmoidPrimeVec.update(sigmoidPrime, OUTPUTS);
  fcErrorVec.update(error, OUTPUTS);
  fcDeltaVec.update(delta, OUTPUTS);
}

function renderCnnParameterViews(snap = latestSnapshot): void {
  if (!snap) return;
  conv1Filters.update(snap.conv1Kernels, CONV1_FILTERS, 1, 3, 3);
  conv1BiasVec.update(snap.conv1Biases, CONV1_FILTERS);
  conv2Filters.update(snap.conv2Kernels, CONV2_FILTERS, CONV1_FILTERS, 3, 3);
  conv2BiasVec.update(snap.conv2Biases, CONV2_FILTERS);
  fcWeights.update(snap.fcWeights, OUTPUTS, FC_INPUT);
  fcBiasVec.update(snap.fcBiases, OUTPUTS);
}

function resetCnnLessonDisplay(): void {
  latestSnapshot = null;
  cnnMapSnapshot = null;
  clearCnnParamSelection();
  setCnnStepsEnabled?.(false);
  drawCnnNetworkMap();

  conv1Filters.update(null, CONV1_FILTERS, 1, 3, 3);
  conv1BiasVec.update(null, CONV1_FILTERS);
  conv1Pre.update(null, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  conv1Relu.update(null, CONV1_FILTERS, CONV1_OUT, CONV1_OUT);
  pool1View.update(null, CONV1_FILTERS, POOL1_OUT, POOL1_OUT);
  conv2Filters.update(null, CONV2_FILTERS, CONV1_FILTERS, 3, 3);
  conv2BiasVec.update(null, CONV2_FILTERS);
  conv2Pre.update(null, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  conv2Relu.update(null, CONV2_FILTERS, CONV2_OUT, CONV2_OUT);
  fcWeights.update(null, OUTPUTS, FC_INPUT);
  fcBiasVec.update(null, OUTPUTS);
  renderCnnOutputState(null);

  const sampleLossEl = document.getElementById('sample-loss');
  if (sampleLossEl) {
    sampleLossEl.innerHTML = '$C=\\tfrac{1}{2}\\sum_k(a_{3,k}-y_k)^2$';
    renderMath(sampleLossEl);
  }
  const trueLabelEl = document.getElementById('true-label');
  if (trueLabelEl) trueLabelEl.textContent = '—';
  const predictedLabelEl = document.getElementById('predicted-label');
  if (predictedLabelEl) {
    predictedLabelEl.textContent = '—';
    predictedLabelEl.style.color = '';
  }
  clearSampleImage();
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

function clearSampleImage(): void {
  const ctx = sampleCanvas.getContext('2d')!;
  ctx.clearRect(0, 0, sampleCanvas.width, sampleCanvas.height);
}

function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}
