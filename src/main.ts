// Top-level orchestration. Owns the worker, the phase state machine, and the
// composed UI components. Mirrors MainWindow.xaml.cs from the WPF original —
// the C# Dispatcher.Invoke pattern is replaced by worker.onmessage on the main
// thread, and the ManualResetEventSlim step-gate is replaced by posting
// {type:'stepNext'} back to the worker.

import './style.css';
import { Phase, type SampleSnapshot, type MainToWorker, type WorkerToMain } from './types';
import { PhaseChips } from './ui/phase-chips';
import { NetworkDiagram, type DiagramState } from './ui/diagram';
import { MatrixView } from './ui/matrix-view';
import { VectorView } from './ui/vector-view';
import { FocusPanel } from './ui/focus-panel';
import { HiddenDeltaPanel } from './ui/hidden-delta-panel';
import { LineChart } from './ui/charts';
import { renderMath } from './ui/katex-util';
import { INPUT_SIDE } from './mnist';

const SAMPLE_DISPLAY = 200;
const HIDDEN_SIZE = 30;
const OUTPUT_SIZE = 10;

// ─── State held on the main thread ────────────────────────────────────────

let currentPhase: Phase = Phase.Forward;
let stepMode = true;
let latestSnapshot: SampleSnapshot | null = null;
let lastTrueLabel = -1;
let lastPredictedLabel = -1;
let focusedWeight: { row: number; col: number } | null = null;
let focusedHidden: number | null = null;
let miniBatchSize = 1;
let learningRate = 3.0;

// ─── Worker ───────────────────────────────────────────────────────────────

let worker = createWorker();
function post(msg: MainToWorker): void { worker.postMessage(msg); }
function createWorker(): Worker {
  const nextWorker = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' });
  nextWorker.onmessage = handleWorkerMessage;
  return nextWorker;
}

// ─── UI components ────────────────────────────────────────────────────────

const phaseChips = new PhaseChips();
const diagram = new NetworkDiagram();
const focusPanel = new FocusPanel();
const hiddenDeltaPanel = new HiddenDeltaPanel();

// 6 vector columns in Row 2 (left section).
const targetView      = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 1.0 });
const weightedInputView = new VectorView({ cellWidth: 34, cellHeight: 18 });
const outputActView   = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 1.0 });
const sigmoidPrimeView = new VectorView({ cellWidth: 34, cellHeight: 18, fixedScale: 0.25 });
const errorView       = new VectorView({ cellWidth: 34, cellHeight: 18 });
const deltaView       = new VectorView({ cellWidth: 34, cellHeight: 18 });
const outputVectorHeaders: HTMLElement[] = [];
const outputVectorViews = [targetView, weightedInputView, outputActView, sigmoidPrimeView, errorView, deltaView];

// Row 2 middle: a[1] (1×30) above per-sample ΔW[2] (10×30).
const hiddenActMatrix = new MatrixView({ cellWidth: 26, cellHeight: 18, fixedScale: 1.0 });
const perSampleNablaW2 = new MatrixView({ cellWidth: 26, cellHeight: 18, rowLabelPrefix: 'y', colLabelPrefix: 'h' });

// Row 3 left: W[2] (10×30) + b[2] + Δb[2].
const w2View = new MatrixView({ cellWidth: 26, cellHeight: 18, rowLabelPrefix: 'y', colLabelPrefix: 'h', fixedScale: 3.0 });
const b2View = new VectorView({ cellWidth: 26, cellHeight: 18, fixedScale: 3.0, title: 'b' });
const nablaB2View = new VectorView({ cellWidth: 26, cellHeight: 18, title: 'Δb' });

// Charts.
const lossChart = new LineChart({ title: 'Test-set loss $\\left(\\frac12\\lVert a-y\\rVert^2\\right)$', color: '#5BBA6F' });
const accuracyChart = new LineChart({ title: 'Test-set accuracy', color: '#FFB347', fixedRange: { min: 0, max: 1 } });

// Sample image canvas (INPUT_SIDE×INPUT_SIDE → 280×280 NearestNeighbor).
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

  // ── HEADER: lesson title + status + phase guide ──────────────────────────
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
  nav.innerHTML = '<a href="/" class="current">Regular NN</a><a href="/convolutional/">CNN</a>';
  headerActions.appendChild(nav);
  const trainingInfo = document.createElement('span');
  trainingInfo.className = 'info-tip compact-info site-info';
  trainingInfo.tabIndex = 0;
  trainingInfo.setAttribute('aria-label', 'About this training demo');
  trainingInfo.innerHTML = `?
    <span class="info-tip-content">
      This demo trains on MNIST digits: $60{,}000$ training samples over $30$ epochs.<br>
      The network updates after every example.<br>
      Batch size is fixed at $1$ for clarity; real training usually averages gradients over mini-batches.<br>
      The step size is controlled by $\\eta$ below.
    </span>`;
  headerActions.appendChild(trainingInfo);
  renderMath(trainingInfo);
  titleRow.appendChild(headerActions);
  header.appendChild(titleRow);

  const statusText = document.createElement('span');
  statusText.id = 'status-text';
  statusText.textContent = 'Ready. Press Start to begin training.';
  statusText.style.flex = '1';
  const miniBatchText = document.createElement('span');
  miniBatchText.id = 'minibatch-text';
  miniBatchText.className = 'status-pill';
  miniBatchText.textContent = 'Update not started';

  header.appendChild(phaseChips.element);
  header.appendChild(phaseChips.explainerElement);
  app.appendChild(header);

  // ── MAIN: scrollable content area ───────────────────────────────────────
  const main = document.createElement('div');
  main.id = 'main';

  // Row 1: the minimal teaching surface — sample + diagram.
  const row1 = document.createElement('div');
  row1.className = 'row primary-stage';
  row1.style.display = 'grid';
  row1.style.gridTemplateColumns = `${SAMPLE_DISPLAY + 28}px 1fr`;
  row1.style.gap = '12px';
  row1.style.flex = '0 0 290px';
  const sampleCol = document.createElement('div');
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
  row1.appendChild(sampleCol);
  const diagramPanel = document.createElement('div');
  diagramPanel.className = 'diagram-panel';
  const diagramHeader = document.createElement('div');
  diagramHeader.className = 'panel-header diagram-header';
  diagramHeader.innerHTML = `<span>Network map</span><small>Click hidden → output lines to inspect one weight.</small>
    <span class="info-tip compact-info" tabindex="0" aria-label="Why is the input side dimmed?">?
      <span class="info-tip-content">
        This demo focuses interaction on the hidden → output layer so the math stays readable.<br>
        The dimmed input → hidden weights and first-layer biases are still being updated in the background using the same backprop idea.
      </span>
    </span>`;
  diagramPanel.appendChild(diagramHeader);
  diagramPanel.appendChild(diagram.element);
  row1.appendChild(diagramPanel);
  main.appendChild(row1);

  const detailsBody = document.createElement('div');
  detailsBody.className = 'details-body';
  const learningWorkspace = document.createElement('div');
  learningWorkspace.className = 'row learning-workspace';
  const learningLeft = document.createElement('div');
  learningLeft.className = 'learning-left-stack';

  // Row 2: backprop math — vectors | a[1]+perSample | ΔW[2].
  const row2 = document.createElement('div');
  row2.className = 'row panel output-learning-panel';
  row2.innerHTML = '<div class="panel-header">Output layer learning: compare prediction to target, form $\\delta_2$, then compute the gradient for $W_2$</div>';
  renderMath(row2);
  const r2Body = document.createElement('div');
  r2Body.className = 'output-learning-body';

  // Vectors with column headers above each.
  const r2Vectors = document.createElement('div');
  r2Vectors.className = 'output-vector-grid';
  const vectorHeaders = ['$y$', '$z_2$', '$a_2$', "$\\sigma'(z_2)$", '$a_2-y$', '$\\delta_2$'];
  for (const h of vectorHeaders) {
    const hd = document.createElement('div');
    hd.className = 'output-vector-heading';
    hd.innerHTML = h;
    renderMath(hd);
    hd.classList.add('data-hidden');
    outputVectorHeaders.push(hd);
    r2Vectors.appendChild(hd);
  }
  for (const v of outputVectorViews) {
    v.element.classList.add('data-hidden');
    r2Vectors.appendChild(v.element);
  }
  const psBlock = document.createElement('div');
  psBlock.className = 'output-learning-block wide';
  const psHeader = document.createElement('div');
  psHeader.className = 'section-header compact gradient-formula-heading';
  psHeader.innerHTML = '$\\frac{\\partial C}{\\partial w_{h_c \\to y_r}} = \\delta_{2,r}a_{1,c}$&nbsp; for each cell';
  renderMath(psHeader);
  psHeader.classList.add('data-hidden');
  const costHeader = document.createElement('div');
  costHeader.id = 'output-cost-equation';
  costHeader.className = 'output-cost-equation';
  costHeader.innerHTML = '$C=\\tfrac{1}{2}\\sum_k(a_{2,k}-y_k)^2$';
  renderMath(costHeader);
  costHeader.classList.add('data-hidden');
  const outputHeaderRow = document.createElement('div');
  outputHeaderRow.className = 'output-equation-row';
  outputHeaderRow.appendChild(psHeader);
  outputHeaderRow.appendChild(costHeader);
  psBlock.appendChild(outputHeaderRow);
  const gradientRow = document.createElement('div');
  gradientRow.className = 'output-gradient-row';
  perSampleNablaW2.element.classList.add('data-hidden');
  gradientRow.appendChild(perSampleNablaW2.element);
  gradientRow.appendChild(r2Vectors);
  psBlock.appendChild(gradientRow);
  r2Body.appendChild(psBlock);

  row2.appendChild(r2Body);
  learningLeft.appendChild(row2);

  // W[2] + b[2] + Δb[2].
  const wBox = document.createElement('div');
  wBox.className = 'panel';
  wBox.style.display = 'flex';
  wBox.style.flexDirection = 'column';
  wBox.style.gap = '0';
  const wHeaderRow = document.createElement('div');
  wHeaderRow.style.display = 'flex';
  wHeaderRow.style.justifyContent = 'space-between';
  wHeaderRow.style.alignItems = 'center';
  wHeaderRow.style.marginBottom = '-2px';
  wHeaderRow.innerHTML = `
    <span class="panel-header">Weights and biases being learned: $W_2$ and $b_2$</span>
    <span class="info-tip" tabindex="0" aria-label="What do weights and biases mean?">?
      <span class="info-tip-content">
        <b>Weight</b>: how strongly one hidden neuron influences one output digit.<br>
        <b>Bias</b>: the output digit's baseline push before hidden neurons vote.<br>
        <b>Δ</b>: the change computed from this sample; learning subtracts a scaled version of it.
      </span>
    </span>`;
  renderMath(wHeaderRow);
  wBox.appendChild(wHeaderRow);
  const wRow = document.createElement('div');
  wRow.style.display = 'flex';
  wRow.style.gap = '12px';
  wRow.style.alignItems = 'flex-start';
  wRow.appendChild(w2View.element);
  const bBlock = document.createElement('div');
  bBlock.appendChild(b2View.element);
  wRow.appendChild(bBlock);
  const dbBlock = document.createElement('div');
  dbBlock.appendChild(nablaB2View.element);
  wRow.appendChild(dbBlock);
  wBox.appendChild(wRow);
  learningLeft.appendChild(wBox);

  // Charts.
  const charts = document.createElement('div');
  charts.style.display = 'grid';
  charts.style.gridTemplateColumns = '1fr 1fr';
  charts.style.gap = '12px';
  charts.style.height = '135px';
  charts.appendChild(lossChart.element);
  charts.appendChild(accuracyChart.element);
  learningLeft.appendChild(charts);
  learningWorkspace.appendChild(learningLeft);
  learningWorkspace.appendChild(focusPanel.element);
  detailsBody.appendChild(learningWorkspace);

  const hiddenSection = document.createElement('section');
  hiddenSection.className = 'row hidden-backprop-shell is-collapsed';
  const hiddenToggle = document.createElement('button');
  hiddenToggle.type = 'button';
  hiddenToggle.className = 'hidden-collapse-toggle';
  hiddenToggle.setAttribute('aria-expanded', 'false');
  hiddenToggle.innerHTML = `
    <span class="collapse-caret" aria-hidden="true">›</span>
    <span class="collapse-title">Step 5: Hidden Layer Backpropagation</span>
    <small>Show hidden activations and $\\delta_1$ details</small>`;
  renderMath(hiddenToggle);
  hiddenSection.appendChild(hiddenToggle);

  const hiddenBottomRow = document.createElement('div');
  hiddenBottomRow.className = 'hidden-collapse-body';
  hiddenBottomRow.hidden = true;
  const hiddenNote = document.createElement('div');
  hiddenNote.className = 'hidden-collapse-note';
  hiddenNote.textContent =
    "The full first-layer update has many more weights than we can show comfortably here. These two tables give a focused view: the hidden activations show what the hidden layer produced, and the hidden δ panel shows how output-layer error is pushed backward to one hidden neuron.";
  hiddenBottomRow.appendChild(hiddenNote);
  const hiddenTablesRow = document.createElement('div');
  hiddenTablesRow.className = 'hidden-bottom-section';
  const hiddenActPanel = document.createElement('div');
  hiddenActPanel.className = 'panel hidden-activations-panel';
  const hiddenActHeader = document.createElement('div');
  hiddenActHeader.className = 'panel-header';
  hiddenActHeader.innerHTML = '$a_1$ (hidden activations, 30)';
  renderMath(hiddenActHeader);
  hiddenActPanel.appendChild(hiddenActHeader);
  hiddenActPanel.appendChild(hiddenActMatrix.element);
  hiddenTablesRow.appendChild(hiddenActPanel);
  hiddenTablesRow.appendChild(hiddenDeltaPanel.element);
  hiddenBottomRow.appendChild(hiddenTablesRow);
  hiddenSection.appendChild(hiddenBottomRow);
  hiddenToggle.addEventListener('click', () => {
    const expanded = hiddenToggle.getAttribute('aria-expanded') === 'true';
    hiddenToggle.setAttribute('aria-expanded', `${!expanded}`);
    hiddenBottomRow.hidden = expanded;
    hiddenSection.classList.toggle('is-collapsed', expanded);
  });
  detailsBody.appendChild(hiddenSection);

  main.appendChild(detailsBody);
  app.appendChild(main);

  // ── FOOTER: controls always at the bottom ───────────────────────────────
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

  const nextBtn = button('Next phase', () => onNext());
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
    phaseChips.setStepMode(stepMode);
    post({ type: 'setStepMode', on: stepMode });
    if (!stepMode) {
      currentPhase = Phase.HiddenDelta;
      applyCurrentPhase();
    } else {
      currentPhase = Phase.Forward;
      applyCurrentPhase();
    }
    nextBtn.disabled = !(stepMode && !stopBtn.disabled);
    updateNextButtonLabel();
  });
  const stepText = document.createElement('span');
  stepText.textContent = 'Walk through phases';
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
  lrLabel.textContent = 'Learning rate';
  const lrInput = document.createElement('input');
  lrInput.type = 'number';
  lrInput.min = '0.1';
  lrInput.max = '10';
  lrInput.step = '0.1';
  lrInput.value = `${learningRate}`;
  lrInput.addEventListener('change', () => { learningRate = Math.max(0.1, +lrInput.value || 3.0); lrInput.value = `${learningRate}`; });
  lrLabel.appendChild(lrInput);
  settingsGroup.appendChild(lrLabel);
  controls.appendChild(settingsGroup);

  app.appendChild(controls);

  // Cache nodes for later updates.
  (window as any).__nodes = {
    statusText, miniBatchText, startBtn, stopBtn, nextBtn, stepCb,
    outputCostEl: document.getElementById('output-cost-equation')!,
    trueLabelEl: document.getElementById('true-label')!,
    predictedLabelEl: document.getElementById('predicted-label')!,
    lrInput,
  };

  function onStart(): void {
    resetLessonDisplay();
    lossChart.clear();
    accuracyChart.clear();
    statusText.textContent = 'Loading training data...';
    post({ type: 'init', learningRate, miniBatchSize, epochs: 30, seed: 42, stepMode });
    post({ type: 'setThrottleMs', ms: +slider.value });
    post({ type: 'start' });
    startBtn.disabled = true;
    stopBtn.disabled = false;
    nextBtn.disabled = !stepMode;
    lrInput.disabled = true;
    updateNextButtonLabel();
  }
  function onNext(): void {
    // Phase < HiddenDelta → advance phase without unblocking the worker.
    // Phase == HiddenDelta → release the worker's step gate to roll a new sample.
    if (stepMode && currentPhase < Phase.HiddenDelta) {
      currentPhase = (currentPhase + 1) as Phase;
      applyCurrentPhase();
      return;
    }
    nextBtn.textContent = 'Loading next sample...';
    post({ type: 'stepNext' });
  }
}

// ─── Phase chip events ────────────────────────────────────────────────────
phaseChips.element.addEventListener('phase-jump', (e: any) => {
  if (!stepMode) return;
  currentPhase = e.detail as Phase;
  applyCurrentPhase();
});

// ─── Click sources for focused weight / focused hidden ────────────────────
w2View.element.addEventListener('cell-click', (e: any) => setFocusedWeight(e.detail));
perSampleNablaW2.element.addEventListener('cell-click', (e: any) => setFocusedWeight(e.detail));
diagram.element.addEventListener('weight-click', (e: any) => {
  // Only W[2] (last layer) maps to the Focus panel for now.
  const d = e.detail;
  if (d.layerIdx === 1) setFocusedWeight({ row: d.row, col: d.col });
});
hiddenActMatrix.element.addEventListener('cell-click', (e: any) => setFocusedHidden(e.detail.col));
diagram.element.addEventListener('hidden-click', (e: any) => setFocusedHidden(e.detail.index));

function setFocusedWeight(cell: { row: number; col: number }): void {
  focusedWeight = cell;
  w2View.setHighlight(cell);
  perSampleNablaW2.setHighlight(cell);
  // Side highlights on δ[2] row and a[1] col.
  deltaView.setHighlight(cell.row);
  errorView.setHighlight(cell.row);
  sigmoidPrimeView.setHighlight(cell.row);
  outputActView.setHighlight(cell.row);
  weightedInputView.setHighlight(cell.row);
  targetView.setHighlight(cell.row);
  hiddenActMatrix.setHighlight({ row: 0, col: cell.col });
  b2View.setHighlight(cell.row);
  nablaB2View.setHighlight(cell.row);
  focusPanel.render({ focused: cell, snapshot: latestSnapshot, learningRate, miniBatchSize });
  pushDiagramState({ highlightedWeight: { layerIdx: 1, row: cell.row, col: cell.col } });
}

function setFocusedHidden(index: number): void {
  focusedHidden = index;
  hiddenActMatrix.setHighlight({ row: 0, col: index });
  hiddenDeltaPanel.render({ focusedHidden: index, snapshot: latestSnapshot, phase: currentPhase });
  pushDiagramState({ focusedHiddenNeuron: index });
}

// ─── Worker message handler ───────────────────────────────────────────────

function handleWorkerMessage(ev: MessageEvent<WorkerToMain>): void {
  const msg = ev.data;
  const nodes = (window as any).__nodes;
  switch (msg.type) {
    case 'status':
      nodes.statusText.textContent = msg.text;
      nodes.statusText.style.color = '';
      break;
    case 'sample':
      latestSnapshot = msg.snapshot;
      lastTrueLabel = msg.snapshot.trueLabel;
      lastPredictedLabel = msg.snapshot.predictedLabel;
      miniBatchSize = msg.snapshot.miniBatchSize;
      // Mini-batch status line + label readouts.
      nodes.miniBatchText.textContent = stepMode
        ? 'Sample update: weights apply after this example'
        : 'Auto preview: showing periodic samples';
      nodes.miniBatchText.style.color = 'var(--gold)';
      nodes.trueLabelEl.textContent = `${msg.snapshot.trueLabel}`;
      nodes.predictedLabelEl.textContent = `${msg.snapshot.predictedLabel}`;
      nodes.predictedLabelEl.style.color =
        msg.snapshot.predictedLabel === msg.snapshot.trueLabel ? 'var(--green)' : 'var(--red)';
      drawSampleImage(msg.snapshot.pixels, msg.snapshot.inputSide);
      currentPhase = stepMode ? Phase.Forward : Phase.HiddenDelta;
      applyCurrentPhase();
      updateNextButtonLabel();
      break;
    case 'miniBatchApplied':
      // Re-render Focus panel — the W[2] / b[2] values just changed.
      if (latestSnapshot) {
        focusPanel.render({ focused: focusedWeight, snapshot: latestSnapshot, learningRate, miniBatchSize });
        hiddenDeltaPanel.render({ focusedHidden, snapshot: latestSnapshot, phase: currentPhase });
      }
      break;
    case 'epoch':
      const label = chartPointLabel(msg.snapshot.label ?? `epoch ${msg.snapshot.epoch + 1}`, msg.snapshot.testTotal);
      lossChart.push(label, msg.snapshot.meanTestLoss, msg.snapshot.meanTestLoss.toFixed(4));
      const acc = msg.snapshot.testTotal > 0 ? msg.snapshot.testCorrect / msg.snapshot.testTotal : 0;
      accuracyChart.push(label, acc, `${msg.snapshot.testCorrect}/${msg.snapshot.testTotal} (${(acc * 100).toFixed(2)}%)`);
      break;
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

// ─── Phase application — gate which views show data based on phase ────────

function applyCurrentPhase(): void {
  phaseChips.setPhase(currentPhase);
  updateNextButtonLabel();

  const snap = latestSnapshot;
  if (!snap) return;

  const revealForward     = currentPhase >= Phase.Forward;
  const revealLoss        = currentPhase >= Phase.Loss;
  const revealOutputDelta = currentPhase >= Phase.OutputDelta;
  const revealGradient    = currentPhase >= Phase.Gradient;
  const outputCostEl = (window as any).__nodes.outputCostEl as HTMLElement;
  const vectorVisibility = [revealLoss, revealForward, revealForward, revealOutputDelta, revealLoss, revealOutputDelta];
  outputVectorHeaders.forEach((header, i) => setDataVisible(header, vectorVisibility[i]));
  outputVectorViews.forEach((view, i) => setDataVisible(view.element, vectorVisibility[i]));
  setDataVisible(outputCostEl, revealLoss);
  setDataVisible(perSampleNablaW2.element, revealGradient);
  document.querySelectorAll<HTMLElement>('.gradient-formula-heading').forEach(el => setDataVisible(el, revealGradient));

  // W[2] / b[2] are network state — always shown.
  w2View.update(snap.weightsW2, OUTPUT_SIZE, HIDDEN_SIZE);
  b2View.update(snap.biasesB2);

  // Phase 1 (Forward).
  hiddenActMatrix.update(revealForward ? snap.hiddenActivation : null, 1, HIDDEN_SIZE);
  weightedInputView.update(revealForward ? snap.outputWeightedInput : null, OUTPUT_SIZE);
  outputActView.update(revealForward ? snap.outputActivation : null, OUTPUT_SIZE);

  // Phase 2 (Loss).
  targetView.update(revealLoss ? snap.target : null, OUTPUT_SIZE);
  errorView.update(revealLoss ? snap.error : null, OUTPUT_SIZE);
  if (revealLoss) {
    let c = 0;
    for (let k = 0; k < OUTPUT_SIZE; k++) {
      const d = snap.outputActivation[k] - snap.target[k];
      c += d * d;
    }
    c *= 0.5;
    outputCostEl.innerHTML = `$C=\\tfrac{1}{2}\\sum_k(a_{2,k}-y_k)^2=$ <span class="cost-result">$\\;${c.toFixed(4)}$</span>`;
    renderMath(outputCostEl);
  } else {
    outputCostEl.innerHTML = '$C=\\tfrac{1}{2}\\sum_k(a_{2,k}-y_k)^2$';
    renderMath(outputCostEl);
  }

  // Phase 3 (Output δ).
  sigmoidPrimeView.update(revealOutputDelta ? snap.outputSigmoidPrime : null, OUTPUT_SIZE);
  deltaView.update(revealOutputDelta ? snap.outputDelta : null, OUTPUT_SIZE);

  // Phase 4 (Gradient).
  perSampleNablaW2.update(revealGradient ? snap.perSampleNablaW2 : null, OUTPUT_SIZE, HIDDEN_SIZE);
  nablaB2View.update(revealGradient ? snap.accumNablaB2 : null);

  // Refresh panels and diagram.
  focusPanel.render({ focused: focusedWeight, snapshot: snap, learningRate, miniBatchSize });
  hiddenDeltaPanel.render({ focusedHidden, snapshot: snap, phase: currentPhase });
  pushDiagramState({});
}

function pushDiagramState(overrides: Partial<DiagramState>): void {
  if (!latestSnapshot) return;
  const snap = latestSnapshot;
  const state: DiagramState = {
    weights: [snap.weightsW1, snap.weightsW2],
    sizes: [snap.pixels.length, HIDDEN_SIZE, OUTPUT_SIZE],
    trueLabel: lastTrueLabel,
    predictedLabel: lastPredictedLabel,
    phase: currentPhase,
    hiddenActivation: snap.hiddenActivation,
    outputActivation: snap.outputActivation,
    hiddenDelta: snap.hiddenDelta,
    outputDelta: snap.outputDelta,
    focusedHiddenNeuron: focusedHidden,
    highlightedWeight: focusedWeight ? { layerIdx: 1, row: focusedWeight.row, col: focusedWeight.col } : null,
    ...overrides,
  };
  diagram.setState(state);
}

function updateNextButtonLabel(): void {
  const nodes = (window as any).__nodes;
  if (!nodes?.nextBtn) return;
  if (!stepMode) {
    nodes.nextBtn.textContent = 'Auto running';
    return;
  }
  nodes.nextBtn.textContent = currentPhase < Phase.HiddenDelta ? 'Next phase' : 'Next sample';
}

function setDataVisible(el: HTMLElement, visible: boolean): void {
  el.classList.toggle('data-hidden', !visible);
}

function resetLessonDisplay(): void {
  const nodes = (window as any).__nodes;
  latestSnapshot = null;
  lastTrueLabel = -1;
  lastPredictedLabel = -1;
  focusedWeight = null;
  focusedHidden = null;
  currentPhase = Phase.Forward;
  phaseChips.setPhase(currentPhase);
  updateNextButtonLabel();

  nodes.trueLabelEl.textContent = '—';
  nodes.predictedLabelEl.textContent = '—';
  nodes.predictedLabelEl.style.color = '';
  nodes.statusText.style.color = '';
  nodes.miniBatchText.textContent = 'Update not started';
  nodes.miniBatchText.style.color = '';

  outputVectorHeaders.forEach(header => setDataVisible(header, false));
  outputVectorViews.forEach(view => {
    setDataVisible(view.element, false);
    view.update(null, OUTPUT_SIZE);
    view.setHighlight(null);
  });

  const outputCostEl = nodes.outputCostEl as HTMLElement;
  outputCostEl.innerHTML = '$C=\\tfrac{1}{2}\\sum_k(a_{2,k}-y_k)^2$';
  renderMath(outputCostEl);
  setDataVisible(outputCostEl, false);
  document.querySelectorAll<HTMLElement>('.gradient-formula-heading').forEach(el => setDataVisible(el, false));

  perSampleNablaW2.update(null, OUTPUT_SIZE, HIDDEN_SIZE);
  perSampleNablaW2.setHighlight(null);
  setDataVisible(perSampleNablaW2.element, false);
  w2View.update(null, OUTPUT_SIZE, HIDDEN_SIZE);
  w2View.setHighlight(null);
  b2View.update(null, OUTPUT_SIZE);
  b2View.setHighlight(null);
  nablaB2View.update(null, OUTPUT_SIZE);
  nablaB2View.setHighlight(null);
  hiddenActMatrix.update(null, 1, HIDDEN_SIZE);
  hiddenActMatrix.setHighlight(null);

  clearSampleImage();
  focusPanel.render({ focused: null, snapshot: null, learningRate, miniBatchSize });
  hiddenDeltaPanel.render({ focusedHidden: null, snapshot: null, phase: currentPhase });
  diagram.clear();
}

function chartPointLabel(label: string, testTotal: number): string {
  return testTotal < 10000 ? `${label} (quick ${testTotal})` : `${label} (full test set)`;
}

// ─── Sample image rendering (INPUT_SIDE×INPUT_SIDE → 280×280 NearestNeighbor) ──

function clearSampleImage(): void {
  const ctx = sampleCanvas.getContext('2d')!;
  ctx.clearRect(0, 0, sampleCanvas.width, sampleCanvas.height);
}

function drawSampleImage(pixels: Float64Array, side: number): void {
  if (sampleCanvas.width !== side || sampleCanvas.height !== side) {
    sampleCanvas.width = side;
    sampleCanvas.height = side;
  }
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

// ─── Small DOM helpers ────────────────────────────────────────────────────
function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}
