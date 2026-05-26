/// <reference lib="webworker" />
// CNN training worker. Mirrors trainer.worker.ts in shape — same MNIST loader,
// same step-mode Promise gate, same message protocol — but owns a ConvNet
// instead of an FC Network and posts ConvNetSnapshot.
//
// Keeping the workers separate avoids conditional logic at the cost of ~30
// lines of duplication; the two pages are independent in every other way.

import { loadMnist, type MnistData } from './mnist';
import { ConvNet, INPUT_SIDE } from './convnet';
import type { MainToWorker, ConvWorkerToMain, ConvNetSnapshot } from './types';

let net: ConvNet | null = null;
let mnist: MnistData | null = null;
let learningRate = 0.1;          // ReLU + smaller params → keep η modest
let miniBatchSize = 10;
let epochs = 30;
let seed = 42;

let stepMode = true;
let throttleMs = 0;
let stopRequested = false;
let started = false;
let stepResolve: (() => void) | null = null;
const QUICK_TEST_COUNT = 200;
const MIN_AUTO_PREVIEW_MS = 200;
const AUTO_YIELD_EVERY = 200;
let lastAutoPreviewAt = 0;

const ctx: DedicatedWorkerGlobalScope = self as any;

ctx.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'init':
      stopRequested = false;
      started = false;
      releaseStepGate();
      learningRate = msg.learningRate;
      miniBatchSize = msg.miniBatchSize;
      epochs = msg.epochs;
      seed = msg.seed;
      stepMode = msg.stepMode;
      lastAutoPreviewAt = 0;
      break;
    case 'start':
      if (started) break;
      started = true;
      void run();
      break;
    case 'stop':
      stopRequested = true;
      releaseStepGate();
      break;
    case 'stepNext':
      releaseStepGate();
      break;
    case 'setStepMode':
      stepMode = msg.on;
      if (!stepMode) releaseStepGate();
      break;
    case 'setThrottleMs':
      throttleMs = Math.max(0, msg.ms | 0);
      break;
  }
};

function post(msg: ConvWorkerToMain): void { ctx.postMessage(msg); }
function releaseStepGate(): void {
  if (stepResolve) { const r = stepResolve; stepResolve = null; r(); }
}
function waitForStep(): Promise<void> {
  if (stepMode) return new Promise<void>(resolve => { stepResolve = resolve; });
  if (throttleMs > 0) return new Promise(resolve => setTimeout(resolve, throttleMs));
  return yieldToEventLoop();
}
function yieldToEventLoop(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

async function run(): Promise<void> {
  try {
    if (!mnist) {
      post({ type: 'status', text: 'Loading MNIST…' });
      mnist = await loadMnist(text => post({ type: 'status', text }));
    }
    net = new ConvNet(seed);

    // Pre-build training pairs: pixels are already Float64Array post-downsample.
    // y is a one-hot Float64Array of length 10.
    const train = mnist.train.images.map((img, i) => ({
      x: img,
      y: oneHot(mnist!.train.labels[i], 10),
    }));
    const test = mnist.test.images.map((img, i) => ({
      x: img,
      label: mnist!.test.labels[i],
    }));
    const quickTest = test.slice(0, Math.min(QUICK_TEST_COUNT, test.length));

    post({ type: 'status', text: `Training CNN: ${train.length} samples, ${epochs} epochs, mini-batch ${miniBatchSize}, η=${learningRate}` });

    const indices = new Int32Array(train.length);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
    const rngShuffle = mulberry32(seed ^ 0xA5A5A5A5);
    let samplesSeen = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (stopRequested) break;

      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rngShuffle() * (i + 1));
        const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      for (let start = 0; start < train.length && !stopRequested; start += miniBatchSize) {
        const end = Math.min(start + miniBatchSize, train.length);
        const batch: { x: Float64Array; y: Float64Array }[] = [];
        for (let k = start; k < end; k++) batch.push(train[indices[k]]);

        await net.updateMiniBatch(batch, learningRate, async (predicted) => {
          if (!net || !net.lastTrace) return;
          const sampleNo = samplesSeen + net.miniBatchSampleIndex + 1;
          if (shouldPostSamplePreview(sampleNo)) {
            const snap = buildSnapshot(net, predicted, train[indices[start + net.miniBatchSampleIndex]]);
            post({ type: 'sample', snapshot: snap });
            await waitForStep();
          } else if (!stepMode && sampleNo % AUTO_YIELD_EVERY === 0) {
            await yieldToEventLoop();
          }
        });
        if (stopRequested) break;
        // In auto mode the full CNN render is fairly heavy. The next preview
        // will carry fresh weights, so avoid a second immediate repaint here.
        if (stepMode) post({ type: 'miniBatchApplied' });
        samplesSeen += end - start;
        if (shouldPostQuickMetric(samplesSeen)) {
          postEvaluation(net, quickTest, epoch + samplesSeen / train.length, `update ${samplesSeen.toLocaleString()}`);
        }
      }

      if (stopRequested) break;
      post({ type: 'status', text: `Evaluating epoch ${epoch + 1}/${epochs}…` });
      const { correct, meanLoss } = net.evaluateWithLoss(test);
      post({ type: 'epoch', snapshot: { epoch, meanTestLoss: meanLoss, testCorrect: correct, testTotal: test.length, label: `epoch ${epoch + 1}` } });
    }

    post({ type: 'done' });
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}

function buildSnapshot(n: ConvNet, predictedLabel: number, sample: { x: Float64Array; y: Float64Array }): ConvNetSnapshot {
  const t = n.lastTrace!;
  return {
    pixels: new Float64Array(sample.x),
    inputSide: INPUT_SIDE,
    trueLabel: argMax(sample.y),
    predictedLabel,
    miniBatchSampleIndex: n.miniBatchSampleIndex,
    miniBatchSize: n.currentMiniBatchSize,
    conv1Pre: new Float64Array(t.conv1Pre),
    conv1Relu: new Float64Array(t.conv1Relu),
    pool1Out: new Float64Array(t.pool1Out),
    conv2Pre: new Float64Array(t.conv2Pre),
    conv2Relu: new Float64Array(t.conv2Relu),
    fcOut: new Float64Array(t.fcOut),
    output: new Float64Array(t.output),
    target: new Float64Array(sample.y),
    conv1Kernels: new Float64Array(n.conv1Kernels),
    conv1Biases: new Float64Array(n.conv1Biases),
    conv2Kernels: new Float64Array(n.conv2Kernels),
    conv2Biases: new Float64Array(n.conv2Biases),
    fcWeights: new Float64Array(n.fcWeights),
    fcBiases: new Float64Array(n.fcBiases),
  };
}

function oneHot(label: number, classes: number): Float64Array {
  const v = new Float64Array(classes);
  v[label] = 1;
  return v;
}
function argMax(v: Float64Array): number {
  let best = 0, bestVal = v[0];
  for (let i = 1; i < v.length; i++) if (v[i] > bestVal) { bestVal = v[i]; best = i; }
  return best;
}
function shouldPostQuickMetric(samplesSeen: number): boolean {
  if (samplesSeen <= 500) return samplesSeen % 50 === 0;
  if (samplesSeen <= 5000) return samplesSeen % 500 === 0;
  return samplesSeen % 5000 === 0;
}
function shouldPostSamplePreview(sampleNo: number): boolean {
  if (stepMode) return true;
  if (sampleNo === 1) {
    lastAutoPreviewAt = performance.now();
    return true;
  }
  const now = performance.now();
  const previewInterval = Math.max(MIN_AUTO_PREVIEW_MS, throttleMs);
  if (now - lastAutoPreviewAt < previewInterval) return false;
  lastAutoPreviewAt = now;
  return true;
}
function postEvaluation(
  n: ConvNet,
  testData: ReadonlyArray<{ x: Float64Array; label: number }>,
  epoch: number,
  label: string,
): void {
  const { correct, meanLoss } = n.evaluateWithLoss(testData);
  post({
    type: 'epoch',
    snapshot: {
      epoch,
      meanTestLoss: meanLoss,
      testCorrect: correct,
      testTotal: testData.length,
      label,
    },
  });
}
function mulberry32(s: number): () => number {
  let a = s >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
