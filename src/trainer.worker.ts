/// <reference lib="webworker" />
// Training worker. Owns the Network instance and the MNIST data; runs the SGD
// loop and posts a SampleSnapshot back to the main thread after every sample.
//
// Step-mode synchronisation: when the user has step mode on, the worker awaits
// `nextStepGate` after each sample. The main thread releases it by posting
// `{type:'stepNext'}`. This replaces the C# `ManualResetEventSlim` blocking
// pattern with a Promise-resolution one — same semantics, web-friendly.

import { INPUT_SIDE, INPUT_PIXELS, loadMnist, type MnistData } from './mnist';
import { Network } from './network';
import { Vector } from './matrix';
import type { MainToWorker, WorkerToMain, SampleSnapshot } from './types';

let net: Network | null = null;
let mnist: MnistData | null = null;
let learningRate = 3.0;
let miniBatchSize = 10;
let epochs = 30;
let seed = 42;

let stepMode = true;
let throttleMs = 0;
let stopRequested = false;
let started = false;
const QUICK_TEST_COUNT = 500;
const MIN_AUTO_PREVIEW_MS = 100;
const AUTO_YIELD_EVERY = 200;
let lastAutoPreviewAt = 0;

// Step-mode gate. When step mode is on, after each sample we await this promise;
// it resolves when the main thread posts {type:'stepNext'} (or step mode flips off).
let stepResolve: (() => void) | null = null;

const ctx: DedicatedWorkerGlobalScope = self as any;

ctx.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'init':
      learningRate = msg.learningRate;
      miniBatchSize = msg.miniBatchSize;
      epochs = msg.epochs;
      seed = msg.seed;
      stepMode = msg.stepMode;
      lastAutoPreviewAt = 0;
      // Defer the actual work until 'start'; init just stores parameters.
      break;
    case 'start':
      if (started) break;
      started = true;
      void run();
      break;
    case 'stop':
      stopRequested = true;
      releaseStepGate();   // let any in-flight await unblock so the loop can observe stopRequested
      break;
    case 'stepNext':
      releaseStepGate();
      break;
    case 'setStepMode':
      stepMode = msg.on;
      lastAutoPreviewAt = 0;
      // Flipping OFF in mid-flight releases the gate so training resumes free-running.
      if (!stepMode) releaseStepGate();
      break;
    case 'setThrottleMs':
      throttleMs = Math.max(0, msg.ms | 0);
      break;
  }
};

function post(msg: WorkerToMain): void { ctx.postMessage(msg); }

function releaseStepGate(): void {
  if (stepResolve) {
    const r = stepResolve;
    stepResolve = null;
    r();
  }
}

/** Wait either for {type:'stepNext'} (step mode) or for `throttleMs` (auto). */
function waitForStep(): Promise<void> {
  if (stepMode) {
    return new Promise<void>(resolve => { stepResolve = resolve; });
  }
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

    const inputCount = INPUT_PIXELS;
    net = new Network([inputCount, 30, 10], seed);

    // Pre-build training pairs as {x: Vector, y: one-hot Vector}. Test data
    // keeps integer labels for accuracy / loss evaluation.
    const train = mnist.train.images.map((img, i) => ({
      x: new Vector(img),
      y: oneHot(mnist!.train.labels[i], 10),
    }));
    const test = mnist.test.images.map((img, i) => ({
      x: new Vector(img),
      label: mnist!.test.labels[i],
    }));
    const quickTest = test.slice(0, Math.min(QUICK_TEST_COUNT, test.length));

    post({
      type: 'status',
      text: miniBatchSize === 1
        ? `Training: ${train.length} samples, ${epochs} epochs, update after every sample, η=${learningRate}`
        : `Training: ${train.length} samples, ${epochs} epochs, mini-batch ${miniBatchSize}, η=${learningRate}`,
    });

    const indices = new Int32Array(train.length);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
    const rngShuffle = mulberry32(seed ^ 0xA5A5A5A5);
    let samplesSeen = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (stopRequested) break;

      // Fisher-Yates shuffle, same seed lineage every epoch (deterministic enough).
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rngShuffle() * (i + 1));
        const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      for (let start = 0; start < train.length && !stopRequested; start += miniBatchSize) {
        const end = Math.min(start + miniBatchSize, train.length);
        const batch: { x: Vector; y: Vector }[] = [];
        for (let k = start; k < end; k++) batch.push(train[indices[k]]);

        let sentSampleForBatch = false;
        await net.updateMiniBatch(batch, learningRate, async (predicted) => {
          if (!net || !net.lastOutputTrace || !net.lastHiddenTrace) return;
          const sampleNo = samplesSeen + net.miniBatchSampleIndex + 1;
          if (shouldPostSamplePreview(sampleNo)) {
            const snap = buildSnapshot(net, predicted, train[indices[start + net.miniBatchSampleIndex]]);
            post({ type: 'sample', snapshot: snap });
            sentSampleForBatch = true;
            await waitForStep();
          } else if (!stepMode && sampleNo % AUTO_YIELD_EVERY === 0) {
            await yieldToEventLoop();
          }
        });
        if (stopRequested) break;
        if (stepMode || sentSampleForBatch) post({ type: 'miniBatchApplied' });
        samplesSeen += end - start;
        if (shouldPostQuickMetric(samplesSeen)) {
          postEvaluation(net, quickTest, epoch + samplesSeen / train.length, `update ${samplesSeen.toLocaleString()}`);
        }
      }

      if (stopRequested) break;
      post({ type: 'status', text: `Evaluating epoch ${epoch + 1}/${epochs}…` });
      const correct = net.evaluate(test);
      const meanLoss = net.meanLoss(test, 10);
      post({
        type: 'epoch',
        snapshot: { epoch, meanTestLoss: meanLoss, testCorrect: correct, testTotal: test.length, label: `epoch ${epoch + 1}` },
      });
    }

    post({ type: 'done' });
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}

/** Assemble the structured-cloneable snapshot for one sample. */
function buildSnapshot(n: Network, predictedLabel: number, sample: { x: Vector; y: Vector }): SampleSnapshot {
  const o = n.lastOutputTrace!;
  const h = n.lastHiddenTrace!;
  return {
    pixels: sample.x.toFlat(),
    inputSide: INPUT_SIDE,

    trueLabel: sample.y.argMax(),
    predictedLabel,

    miniBatchSampleIndex: n.miniBatchSampleIndex,
    miniBatchSize: n.currentMiniBatchSize,

    hiddenWeightedInput: h.weightedInput.toFlat(),
    hiddenActivation: o.hiddenActivation.toFlat(),
    outputWeightedInput: o.weightedInput.toFlat(),
    outputActivation: o.outputActivation.toFlat(),

    target: o.target.toFlat(),
    error: o.error.toFlat(),

    outputSigmoidPrime: o.sigmoidPrime.toFlat(),
    outputDelta: o.delta.toFlat(),

    perSampleNablaW2: o.perSampleNablaW.toFlat(),

    hiddenBackpropPreSigmoid: h.backpropPreSigmoid.toFlat(),
    hiddenSigmoidPrime: h.sigmoidPrime.toFlat(),
    hiddenDelta: h.delta.toFlat(),

    weightsW1: n.weights[0].toFlat(),
    weightsW2: n.weights[1].toFlat(),
    biasesB1: n.biases[0].toFlat(),
    biasesB2: n.biases[1].toFlat(),

    accumNablaW2: n.currentNablaW ? n.currentNablaW[1].toFlat() : null,
    accumNablaB2: n.currentNablaB ? n.currentNablaB[1].toFlat() : null,
  };
}

function oneHot(label: number, classes: number): Vector {
  const v = Vector.zeros(classes);
  v.data[label] = 1;
  return v;
}

function shouldPostQuickMetric(samplesSeen: number): boolean {
  if (samplesSeen <= 500) return samplesSeen % 50 === 0;
  return samplesSeen % 1000 === 0;
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
  n: Network,
  testData: ReadonlyArray<{ x: Vector; label: number }>,
  epoch: number,
  label: string,
): void {
  let correct = 0;
  let totalLoss = 0;
  for (const { x, label: targetLabel } of testData) {
    const a = n.feedForward(x);
    if (a.argMax() === targetLabel) correct++;
    let s = 0;
    for (let k = 0; k < 10; k++) {
      const diff = a.data[k] - (k === targetLabel ? 1 : 0);
      s += diff * diff;
    }
    totalLoss += 0.5 * s;
  }
  post({
    type: 'epoch',
    snapshot: {
      epoch,
      meanTestLoss: totalLoss / testData.length,
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
