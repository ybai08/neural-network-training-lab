// Convolutional teaching network. Hand-written forward + backprop — no TF.js,
// no autograd, no library. The whole network has ~1600 parameters; everything
// runs in single-digit microseconds per sample.
//
// Architecture:
//   Input         14×14×1
//   Conv1 (3×3, 4 filters, valid)    → 12×12×4
//   ReLU                              → 12×12×4
//   MaxPool (2×2, stride 2)           → 6×6×4
//   Conv2 (3×3, 8 filters, valid)     → 4×4×8
//   ReLU                              → 4×4×8
//   Flatten                           → 128
//   FC (128 → 10)                     → 10
//   Sigmoid                           → 10
//   Loss: ½‖a − y‖²  (same as the FC teaching net, so the cost story carries over)
//
// Storage conventions:
//   • 3D activations are channel-major: tensor[c · H · W + y · W + x]
//   • 4D kernels are output-channel-major: K[(oc · IC + ic) · KH + ky) · KW + kx]
//
// All arrays are Float64Array so they survive postMessage() into the main
// thread without copying classes.

import { mulberry32 } from './matrix';

// ─── Architecture constants — also exported so the UI can label things ────

export const INPUT_SIDE = 14;
export const KERNEL = 3;
export const CONV1_FILTERS = 4;
export const CONV2_FILTERS = 8;
export const CONV1_OUT = INPUT_SIDE - KERNEL + 1;     // 12
export const POOL1_OUT = CONV1_OUT / 2;               // 6
export const CONV2_OUT = POOL1_OUT - KERNEL + 1;      // 4
export const FC_INPUT = CONV2_OUT * CONV2_OUT * CONV2_FILTERS; // 128
export const OUTPUTS = 10;

// ─── Forward trace exposes every intermediate the UI needs to render ──────

export interface ForwardTrace {
  conv1Pre: Float64Array;    // 12×12×4 — Conv1 output BEFORE ReLU
  conv1Relu: Float64Array;   // 12×12×4 — after ReLU
  pool1Out: Float64Array;    // 6×6×4
  pool1Idx: Uint8Array;      // 6×6×4 — argmax indices for backward pass
  conv2Pre: Float64Array;    // 4×4×8
  conv2Relu: Float64Array;   // 4×4×8 (== flatten contents)
  fcOut: Float64Array;       // 10 — z = W·flatten + b  (pre-sigmoid)
  output: Float64Array;      // 10 — σ(z)
}

export interface GradientTrace {
  dConv1K: Float64Array;
  dConv1B: Float64Array;
  dConv2K: Float64Array;
  dConv2B: Float64Array;
  dFcW: Float64Array;
  dFcB: Float64Array;
}

export class ConvNet {
  // Parameters (live network state). All Float64Array for cheap postMessage.
  conv1Kernels: Float64Array;  // 4 × 1 × 3 × 3 = 36
  conv1Biases: Float64Array;   // 4
  conv2Kernels: Float64Array;  // 8 × 4 × 3 × 3 = 288
  conv2Biases: Float64Array;   // 8
  fcWeights: Float64Array;     // 10 × 128 = 1280
  fcBiases: Float64Array;      // 10

  // Mini-batch state for the teaching UI.
  miniBatchSampleIndex = 0;
  currentMiniBatchSize = 0;
  // Accumulated gradients across the current mini-batch — null in the gap
  // between batches so the UI can show "accumulator reset".
  gConv1K: Float64Array | null = null;
  gConv1B: Float64Array | null = null;
  gConv2K: Float64Array | null = null;
  gConv2B: Float64Array | null = null;
  gFcW: Float64Array | null = null;
  gFcB: Float64Array | null = null;

  lastTrace: ForwardTrace | null = null;

  constructor(seed: number) {
    const rng = mulberry32(seed);
    // He init for ReLU layers: weights ~ N(0, sqrt(2/fanIn)).
    this.conv1Kernels = randomNormal(CONV1_FILTERS * 1 * KERNEL * KERNEL, rng, Math.sqrt(2 / (1 * KERNEL * KERNEL)));
    this.conv1Biases = new Float64Array(CONV1_FILTERS);
    this.conv2Kernels = randomNormal(CONV2_FILTERS * CONV1_FILTERS * KERNEL * KERNEL, rng, Math.sqrt(2 / (CONV1_FILTERS * KERNEL * KERNEL)));
    this.conv2Biases = new Float64Array(CONV2_FILTERS);
    // FC layer ends in sigmoid — scaled-Xavier-ish init (σ keeps gradients smaller).
    this.fcWeights = randomNormal(OUTPUTS * FC_INPUT, rng, 1 / Math.sqrt(FC_INPUT));
    this.fcBiases = new Float64Array(OUTPUTS);
  }

  /** Forward pass. Returns every intermediate tensor; cheap to throw away. */
  forward(input: Float64Array): ForwardTrace {
    const conv1Pre = conv2dForward(input, INPUT_SIDE, INPUT_SIDE, 1, this.conv1Kernels, this.conv1Biases, CONV1_FILTERS, KERNEL);
    const conv1Relu = reluForward(conv1Pre);
    const pool1Out = new Float64Array(POOL1_OUT * POOL1_OUT * CONV1_FILTERS);
    const pool1Idx = new Uint8Array(POOL1_OUT * POOL1_OUT * CONV1_FILTERS);
    maxPool2dForward(conv1Relu, CONV1_OUT, CONV1_OUT, CONV1_FILTERS, pool1Out, pool1Idx);
    const conv2Pre = conv2dForward(pool1Out, POOL1_OUT, POOL1_OUT, CONV1_FILTERS, this.conv2Kernels, this.conv2Biases, CONV2_FILTERS, KERNEL);
    const conv2Relu = reluForward(conv2Pre);

    // FC: 128 → 10  (flatten reuses conv2Relu's storage; we just walk it linearly).
    const fcOut = new Float64Array(OUTPUTS);
    for (let o = 0; o < OUTPUTS; o++) {
      let sum = this.fcBiases[o];
      const wRow = o * FC_INPUT;
      for (let i = 0; i < FC_INPUT; i++) sum += this.fcWeights[wRow + i] * conv2Relu[i];
      fcOut[o] = sum;
    }
    const output = new Float64Array(OUTPUTS);
    for (let o = 0; o < OUTPUTS; o++) output[o] = 1 / (1 + Math.exp(-fcOut[o]));

    const trace = { conv1Pre, conv1Relu, pool1Out, pool1Idx, conv2Pre, conv2Relu, fcOut, output };
    this.lastTrace = trace;
    return trace;
  }

  /**
   * Backward pass — sigmoid + ½‖a−y‖² at the output, then chain rule back
   * through FC → flatten → ReLU → Conv2 → MaxPool → ReLU → Conv1.
   * Returns per-parameter gradients for this single sample.
   */
  backward(input: Float64Array, trace: ForwardTrace, target: Float64Array): GradientTrace {
    // Output δ: sigmoid + MSE gives  δ = (a − y) · σ'(z) = (a − y) · a · (1 − a).
    const dFcOut = new Float64Array(OUTPUTS);
    for (let o = 0; o < OUTPUTS; o++) {
      const a = trace.output[o];
      dFcOut[o] = (a - target[o]) * a * (1 - a);
    }

    // FC backward.
    const dFcW = new Float64Array(this.fcWeights.length);
    const dFcB = new Float64Array(OUTPUTS);
    const dFlatten = new Float64Array(FC_INPUT);
    for (let o = 0; o < OUTPUTS; o++) {
      const g = dFcOut[o];
      dFcB[o] = g;
      const wRow = o * FC_INPUT;
      for (let i = 0; i < FC_INPUT; i++) {
        dFcW[wRow + i] = g * trace.conv2Relu[i];
        dFlatten[i] += this.fcWeights[wRow + i] * g;
      }
    }

    // dFlatten has the same shape as conv2Relu — they share layout, so reuse it.
    const dConv2Pre = reluBackward(trace.conv2Pre, dFlatten);

    // Conv2 backward.
    const dConv2K = new Float64Array(this.conv2Kernels.length);
    const dConv2B = new Float64Array(CONV2_FILTERS);
    const dPool1Out = new Float64Array(trace.pool1Out.length);
    conv2dBackward(
      trace.pool1Out, POOL1_OUT, POOL1_OUT, CONV1_FILTERS,
      this.conv2Kernels, CONV2_FILTERS, KERNEL,
      dConv2Pre, dConv2K, dConv2B, dPool1Out,
    );

    // MaxPool backward — route each grad back to the argmax-cell only.
    const dConv1Relu = new Float64Array(trace.conv1Relu.length);
    maxPool2dBackward(dPool1Out, trace.pool1Idx, POOL1_OUT, POOL1_OUT, CONV1_FILTERS, dConv1Relu);

    const dConv1Pre = reluBackward(trace.conv1Pre, dConv1Relu);

    // Conv1 backward — we don't need ∂L/∂input (no layer below), so pass null.
    const dConv1K = new Float64Array(this.conv1Kernels.length);
    const dConv1B = new Float64Array(CONV1_FILTERS);
    conv2dBackward(
      input, INPUT_SIDE, INPUT_SIDE, 1,
      this.conv1Kernels, CONV1_FILTERS, KERNEL,
      dConv1Pre, dConv1K, dConv1B, null,
    );

    return { dConv1K, dConv1B, dConv2K, dConv2B, dFcW, dFcB };
  }

  /**
   * One mini-batch of SGD. Fires onSample after each sample so the teaching
   * UI can step through (and so step-mode can gate). Identical control flow
   * to the FC Network.updateMiniBatch.
   */
  async updateMiniBatch(
    batch: ReadonlyArray<{ x: Float64Array; y: Float64Array }>,
    learningRate: number,
    onSample: (predictedLabel: number) => Promise<void> | void,
  ): Promise<void> {
    this.currentMiniBatchSize = batch.length;
    this.gConv1K = new Float64Array(this.conv1Kernels.length);
    this.gConv1B = new Float64Array(this.conv1Biases.length);
    this.gConv2K = new Float64Array(this.conv2Kernels.length);
    this.gConv2B = new Float64Array(this.conv2Biases.length);
    this.gFcW = new Float64Array(this.fcWeights.length);
    this.gFcB = new Float64Array(this.fcBiases.length);

    for (let k = 0; k < batch.length; k++) {
      this.miniBatchSampleIndex = k;
      const { x, y } = batch[k];
      const trace = this.forward(x);
      const grad = this.backward(x, trace, y);

      addInPlace(this.gConv1K, grad.dConv1K);
      addInPlace(this.gConv1B, grad.dConv1B);
      addInPlace(this.gConv2K, grad.dConv2K);
      addInPlace(this.gConv2B, grad.dConv2B);
      addInPlace(this.gFcW, grad.dFcW);
      addInPlace(this.gFcB, grad.dFcB);

      await onSample(argMax(trace.output));
    }

    const scale = learningRate / batch.length;
    axpyInPlace(this.conv1Kernels, scale, this.gConv1K);
    axpyInPlace(this.conv1Biases, scale, this.gConv1B);
    axpyInPlace(this.conv2Kernels, scale, this.gConv2K);
    axpyInPlace(this.conv2Biases, scale, this.gConv2B);
    axpyInPlace(this.fcWeights, scale, this.gFcW);
    axpyInPlace(this.fcBiases, scale, this.gFcB);

    this.gConv1K = this.gConv1B = this.gConv2K = this.gConv2B = this.gFcW = this.gFcB = null;
  }

  /** Argmax accuracy on a test set. */
  evaluate(test: ReadonlyArray<{ x: Float64Array; label: number }>): number {
    let correct = 0;
    for (const { x, label } of test) {
      const t = this.forward(x);
      if (argMax(t.output) === label) correct++;
    }
    return correct;
  }

  /** Mean ½‖a − y‖² over the test set with y one-hot. */
  meanLoss(test: ReadonlyArray<{ x: Float64Array; label: number }>): number {
    let total = 0;
    for (const { x, label } of test) {
      const t = this.forward(x);
      let s = 0;
      for (let k = 0; k < OUTPUTS; k++) {
        const d = t.output[k] - (k === label ? 1 : 0);
        s += d * d;
      }
      total += 0.5 * s;
    }
    return total / test.length;
  }
}

// ─── Layer primitives ─────────────────────────────────────────────────────

function conv2dForward(
  input: Float64Array,
  inH: number, inW: number, inC: number,
  kernels: Float64Array,
  biases: Float64Array,
  outC: number, kSize: number,
): Float64Array {
  const outH = inH - kSize + 1;
  const outW = inW - kSize + 1;
  const output = new Float64Array(outC * outH * outW);
  for (let oc = 0; oc < outC; oc++) {
    const ocK = oc * inC * kSize * kSize;
    const ocOut = oc * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = biases[oc];
        for (let ic = 0; ic < inC; ic++) {
          const icK = ocK + ic * kSize * kSize;
          const icIn = ic * inH * inW;
          for (let ky = 0; ky < kSize; ky++) {
            const inRow = icIn + (oy + ky) * inW;
            const kRow = icK + ky * kSize;
            for (let kx = 0; kx < kSize; kx++) {
              sum += input[inRow + ox + kx] * kernels[kRow + kx];
            }
          }
        }
        output[ocOut + oy * outW + ox] = sum;
      }
    }
  }
  return output;
}

function conv2dBackward(
  input: Float64Array,
  inH: number, inW: number, inC: number,
  kernels: Float64Array,
  outC: number, kSize: number,
  dOutput: Float64Array,
  dKernels: Float64Array,
  dBiases: Float64Array,
  dInput: Float64Array | null,
): void {
  const outH = inH - kSize + 1;
  const outW = inW - kSize + 1;
  for (let oc = 0; oc < outC; oc++) {
    const ocK = oc * inC * kSize * kSize;
    const ocOut = oc * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const g = dOutput[ocOut + oy * outW + ox];
        dBiases[oc] += g;
        for (let ic = 0; ic < inC; ic++) {
          const icK = ocK + ic * kSize * kSize;
          const icIn = ic * inH * inW;
          for (let ky = 0; ky < kSize; ky++) {
            const inRow = icIn + (oy + ky) * inW;
            const kRow = icK + ky * kSize;
            for (let kx = 0; kx < kSize; kx++) {
              dKernels[kRow + kx] += input[inRow + ox + kx] * g;
              if (dInput) dInput[inRow + ox + kx] += kernels[kRow + kx] * g;
            }
          }
        }
      }
    }
  }
}

function maxPool2dForward(
  input: Float64Array, inH: number, inW: number, inC: number,
  output: Float64Array, indices: Uint8Array,
): void {
  const outH = inH / 2;
  const outW = inW / 2;
  for (let c = 0; c < inC; c++) {
    const inStride = c * inH * inW;
    const outStride = c * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let max = -Infinity;
        let argIdx = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const v = input[inStride + (2 * oy + dy) * inW + (2 * ox + dx)];
            if (v > max) { max = v; argIdx = dy * 2 + dx; }
          }
        }
        const idx = outStride + oy * outW + ox;
        output[idx] = max;
        indices[idx] = argIdx;
      }
    }
  }
}

function maxPool2dBackward(
  dOutput: Float64Array, indices: Uint8Array,
  outH: number, outW: number, inC: number,
  dInput: Float64Array,
): void {
  const inH = outH * 2;
  const inW = outW * 2;
  for (let c = 0; c < inC; c++) {
    const inStride = c * inH * inW;
    const outStride = c * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const idx = indices[outStride + oy * outW + ox];
        const dy = idx >> 1;
        const dx = idx & 1;
        dInput[inStride + (2 * oy + dy) * inW + (2 * ox + dx)] = dOutput[outStride + oy * outW + ox];
      }
    }
  }
}

function reluForward(input: Float64Array): Float64Array {
  const out = new Float64Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = input[i] > 0 ? input[i] : 0;
  return out;
}

function reluBackward(preActivation: Float64Array, dOutput: Float64Array): Float64Array {
  const out = new Float64Array(preActivation.length);
  for (let i = 0; i < preActivation.length; i++) out[i] = preActivation[i] > 0 ? dOutput[i] : 0;
  return out;
}

// ─── Small numeric helpers ────────────────────────────────────────────────

function addInPlace(a: Float64Array, b: Float64Array): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}
function axpyInPlace(a: Float64Array, scale: number, b: Float64Array): void {
  for (let i = 0; i < a.length; i++) a[i] -= scale * b[i];
}
function argMax(v: Float64Array): number {
  let best = 0, bestVal = v[0];
  for (let i = 1; i < v.length; i++) if (v[i] > bestVal) { bestVal = v[i]; best = i; }
  return best;
}
function randomNormal(n: number, rng: () => number, stddev: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = standardNormal(rng) * stddev;
  return out;
}
function standardNormal(rng: () => number): number {
  let u1 = rng();
  while (u1 <= 0) u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
