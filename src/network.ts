// Port of Michael Nielsen's network.py (and the C# port that this is replacing):
// fully-connected feed-forward sigmoid net trained with mini-batch SGD and
// vanilla backprop. Hand-translated rather than using TF.js because the
// pedagogical goal is to expose every intermediate (z, a, δ) cleanly — the
// arithmetic is twenty lines, no library needed.

import { Matrix, Vector, mulberry32 } from './matrix';

/**
 * Backprop trace for the output layer on one sample. Identical fields to the
 * C# OutputBackpropSnapshot record — kept around because the UI references
 * each of them by name when rendering the chain rule.
 */
export interface OutputTrace {
  hiddenActivation: Vector;    // a[L-1] feeding into the output layer
  outputActivation: Vector;    // a[L] (what the net said)
  weightedInput: Vector;       // z[L]
  sigmoidPrime: Vector;        // σ'(z[L])
  target: Vector;              // y (one-hot)
  error: Vector;               // a[L] − y
  delta: Vector;               // δ[L] = error ⊙ σ'(z[L])
  perSampleNablaW: Matrix;     // δ[L] · (a[L-1])ᵀ
}

/**
 * Backprop trace for the (single) hidden layer — what the C# version called
 * HiddenBackpropSnapshot. Useful for the "responsibility flow backward" panel.
 */
export interface HiddenTrace {
  input: Vector;                  // a[0] = x
  weightedInput: Vector;          // z[1]
  sigmoidPrime: Vector;           // σ'(z[1])
  backpropPreSigmoid: Vector;     // W[2]ᵀ · δ[2] — aggregated blame before ⊙ σ'
  delta: Vector;                  // δ[1]
}

export class Network {
  readonly sizes: readonly number[];
  /** Biases for layers 1..L. `biases[i]` has length sizes[i+1]. */
  readonly biases: Vector[];
  /** Weights between consecutive layers. `weights[i]` has shape sizes[i+1] × sizes[i]. */
  readonly weights: Matrix[];

  // Mini-batch state, exposed so the teaching UI can show "accumulator so far".
  miniBatchSampleIndex = 0;
  currentMiniBatchSize = 0;
  currentNablaW: Matrix[] | null = null;
  currentNablaB: Vector[] | null = null;

  // Per-sample teaching snapshots, refreshed at the end of each Backprop call.
  lastOutputTrace: OutputTrace | null = null;
  lastHiddenTrace: HiddenTrace | null = null;

  constructor(sizes: readonly number[], seed: number) {
    if (sizes.length < 2) throw new Error('Need at least an input and an output layer.');
    this.sizes = sizes.slice();
    const rng = mulberry32(seed);
    this.biases = [];
    this.weights = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const rows = sizes[i + 1];
      const cols = sizes[i];
      // Match the C# Random + Normal(0,1) init from Network.cs: weights and biases ~ N(0, 1).
      const bData = new Float64Array(rows);
      for (let r = 0; r < rows; r++) bData[r] = standardNormal(rng);
      this.biases.push(new Vector(bData));
      this.weights.push(Matrix.randn(rows, cols, rng));
    }
  }

  /** Forward pass; returns the output-layer activation a[L]. */
  feedForward(input: Vector): Vector {
    let a = input;
    for (let i = 0; i < this.weights.length; i++) {
      a = this.weights[i].mulVec(a).add(this.biases[i]).sigmoid();
    }
    return a;
  }

  /**
   * One mini-batch SGD update: backprop each sample's gradient into the
   * accumulator, then apply `w ← w − (η/m)·ΔW`. Fires the `onSample` callback
   * after each sample so the teaching UI can step through.
   */
  updateMiniBatch(
    batch: ReadonlyArray<{ x: Vector; y: Vector }>,
    learningRate: number,
    onSample: (predictedLabel: number) => Promise<void> | void,
  ): Promise<void> {
    return (async () => {
      this.currentMiniBatchSize = batch.length;
      this.currentNablaB = this.biases.map(b => Vector.zeros(b.size));
      this.currentNablaW = this.weights.map(w => Matrix.zeros(w.rows, w.cols));

      for (let k = 0; k < batch.length; k++) {
        this.miniBatchSampleIndex = k;
        const { x, y } = batch[k];
        const { nablaB, nablaW } = this.backprop(x, y);
        for (let i = 0; i < this.currentNablaB.length; i++) {
          this.currentNablaB[i].addInPlace(nablaB[i]);
          this.currentNablaW[i].addInPlace(nablaW[i]);
        }
        const predicted = this.lastOutputTrace!.outputActivation.argMax();
        await onSample(predicted);
      }

      const scale = learningRate / batch.length;
      for (let i = 0; i < this.weights.length; i++) {
        this.weights[i].axpyInPlace(scale, this.currentNablaW![i]);
        this.biases[i].axpyInPlace(scale, this.currentNablaB![i]);
      }

      // Conceptual reset; UI uses null to signal "between batches".
      this.currentNablaB = null;
      this.currentNablaW = null;
    })();
  }

  /**
   * One sample's gradient via backprop. Captures the OutputTrace + HiddenTrace
   * along the way so the teaching panels don't have to recompute.
   */
  private backprop(x: Vector, y: Vector): { nablaB: Vector[]; nablaW: Matrix[] } {
    const nablaB: Vector[] = this.biases.map(b => Vector.zeros(b.size));
    const nablaW: Matrix[] = this.weights.map(w => Matrix.zeros(w.rows, w.cols));

    // Forward, remembering every z and a.
    const activations: Vector[] = [x];
    const zs: Vector[] = [];
    let a = x;
    for (let i = 0; i < this.weights.length; i++) {
      const z = this.weights[i].mulVec(a).add(this.biases[i]);
      zs.push(z);
      a = z.sigmoid();
      activations.push(a);
    }

    // Output layer δ.
    const outputActivation = activations[activations.length - 1];
    const zL = zs[zs.length - 1];
    const spL = zL.sigmoidPrime();
    const error = outputActivation.sub(y);
    let delta = error.pointwiseMul(spL);
    const hiddenActivation = activations[activations.length - 2];

    nablaB[nablaB.length - 1] = delta;
    const perSampleNablaWL = delta.outer(hiddenActivation);
    nablaW[nablaW.length - 1] = perSampleNablaWL;

    this.lastOutputTrace = {
      hiddenActivation,
      outputActivation,
      weightedInput: zL,
      sigmoidPrime: spL,
      target: y,
      error,
      delta,
      perSampleNablaW: perSampleNablaWL,
    };

    // Walk backward through hidden layers (for a 3-layer net this fires once, at l = 2).
    this.lastHiddenTrace = null;
    for (let l = 2; l < this.sizes.length; l++) {
      const z = zs[zs.length - l];
      const sp = z.sigmoidPrime();
      const preSigmoid = this.weights[this.weights.length - l + 1].transposeMulVec(delta);
      delta = preSigmoid.pointwiseMul(sp);
      nablaB[nablaB.length - l] = delta;
      nablaW[nablaW.length - l] = delta.outer(activations[activations.length - l - 1]);

      if (l === 2) {
        this.lastHiddenTrace = {
          input: activations[0],
          weightedInput: z,
          sigmoidPrime: sp,
          backpropPreSigmoid: preSigmoid,
          delta,
        };
      }
    }

    return { nablaB, nablaW };
  }

  /** Count of correctly-classified samples in the test set. */
  evaluate(testData: ReadonlyArray<{ x: Vector; label: number }>): number {
    let correct = 0;
    for (const { x, label } of testData) {
      if (this.feedForward(x).argMax() === label) correct++;
    }
    return correct;
  }

  /** Mean ½‖a − y‖² over the test set, with `y` one-hot encoded. */
  meanLoss(testData: ReadonlyArray<{ x: Vector; label: number }>, classes: number): number {
    let total = 0;
    for (const { x, label } of testData) {
      const a = this.feedForward(x);
      let s = 0;
      for (let k = 0; k < classes; k++) {
        const diff = a.data[k] - (k === label ? 1 : 0);
        s += diff * diff;
      }
      total += 0.5 * s;
    }
    return total / testData.length;
  }
}

/** Box–Muller normal draw — local copy so this file doesn't leak the helper. */
function standardNormal(rng: () => number): number {
  let u1 = rng();
  while (u1 <= 0) u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
