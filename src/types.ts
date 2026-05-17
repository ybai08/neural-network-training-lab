// Shared types between main thread and worker. Worker can't share JS objects with
// the main thread by reference; everything moves through structured-clone via
// postMessage, so the snapshot shapes here are intentionally plain (typed arrays
// + numbers + strings — no class instances).

/**
 * Phase within one sample's backprop journey. Order matches the per-layer
 * "have δ → update this layer (Gradient) → propagate δ upstream (HiddenDelta)" loop.
 */
export const Phase = {
  Forward: 0,
  Loss: 1,
  OutputDelta: 2,
  Gradient: 3,
  HiddenDelta: 4,
} as const;
export type Phase = (typeof Phase)[keyof typeof Phase];
export const PhaseCount = 5;

/**
 * Per-sample backprop intermediates the UI needs. Mirrors the union of
 * OutputBackpropSnapshot + HiddenBackpropSnapshot from the C# version, but
 * flattened so it survives structured-clone cheaply.
 *
 * All vectors / matrices are row-major Float64Array. Matrix shape is documented
 * per field; the `MatrixView` / `VectorView` UI knows the layout.
 */
export interface SampleSnapshot {
  // Raw input pixel intensities (0..1), `inputSide * inputSide` long.
  pixels: Float64Array;
  inputSide: number;             // = 14 for our downsampled net

  trueLabel: number;             // 0..9
  predictedLabel: number;        // argmax(a[2])

  miniBatchSampleIndex: number;  // 0-based index of THIS sample inside its batch
  miniBatchSize: number;

  // Forward pass
  hiddenWeightedInput: Float64Array; // z[1], len = 30
  hiddenActivation: Float64Array;    // a[1], len = 30
  outputWeightedInput: Float64Array; // z[2], len = 10
  outputActivation: Float64Array;    // a[2], len = 10

  // Loss
  target: Float64Array;              // y (one-hot), len = 10
  error: Float64Array;               // a[2] - y, len = 10

  // Output δ
  outputSigmoidPrime: Float64Array;  // σ'(z[2]), len = 10
  outputDelta: Float64Array;         // δ[2] = error ⊙ σ'(z[2]), len = 10

  // Per-sample gradient at the output layer (also = δ[2] · a[1]ᵀ).
  perSampleNablaW2: Float64Array;    // 10 × 30, row-major

  // Backprop into hidden layer
  hiddenBackpropPreSigmoid: Float64Array; // W[2]ᵀ · δ[2], len = 30
  hiddenSigmoidPrime: Float64Array;       // σ'(z[1]), len = 30
  hiddenDelta: Float64Array;              // δ[1], len = 30

  // Network's current weights / biases (post-update if this sample triggered one).
  // Kept here because the UI re-renders them in step with each sample anyway.
  weightsW1: Float64Array; // 30 × 196, row-major   (W[1])
  weightsW2: Float64Array; // 10 × 30,  row-major   (W[2])
  biasesB1: Float64Array;  // len 30                 (b[1])
  biasesB2: Float64Array;  // len 10                 (b[2])

  // Mini-batch accumulators after this sample's gradient was folded in.
  // null between batches (after Apply, before the next batch starts).
  accumNablaW2: Float64Array | null; // 10 × 30
  accumNablaB2: Float64Array | null; // 10
}

/** Per-epoch evaluation snapshot — used to plot loss / accuracy curves. */
export interface EpochSnapshot {
  epoch: number;        // 0-based
  meanTestLoss: number; // ½‖a − y‖² averaged across the test set
  testCorrect: number;
  testTotal: number;
  label?: string;       // e.g. "start", "sample 100", or "epoch 1"
}

/**
 * Per-sample snapshot for the /convolutional page. Includes every feature
 * map along the pipeline + the current filter banks + the FC layer's weights
 * so the UI can render the whole forward pass live.
 */
export interface ConvNetSnapshot {
  pixels: Float64Array;        // 14×14
  inputSide: number;           // 14
  trueLabel: number;
  predictedLabel: number;
  miniBatchSampleIndex: number;
  miniBatchSize: number;

  // Forward activations (post-arithmetic, pre-next-layer).
  conv1Pre: Float64Array;      // 12×12×4 — output of Conv1 before ReLU
  conv1Relu: Float64Array;     // 12×12×4
  pool1Out: Float64Array;      // 6×6×4
  conv2Pre: Float64Array;      // 4×4×8
  conv2Relu: Float64Array;     // 4×4×8 (== flatten contents)
  fcOut: Float64Array;         // 10 — z = W·flatten + b
  output: Float64Array;        // 10 — σ(z)
  target: Float64Array;        // 10 — one-hot

  // Live parameter snapshot. Lets the UI show "what each filter currently
  // looks like" alongside its feature map.
  conv1Kernels: Float64Array;  // 4 × 1 × 3 × 3
  conv1Biases: Float64Array;   // 4
  conv2Kernels: Float64Array;  // 8 × 4 × 3 × 3
  conv2Biases: Float64Array;   // 8
  fcWeights: Float64Array;     // 10 × 128
  fcBiases: Float64Array;      // 10
}

// ─── Messages: main → worker ───────────────────────────────────────────────

export type MainToWorker =
  | { type: 'init'; learningRate: number; miniBatchSize: number; epochs: number; seed: number; stepMode: boolean }
  | { type: 'start' }
  | { type: 'stop' }
  | { type: 'stepNext' }      // advance one sample (only meaningful in step mode)
  | { type: 'setStepMode'; on: boolean }
  | { type: 'setThrottleMs'; ms: number };

// ─── Messages: worker → main ───────────────────────────────────────────────

export type WorkerToMain =
  | { type: 'status'; text: string }
  | { type: 'sample'; snapshot: SampleSnapshot }
  | { type: 'miniBatchApplied' }
  | { type: 'epoch'; snapshot: EpochSnapshot }
  | { type: 'done' }
  | { type: 'error'; message: string };

/** Convolutional-page version of the worker → main protocol. */
export type ConvWorkerToMain =
  | { type: 'status'; text: string }
  | { type: 'sample'; snapshot: ConvNetSnapshot }
  | { type: 'miniBatchApplied' }
  | { type: 'epoch'; snapshot: EpochSnapshot }
  | { type: 'done' }
  | { type: 'error'; message: string };
