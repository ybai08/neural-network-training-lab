// Minimal Float64Array-backed linear algebra. Only the ops Backprop actually
// uses are here — no general matrix library aspirations. Row-major storage
// throughout (data[r * cols + c]).

export class Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
    if (this.data.length !== rows * cols) {
      throw new Error(`Matrix data length ${this.data.length} != ${rows}×${cols}`);
    }
  }

  /** Construct a matrix of `rows × cols` zeros. */
  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /** Sample i.i.d. N(0, 1) entries via the given RNG. */
  static randn(rows: number, cols: number, rng: () => number): Matrix {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = standardNormal(rng);
    return m;
  }

  at(r: number, c: number): number { return this.data[r * this.cols + c]; }
  set(r: number, c: number, v: number): void { this.data[r * this.cols + c] = v; }

  /** y = A · x, where A is this (rows × cols) and x is a Vector of length `cols`. */
  mulVec(x: Vector): Vector {
    if (x.size !== this.cols) throw new Error(`mulVec: size mismatch ${this.cols} vs ${x.size}`);
    const out = new Vector(this.rows);
    for (let r = 0; r < this.rows; r++) {
      let sum = 0;
      const off = r * this.cols;
      for (let c = 0; c < this.cols; c++) sum += this.data[off + c] * x.data[c];
      out.data[r] = sum;
    }
    return out;
  }

  /** y = Aᵀ · x — used to propagate δ backward through a weight layer. */
  transposeMulVec(x: Vector): Vector {
    if (x.size !== this.rows) throw new Error(`transposeMulVec: size mismatch ${this.rows} vs ${x.size}`);
    const out = new Vector(this.cols);
    for (let c = 0; c < this.cols; c++) {
      let sum = 0;
      for (let r = 0; r < this.rows; r++) sum += this.data[r * this.cols + c] * x.data[r];
      out.data[c] = sum;
    }
    return out;
  }

  /** In-place: this += other. */
  addInPlace(other: Matrix): void {
    if (other.rows !== this.rows || other.cols !== this.cols) throw new Error('addInPlace: shape mismatch');
    for (let i = 0; i < this.data.length; i++) this.data[i] += other.data[i];
  }

  /** In-place: this -= scale · other.  (Used for the SGD update.) */
  axpyInPlace(scale: number, other: Matrix): void {
    if (other.rows !== this.rows || other.cols !== this.cols) throw new Error('axpyInPlace: shape mismatch');
    for (let i = 0; i < this.data.length; i++) this.data[i] -= scale * other.data[i];
  }

  /** Copy entries into a fresh Float64Array (for postMessage; preserves data layout). */
  toFlat(): Float64Array {
    return new Float64Array(this.data);
  }
}

export class Vector {
  readonly size: number;
  readonly data: Float64Array;

  constructor(size: number | Float64Array) {
    if (typeof size === 'number') {
      this.size = size;
      this.data = new Float64Array(size);
    } else {
      this.size = size.length;
      this.data = size;
    }
  }

  static zeros(size: number): Vector { return new Vector(size); }
  static of(values: number[] | Float64Array): Vector { return new Vector(Float64Array.from(values)); }

  /** Element-wise σ.  σ(z) = 1 / (1 + e^(−z)). */
  sigmoid(): Vector {
    const out = new Vector(this.size);
    for (let i = 0; i < this.size; i++) out.data[i] = 1 / (1 + Math.exp(-this.data[i]));
    return out;
  }

  /** Element-wise σ'(z) = σ(z)·(1 − σ(z)). */
  sigmoidPrime(): Vector {
    const out = new Vector(this.size);
    for (let i = 0; i < this.size; i++) {
      const s = 1 / (1 + Math.exp(-this.data[i]));
      out.data[i] = s * (1 - s);
    }
    return out;
  }

  /** this + other.  Returns a new vector. */
  add(other: Vector): Vector {
    if (other.size !== this.size) throw new Error('Vector.add: size mismatch');
    const out = new Vector(this.size);
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] + other.data[i];
    return out;
  }

  /** this − other.  Returns a new vector. */
  sub(other: Vector): Vector {
    if (other.size !== this.size) throw new Error('Vector.sub: size mismatch');
    const out = new Vector(this.size);
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] - other.data[i];
    return out;
  }

  /** Element-wise (Hadamard) multiply. */
  pointwiseMul(other: Vector): Vector {
    if (other.size !== this.size) throw new Error('Vector.pointwiseMul: size mismatch');
    const out = new Vector(this.size);
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] * other.data[i];
    return out;
  }

  /** In-place: this += other. */
  addInPlace(other: Vector): void {
    if (other.size !== this.size) throw new Error('Vector.addInPlace: size mismatch');
    for (let i = 0; i < this.size; i++) this.data[i] += other.data[i];
  }

  /** In-place: this -= scale · other. */
  axpyInPlace(scale: number, other: Vector): void {
    if (other.size !== this.size) throw new Error('Vector.axpyInPlace: size mismatch');
    for (let i = 0; i < this.size; i++) this.data[i] -= scale * other.data[i];
  }

  /** Outer product: this ⊗ other → Matrix(this.size × other.size). */
  outer(other: Vector): Matrix {
    const out = new Matrix(this.size, other.size);
    for (let r = 0; r < this.size; r++) {
      const off = r * other.size;
      const v = this.data[r];
      for (let c = 0; c < other.size; c++) out.data[off + c] = v * other.data[c];
    }
    return out;
  }

  /** Index of the largest entry. */
  argMax(): number {
    let best = 0;
    let bestVal = this.data[0];
    for (let i = 1; i < this.size; i++) if (this.data[i] > bestVal) { bestVal = this.data[i]; best = i; }
    return best;
  }

  /** Copy entries into a fresh Float64Array (used for snapshots → postMessage). */
  toFlat(): Float64Array { return new Float64Array(this.data); }
}

// ─── RNG ──────────────────────────────────────────────────────────────────

/**
 * Deterministic 32-bit RNG (mulberry32). Seeded so weight initialisation can
 * match the C# version's run-to-run reproducibility.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box–Muller transform: turn two uniforms into one standard-normal sample.
 * We discard the second sample for simplicity; a 2× perf win is irrelevant
 * at network-initialisation scale.
 */
function standardNormal(rng: () => number): number {
  // u1 mustn't be zero; redraw if it is (vanishingly rare).
  let u1 = rng();
  while (u1 <= 0) u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
