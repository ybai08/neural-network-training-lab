// MNIST loader — fetches the four .gz files from a public mirror, decompresses
// them via the browser's native DecompressionStream (no pako dep), parses the
// IDX binary format, and block-mean-downsamples each 28×28 image to 14×14 so
// diagram can show all 196 input neurons without truncating.
//
// IDX format reference (Yann LeCun): big-endian magic + dims + raw uint8 pixel
// or label bytes. Image magic 0x00000803, label magic 0x00000801.

// MNIST .gz files are bundled in /public/mnist/ so Vite serves them same-origin.
// (The S3 mirror at ossci-datasets.s3.amazonaws.com sends no CORS headers, so a
// direct cross-origin fetch from the browser fails.) Vite resolves `import.meta.env.BASE_URL`
// to the build's base path so this works both in dev and after a static deploy.
const MIRROR_BASE = `${import.meta.env.BASE_URL}mnist/`;
// Files are renamed from `.gz` to `.bin` so static hosts don't slap a
// `Content-Encoding: gzip` header on them (the browser would then auto-decompress
// before our DecompressionStream ran, leaving us decoding twice).
const TRAIN_IMAGES = 'train-images-idx3-ubyte.bin';
const TRAIN_LABELS = 'train-labels-idx1-ubyte.bin';
const TEST_IMAGES = 't10k-images-idx3-ubyte.bin';
const TEST_LABELS = 't10k-labels-idx1-ubyte.bin';

const SRC_SIDE = 28;
export const INPUT_SIDE = 14;            // post-downsample side
export const INPUT_PIXELS = INPUT_SIDE * INPUT_SIDE;

export interface MnistDataset {
  /** Flat per-image pixel arrays. Length = count; each entry = INPUT_PIXELS doubles in [0, 1]. */
  images: Float64Array[];
  /** Class labels 0..9, same length as images. */
  labels: Uint8Array;
}

export interface MnistData {
  train: MnistDataset;
  test: MnistDataset;
}

export type ProgressFn = (text: string) => void;

/**
 * Load and downsample MNIST. Roughly 1-2 s on a warm cache; first run
 * downloads ~11 MB total from the public mirror.
 */
export async function loadMnist(progress: ProgressFn = () => {}): Promise<MnistData> {
  progress('Loading MNIST — train images...');
  const trainImages = await loadImages(TRAIN_IMAGES);
  progress('Loading MNIST — train labels...');
  const trainLabels = await loadLabels(TRAIN_LABELS);
  progress('Loading MNIST — test images...');
  const testImages = await loadImages(TEST_IMAGES);
  progress('Loading MNIST — test labels...');
  const testLabels = await loadLabels(TEST_LABELS);

  if (trainImages.length !== trainLabels.length) throw new Error('Train image/label count mismatch');
  if (testImages.length !== testLabels.length) throw new Error('Test image/label count mismatch');

  return {
    train: { images: trainImages, labels: trainLabels },
    test: { images: testImages, labels: testLabels },
  };
}

async function loadImages(file: string): Promise<Float64Array[]> {
  const bytes = await fetchAndGunzip(file);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, false);
  if (magic !== 0x00000803) throw new Error(`Unexpected magic 0x${magic.toString(16)} in ${file}`);
  const count = view.getUint32(4, false);
  const rows = view.getUint32(8, false);
  const cols = view.getUint32(12, false);
  if (rows !== SRC_SIDE || cols !== SRC_SIDE) {
    throw new Error(`Expected ${SRC_SIDE}×${SRC_SIDE} images, got ${rows}×${cols}`);
  }
  const pixels = rows * cols;
  const out: Float64Array[] = new Array(count);
  let off = 16;
  for (let i = 0; i < count; i++) {
    // The raw byte slice is normalised to [0, 1] doubles, then block-mean-downsampled.
    out[i] = downsample(bytes.subarray(off, off + pixels));
    off += pixels;
  }
  return out;
}

async function loadLabels(file: string): Promise<Uint8Array> {
  const bytes = await fetchAndGunzip(file);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, false);
  if (magic !== 0x00000801) throw new Error(`Unexpected magic 0x${magic.toString(16)} in ${file}`);
  const count = view.getUint32(4, false);
  return bytes.subarray(8, 8 + count);
}

/**
 * Mirrors `TrainingSession.ResizeInput` in C#: each output pixel = mean of the
 * corresponding (28/INPUT_SIDE)² block. With INPUT_SIDE=14 the block is 2×2,
 * so each downsampled pixel is the average of 4 source pixels.
 */
function downsample(src: Uint8Array): Float64Array {
  const block = SRC_SIDE / INPUT_SIDE;
  if (!Number.isInteger(block)) throw new Error('INPUT_SIDE must divide 28 evenly.');
  const inv = 1 / (block * block * 255);
  const dst = new Float64Array(INPUT_PIXELS);
  for (let y = 0; y < INPUT_SIDE; y++) {
    for (let x = 0; x < INPUT_SIDE; x++) {
      let sum = 0;
      const sy0 = y * block;
      const sx0 = x * block;
      for (let dy = 0; dy < block; dy++) {
        const rowOff = (sy0 + dy) * SRC_SIDE;
        for (let dx = 0; dx < block; dx++) sum += src[rowOff + sx0 + dx];
      }
      dst[y * INPUT_SIDE + x] = sum * inv;
    }
  }
  return dst;
}

/**
 * Fetch a .gz file and decompress it via the browser's native streaming
 * gunzip. Returns the decompressed bytes as a single contiguous Uint8Array.
 */
async function fetchAndGunzip(file: string): Promise<Uint8Array> {
  const resp = await fetch(MIRROR_BASE + file);
  if (!resp.ok) throw new Error(`Failed to fetch ${file}: ${resp.status}`);
  if (!resp.body) throw new Error(`No body for ${file}`);
  const ds = new DecompressionStream('gzip');
  const stream = resp.body.pipeThrough(ds);
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = stream.getReader();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}
