// ReLU networks: forward pass, activation patterns, data-supported regions and
// the active subnetwork of a region. A port of src_experiment/routing_estimator.py
// and compute_active_subnetwork in src_experiment/functional_quotient.py.
//
// Weights come from site/data/networks.json (PyTorch layout, W[i] is out × in).
// The pipeline computes in float32; here every pre-activation is rounded to
// float32 (Math.fround) so points land in the same region except, rarely,
// points lying exactly on a region boundary.

const f32 = Math.fround;

/** Network from one checkpoint of networks.json: [{W, b}, ...], hidden layers + output. */
export function makeNet(layers) {
  return layers.map(({ W, b }) => ({
    out: W.length,
    in: W[0].length,
    W: Float32Array.from(W.flat()),
    b: Float32Array.from(b),
  }));
}

/** Forward pass of N points (X: Float32Array, N × d). Returns the activation
 *  pattern of every hidden layer (Uint8Array, N × width) and the logits. */
export function forward(net, X, N) {
  const patterns = [];
  let a = X;
  for (let l = 0; l < net.length; l++) {
    const { out, in: inp, W, b } = net[l];
    const z = new Float32Array(N * out);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < out; j++) {
        let s = 0;
        for (let k = 0; k < inp; k++) s += W[j * inp + k] * a[i * inp + k];
        z[i * out + j] = f32(f32(s) + b[j]);
      }
    }
    if (l === net.length - 1) return { patterns, logits: z };
    const p = new Uint8Array(N * out);
    for (let t = 0; t < z.length; t++) {
      if (z[t] > 0) p[t] = 1;
      else z[t] = 0;
    }
    patterns.push(p);
    a = z;
  }
}

/** Region id of every point at hidden layer `layer` (1-indexed): points share an
 *  id iff their cumulative patterns (layers 1..layer) agree. Ids are numbered in
 *  first-encounter order; `first[id]` is the first point of each region and
 *  `keys[id]` its pattern key (comparable across calls). */
export function regionIds(patterns, N, layer) {
  const widths = patterns.slice(0, layer).map((p) => p.length / N);
  const ids = new Int32Array(N);
  const index = new Map();
  const first = [];
  const keys = [];
  for (let i = 0; i < N; i++) {
    let key = "";
    for (let l = 0; l < layer; l++) {
      const w = widths[l];
      const p = patterns[l];
      let word = 0;
      for (let j = 0; j < w; j++) {
        word = (word << 1) | p[i * w + j];
        if ((j + 1) % 30 === 0 || j === w - 1) { key += word.toString(36) + "."; word = 0; }
      }
      key += "|";
    }
    let id = index.get(key);
    if (id === undefined) { id = first.length; index.set(key, id); first.push(i); keys.push(key); }
    ids[i] = id;
  }
  return { ids, count: first.length, first, keys };
}

/** Accuracy of argmax(logits) against labels. */
export function accuracy(logits, y, N) {
  const C = logits.length / N;
  let correct = 0;
  for (let i = 0; i < N; i++) {
    let best = 0;
    for (let c = 1; c < C; c++) if (logits[i * C + c] > logits[i * C + best]) best = c;
    if (best === y[i]) correct++;
  }
  return correct / N;
}

/** Active subnetwork of the region containing point i at hidden layer `layer`:
 *  Ã = W^l[S_l, S_{l-1}] … W^1[S_1, :]  (|S_l| × n0, float64) and the key of S_l.
 *  The bias is not needed: the quotient criterion uses the linear part only. */
export function activeSubnetwork(net, patterns, N, i, layer) {
  const n0 = net[0].in;
  let S = Array.from({ length: n0 }, (_, k) => k);
  let A = new Float64Array(n0 * n0);
  for (let k = 0; k < n0; k++) A[k * n0 + k] = 1;
  for (let l = 0; l < layer; l++) {
    const { out, in: inp, W } = net[l];
    const p = patterns[l];
    const Snew = [];
    for (let j = 0; j < out; j++) if (p[i * out + j]) Snew.push(j);
    const B = new Float64Array(Snew.length * n0);
    for (let r = 0; r < Snew.length; r++) {
      for (let c = 0; c < n0; c++) {
        let s = 0;
        for (let m = 0; m < S.length; m++) s += W[Snew[r] * inp + S[m]] * A[m * n0 + c];
        B[r * n0 + c] = s;
      }
    }
    A = B;
    S = Snew;
  }
  return { A, rows: S.length, key: S.join(",") };
}
