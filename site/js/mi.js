// Plug-in and Miller–Madow mutual information between region ids and labels,
// as in src_experiment/estimators.py (bits; the Miller–Madow correction uses the
// number of occupied regions and observed classes).

/** {plugIn, millerMadow, hY, regions, classes} for integer ids and labels. */
export function routingMI(ids, y, N = ids.length) {
  const joint = new Map();
  const nOmega = new Map();
  const nY = new Map();
  for (let i = 0; i < N; i++) {
    const w = ids[i];
    const c = y[i];
    const key = w * 1024 + c;
    joint.set(key, (joint.get(key) || 0) + 1);
    nOmega.set(w, (nOmega.get(w) || 0) + 1);
    nY.set(c, (nY.get(c) || 0) + 1);
  }
  let plugIn = 0;
  for (const [key, n] of joint) {
    const w = Math.floor(key / 1024);
    const c = key % 1024;
    plugIn += (n / N) * Math.log2((n * N) / (nOmega.get(w) * nY.get(c)));
  }
  let hY = 0;
  for (const n of nY.values()) hY -= (n / N) * Math.log2(n / N);
  const R = nOmega.size;
  const C = nY.size;
  const millerMadow = plugIn - ((R - 1) * (C - 1)) / (2 * N * Math.LN2);
  return { plugIn, millerMadow, hY, regions: R, classes: C };
}
