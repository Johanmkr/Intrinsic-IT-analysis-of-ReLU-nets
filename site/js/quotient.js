// Functional quotient: ε-clustering of data-supported regions by their active
// linear maps. A port of cluster_functional in src_experiment/functional_quotient.py:
// regions are visited in first-encounter order, only regions with the same
// active set S_l are compared, and a region joins the first class whose leader
// is within relative Frobenius distance ε (bias excluded).

import { activeSubnetwork, regionIds } from "./relu.js";

function relativeFrobenius(A1, A2) {
  let d = 0, n1 = 0, n2 = 0;
  for (let k = 0; k < A1.length; k++) {
    d += (A1[k] - A2[k]) ** 2;
    n1 += A1[k] ** 2;
    n2 += A2[k] ** 2;
  }
  const denom = 0.5 * (Math.sqrt(n1) + Math.sqrt(n2));
  return denom < 1e-12 ? 0 : Math.sqrt(d) / denom;
}

/** Active maps of the data-supported regions at `layer`, in first-encounter order. */
export function regionMaps(net, patterns, N, layer) {
  const regions = regionIds(patterns, N, layer);
  const maps = regions.first.map((i) => activeSubnetwork(net, patterns, N, i, layer));
  return { ...regions, maps };
}

/** Quotient class of every region (Int32Array over region ids) and the class count. */
export function clusterFunctional(maps, eps) {
  const qid = new Int32Array(maps.length);
  const leaders = new Map(); // active-set key → [[class id, A], ...]
  let next = 0;
  maps.forEach((m, r) => {
    let bucket = leaders.get(m.key);
    if (!bucket) { bucket = []; leaders.set(m.key, bucket); }
    for (const [q, A] of bucket) {
      if (relativeFrobenius(m.A, A) <= eps) { qid[r] = q; return; }
    }
    qid[r] = next;
    bucket.push([next, m.A]);
    next++;
  });
  return { qid, count: next };
}
