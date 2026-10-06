// One network evaluated on the drawing grid and on the data points: regions,
// routing MI and (optionally) the functional quotient, as the demos display them.

import { accuracy, forward, makeNet, regionIds } from "./relu.js";
import { routingMI } from "./mi.js";
import { clusterFunctional, regionMaps } from "./quotient.js";

const cache = new Map();

/** Regions of `layers` (one checkpoint) at hidden layer `layer`, on grid and data. */
export function evaluate(layers, layer, grid, G, data, cacheKey = null) {
  const key = cacheKey && `${cacheKey}|${layer}|${G}`;
  if (key && cache.has(key)) return cache.get(key);
  const net = makeNet(layers);
  const pts = forward(net, data.X, data.N);
  const onData = regionMaps(net, pts.patterns, data.N, layer); // first-encounter order, as in the pipeline
  const onGrid = regionIds(forward(net, grid, G * G).patterns, G * G, layer);
  const dataIndex = new Map(onData.keys.map((k, i) => [k, i]));
  const gridToData = onGrid.keys.map((k) => dataIndex.get(k) ?? -1);
  const cell = Int32Array.from(onGrid.ids, (g) => gridToData[g]);
  const mi = routingMI(onData.ids, data.y, data.N);
  const out = {
    net, onData, cell, gridRegions: onGrid.ids, regionsInView: onGrid.count, mi,
    accuracy: accuracy(pts.logits, data.y, data.N),
    rho: onData.count / data.N,
  };
  if (key) { cache.set(key, out); if (cache.size > 60) cache.delete(cache.keys().next().value); }
  return out;
}

/** Routing MI and ρ on the data points only (no grid): for curves over epochs. */
export function evaluatePoints(layers, layer, data, cacheKey) {
  const key = `pts|${cacheKey}|${layer}`;
  if (cache.has(key)) return cache.get(key);
  const pts = forward(makeNet(layers), data.X, data.N);
  const { ids, count } = regionIds(pts.patterns, data.N, layer);
  const out = { mi: routingMI(ids, data.y, data.N), rho: count / data.N, accuracy: accuracy(pts.logits, data.y, data.N) };
  cache.set(key, out);
  return out;
}

/** Functional quotient of an evaluated view at tolerance ε. */
export function quotient(view, eps, data) {
  const { qid, count } = clusterFunctional(view.onData.maps, eps);
  const mi = routingMI(Int32Array.from(view.onData.ids, (r) => qid[r]), data.y, data.N);
  const cell = Int32Array.from(view.cell, (r) => (r < 0 ? -1 : qid[r]));
  return { qid, count, mi, cell, rhoFunc: count / view.onData.count };
}
