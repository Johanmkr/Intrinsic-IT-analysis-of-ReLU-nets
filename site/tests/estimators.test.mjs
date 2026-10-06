// The browser estimators reproduce the pipeline on the exported networks:
// identical region counts and quotient sizes, MI equal to rounding error.
// Run: node --test "site/tests/*.test.mjs"  (or: cd site && npm test)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { forward, makeNet, regionIds, accuracy } from "../js/relu.js";
import { routingMI } from "../js/mi.js";
import { clusterFunctional, regionMaps } from "../js/quotient.js";

const data = (f) => JSON.parse(readFileSync(new URL(`../data/${f}`, import.meta.url)));
const points = data("points.json");
const networks = data("networks.json");
const reference = data("reference.json");
const N = points.N;
const X = Float32Array.from(points.X.flat());
const y = Int32Array.from(points.y);

for (const [key, net] of Object.entries(networks)) {
  test(`${key}: regions, routing MI and accuracy match the pipeline at every exported epoch`, () => {
    let worstMI = 0, worstRegions = 0;
    for (const [epoch, layers] of Object.entries(net.weights)) {
      const { patterns, logits } = forward(makeNet(layers), X, N);
      for (const ref of reference[key].filter((r) => r.epoch === +epoch && r.epsilon === 0)) {
        const { ids } = regionIds(patterns, N, ref.layer);
        const mi = routingMI(ids, y, N);
        worstRegions = Math.max(worstRegions, Math.abs(mi.regions - ref.num_regions));
        worstMI = Math.max(worstMI, Math.abs(mi.plugIn - ref.plug_in_bits));
        assert.ok(Math.abs(mi.hY - ref.H_Y_bits) < 1e-9);
        assert.ok(Math.abs(accuracy(logits, y, N) - ref.accuracy) < 1e-12, `accuracy epoch ${epoch}`);
      }
    }
    assert.equal(worstRegions, 0, `region count off by ${worstRegions}`);
    assert.ok(worstMI <= 1e-9, `plug-in MI off by ${worstMI}`);
  });

  test(`${key}: functional quotient matches the pipeline at the last epoch`, () => {
    const { patterns } = forward(makeNet(net.weights["150"]), X, N);
    let worstQ = 0, worstMI = 0;
    for (const ref of reference[key].filter((r) => r.epoch === 150)) {
      const { ids, maps, count } = regionMaps(makeNet(net.weights["150"]), patterns, N, ref.layer);
      const { qid, count: q } = clusterFunctional(maps, ref.epsilon);
      const mi = routingMI(ids.map((r) => qid[r]), y, N);
      worstQ = Math.max(worstQ, Math.abs(q - ref.num_quotient));
      worstMI = Math.max(worstMI, Math.abs(mi.plugIn - ref.plug_in_func_bits));
      assert.ok(q <= count);
    }
    assert.equal(worstQ, 0, `quotient size off by ${worstQ}`);
    assert.ok(worstMI <= 1e-9, `quotient MI off by ${worstMI}`);
  });
}

test("a single layer never merges distinct regions (ρ_func = 1 at layer 1)", () => {
  const net = makeNet(networks.c999.weights["150"]);
  const { patterns } = forward(net, X, N);
  const { maps, count } = regionMaps(net, patterns, N, 1);
  assert.equal(clusterFunctional(maps, 0.3).count, count);
});
