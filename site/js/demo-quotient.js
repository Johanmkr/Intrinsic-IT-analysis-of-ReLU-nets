// Demo 2: the functional quotient — regions whose active linear maps are within
// relative Frobenius distance ε merge into one class (first-encounter order).

import { chart, css, drawPartition, gridPoints } from "./plot.js";
import { evaluate, quotient } from "./view.js";
import { $, bits, onInput, onRedraw, stats } from "./ui.js";

const G = 220;
const EPS_GRID = Array.from({ length: 41 }, (_, i) => i * 0.05);

export function initQuotient(root, data) {
  const net = $("[name=net]", root), layer = $("[name=layer]", root), eps = $("[name=eps]", root);
  const epsOut = $("output[for=d2-eps]", root);
  const map = $("canvas.map", root), plot = $("canvas.chart", root), dl = $("dl", root);
  const grid = gridPoints(G);
  const curves = new Map();

  function draw() {
    const v = evaluate(data.networks[net.value].weights["150"], +layer.value, grid, G, data.points, `${net.value}|150`);
    const e = +eps.value / 100;
    epsOut.value = e.toFixed(2);
    const q = quotient(v, e, data.points);
    drawPartition(map, G, q.cell, { edge: q.cell, faint: v.gridRegions });
    stats(dl, [
      ["|Ω<sub>D</sub>|", v.onData.count, "data-supported regions"],
      ["|Ω<sub>func</sub>|", q.count, "quotient classes"],
      ["ρ<sub>func</sub>", q.rhoFunc.toFixed(3), "|Ω_func| / |Ω_D|"],
      ["Î(Y; Ω<sub>D</sub>)", bits(v.mi.plugIn)],
      ["Î<sub>func</sub>", bits(q.mi.plugIn)],
      ["H(Y)", bits(v.mi.hY)],
    ]);
    const key = `${net.value}|${layer.value}`;
    if (!curves.has(key)) curves.set(key, EPS_GRID.map((x) => quotient(v, x, data.points)));
    const c = curves.get(key);
    chart(plot, [
      { x: EPS_GRID, y: c.map((r) => r.rhoFunc), color: css("--accent"), label: "ρ_func = |Ω_func| / |Ω_D|" },
      { x: EPS_GRID, y: c.map((r) => r.mi.plugIn / v.mi.hY), color: css("--accent2"), dash: true, label: "Î_func / H(Y)" },
    ], { xlabel: "ε", ylabel: "fraction", ymin: 0, ymax: 1.05, xmin: 0, xmax: 2, marker: e, legend: "bottom-left" });
  }

  onInput([net, layer, eps], draw);
  onRedraw(draw);
  draw();
}
