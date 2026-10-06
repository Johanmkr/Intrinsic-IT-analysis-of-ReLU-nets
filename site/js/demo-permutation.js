// Demo 4: a network trained on the true labels next to one trained on randomly
// permuted labels. Both are scored against the true labels of the held-out points.

import { drawPartition, gridPoints } from "./plot.js";
import { evaluate, quotient } from "./view.js";
import { $, bits, onInput, onRedraw, pct, stats } from "./ui.js";

const G = 220;
const EPS = 0.3;

export function initPermutation(root, data) {
  const arch = $("[name=arch]", root), layer = $("[name=layer]", root), showPoints = $("[name=points]", root);
  const grid = gridPoints(G);

  function panel(key, side) {
    const v = evaluate(data.networks[key].weights["150"], +layer.value, grid, G, data.points, `${key}|150`);
    const q = quotient(v, EPS, data.points);
    drawPartition($(`canvas.${side}`, root), G, v.cell, { faint: v.gridRegions, ...(showPoints.checked && { points: data.points.X, labels: data.points.y }) });
    stats($(`dl.${side}`, root), [
      ["test accuracy", pct(v.accuracy)],
      ["|Ω<sub>D</sub>|", v.onData.count],
      ["ρ", v.rho.toFixed(3)],
      ["Î(Y; Ω<sub>D</sub>)", bits(v.mi.plugIn)],
      [`Î<sub>func</sub> (ε = ${EPS})`, bits(q.mi.plugIn)],
      ["H(Y)", bits(v.mi.hY)],
    ]);
  }

  function draw() {
    panel(`c${arch.value}`, "clean");
    panel(`p${arch.value}`, "permuted");
  }

  onInput([arch, layer, showPoints], draw);
  onRedraw(draw);
  draw();
}
