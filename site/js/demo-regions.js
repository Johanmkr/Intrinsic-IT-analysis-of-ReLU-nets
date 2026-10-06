// Demo 1: linear regions and routing information, across layers and training.

import { chart, css, drawPartition, gridPoints } from "./plot.js";
import { evaluate, evaluatePoints } from "./view.js";
import { $, bits, onInput, onRedraw, pct, stats } from "./ui.js";

const G = 220;

export function initRegions(root, data) {
  const net = $("[name=net]", root), layer = $("[name=layer]", root), epoch = $("[name=epoch]", root);
  const showPoints = $("[name=points]", root), epochOut = $("output[for=d1-epoch]", root);
  const map = $("canvas.map", root), plot = $("canvas.chart", root), dl = $("dl", root);
  const grid = gridPoints(G);

  function epochs() { return Object.keys(data.networks[net.value].weights).map(Number); }

  function draw() {
    const eps = epochs();
    epoch.max = eps.length - 1;
    const e = eps[+epoch.value];
    const l = +layer.value;
    const nw = data.networks[net.value];
    const v = evaluate(nw.weights[e], l, grid, G, data.points, `${net.value}|${e}`);
    epochOut.value = e;
    drawPartition(map, G, v.cell, { faint: v.gridRegions, ...(showPoints.checked && { points: data.points.X, labels: data.points.y }) });
    stats(dl, [
      ["|Ω<sub>D</sub>|", v.onData.count, "data-supported regions"],
      ["regions in view", v.regionsInView],
      ["ρ = |Ω<sub>D</sub>| / N", v.rho.toFixed(3)],
      ["Î(Y; Ω<sub>D</sub>)", bits(v.mi.plugIn)],
      ["H(Y)", bits(v.mi.hY)],
      ["test accuracy", pct(v.accuracy)],
    ]);
    const curve = eps.map((ep) => evaluatePoints(nw.weights[ep], l, data.points, `${net.value}|${ep}`));
    chart(plot, [
      { x: eps, y: curve.map((c) => c.mi.plugIn), color: css("--accent"), label: "plug-in Î" },
      { x: eps, y: curve.map((c) => c.mi.millerMadow), color: css("--accent2"), dash: true, label: "Miller–Madow Ĩ" },
    ], { xlabel: "training epoch", ylabel: "bits", ymin: Math.min(0, ...curve.map((c) => c.mi.millerMadow)),
         ymax: Math.ceil(v.mi.hY * 10) / 10 + 0.1, marker: e, hlines: [{ y: v.mi.hY, label: "H(Y)" }], legend: "bottom-right" });
  }

  onInput([net, layer, epoch, showPoints], draw);
  onRedraw(draw);
  draw();
}
