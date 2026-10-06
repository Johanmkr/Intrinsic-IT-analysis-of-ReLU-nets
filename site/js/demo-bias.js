// Demo 3: the plug-in bias as ρ = |Ω_D| / N grows. Simulated routing with a
// known true MI, compared with the paper's 138 network cells.

import { chart, css } from "./plot.js";
import { routingMI } from "./mi.js";
import { $, bits, onInput, onRedraw, stats } from "./ui.js";

const C = 4;                                        // classes in the simulation
const N_GRID = Array.from({ length: 28 }, (_, i) => Math.round(10 ** (1.3 + (i * 3.2) / 27)));
const logSlider = (el, lo, hi) => Math.round(10 ** (lo + ((hi - lo) * el.value) / 1000));

function rng(seed) {                                // mulberry32, reproducible draws
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** K equally likely regions; region ω has home class ω mod C, kept with prob. q. */
function trueMI(K, q) {
  const pHome = q + (1 - q) / C, pOther = (1 - q) / C;
  const pY = new Array(C).fill(0);
  for (let w = 0; w < K; w++) for (let c = 0; c < C; c++) pY[c] += (c === w % C ? pHome : pOther) / K;
  const H = (p) => -p.filter((v) => v > 0).reduce((s, v) => s + v * Math.log2(v), 0);
  const hY = H(pY);
  return { mi: hY - H([pHome, ...new Array(C - 1).fill(pOther)]), hY };
}

function simulate(K, q, N, seed) {
  const r = rng(seed);
  const ids = new Int32Array(N), y = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const w = Math.floor(r() * K);
    ids[i] = w;
    y[i] = r() < q ? w % C : Math.floor(r() * C);
  }
  return routingMI(ids, y, N);
}

export function initBias(root, data) {
  const k = $("[name=k]", root), sig = $("[name=signal]", root), n = $("[name=n]", root);
  const kOut = $("output[for=d3-k]", root), sigOut = $("output[for=d3-signal]", root), nOut = $("output[for=d3-n]", root);
  const plot = $("canvas.chart", root), dl = $("dl", root);
  const cells = data.biasCells;

  function draw() {
    const K = logSlider(k, 0.5, 4), q = +sig.value / 100, N = logSlider(n, 1.3, 4.5);
    kOut.value = K; sigOut.value = q.toFixed(2); nOut.value = N;
    const truth = trueMI(K, q);
    const sweep = N_GRID.map((m, i) => {
      const a = simulate(K, q, m, 1000 + i), b = simulate(K, q, m, 2000 + i);
      return { rho: (a.regions / m + b.regions / m) / 2, plugIn: (a.plugIn + b.plugIn) / 2, mm: (a.millerMadow + b.millerMadow) / 2 };
    });
    const now = simulate(K, q, N, 7);
    stats(dl, [
      ["true I(Y; Ω)", bits(truth.mi)],
      ["ρ = |Ω<sub>D</sub>| / N", (now.regions / N).toFixed(3)],
      ["plug-in Î", bits(now.plugIn)],
      ["Miller–Madow Ĩ", bits(now.millerMadow)],
      ["H(Y)", bits(truth.hY)],
    ]);
    chart(plot, [
      { x: cells.map((c) => c.rho), y: cells.map((c) => c.plug_in / c.H_Y), color: css("--accent"), dots: true, label: "networks: plug-in" },
      { x: cells.map((c) => c.rho), y: cells.map((c) => c.miller_madow / c.H_Y), color: css("--accent2"), dots: true, hollow: true, label: "networks: Miller–Madow" },
      { x: sweep.map((s) => s.rho), y: sweep.map((s) => s.plugIn / truth.hY), color: css("--accent"), label: "simulation: plug-in" },
      { x: sweep.map((s) => s.rho), y: sweep.map((s) => s.mm / truth.hY), color: css("--accent2"), dash: true, label: "simulation: Miller–Madow" },
    ], { xlabel: "ρ = |Ω_D| / N", ylabel: "estimate / H(Y)", xlog: true, xmin: 1e-3, xmax: 1, ymin: -0.6, ymax: 1.15,
         marker: Math.max(1e-3, now.regions / N), hlines: [{ y: truth.mi / truth.hY, label: "true MI (simulation)" }, { y: 1, label: "H(Y)" }], legend: "bottom-left" });
  }

  onInput([k, sig, n], draw);
  onRedraw(draw);
  draw();
}
