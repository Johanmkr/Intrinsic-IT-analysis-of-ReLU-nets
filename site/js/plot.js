// Canvas drawing: partition maps of the input square and small line/scatter charts.
// Colours come from CSS custom properties, so light and dark mode both work.

export const EXTENT = 1.05; // the maps show [-EXTENT, EXTENT]^2
export const CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#8C6D1F"];

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const isDark = () => matchMedia("(prefers-color-scheme: dark)").matches;

/** Pixel grid over the square: Float32Array of G*G points, row-major from the top. */
export function gridPoints(G) {
  const X = new Float32Array(G * G * 2);
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const t = 2 * (r * G + c);
      X[t] = -EXTENT + (2 * EXTENT * (c + 0.5)) / G;
      X[t + 1] = EXTENT - (2 * EXTENT * (r + 0.5)) / G;
    }
  }
  return X;
}

/** A distinct, theme-aware colour for region/class number k ([r, g, b]). */
export function regionColor(k) {
  const h = (k * 137.508) % 360;
  const [s, l] = isDark() ? [0.45, 0.32 + 0.1 * ((k * 7) % 3) / 2] : [0.6, 0.78 - 0.08 * ((k * 7) % 3) / 2];
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const kk = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(kk - 3, 9 - kk, 1))));
  };
  return [f(0), f(8), f(4)];
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/** Partition map: cell[i] is the colour index of pixel i (−1 = no data, grey);
 *  strong boundaries where neighbouring pixels differ in `edge`, light ones
 *  where they differ in `faint`. */
export function drawPartition(canvas, G, cell, { edge = null, faint = null, points = null, labels = null } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const img = new ImageData(G, G);
  const grey = isDark() ? [58, 60, 66] : [232, 232, 236];
  const line = isDark() ? [235, 235, 240] : [25, 25, 35];
  const mix = (rgb, to, t) => rgb.map((v, k) => Math.round(v + (to[k] - v) * t));
  for (let i = 0; i < G * G; i++) {
    const r = Math.floor(i / G), c = i % G;
    let rgb = cell[i] < 0 ? grey : regionColor(cell[i]);
    const crosses = (a) => a && ((c + 1 < G && a[i + 1] !== a[i]) || (r + 1 < G && a[i + G] !== a[i]));
    if (crosses(edge)) rgb = mix(rgb, line, 0.75);
    else if (crosses(faint)) rgb = mix(rgb, line, 0.25);
    img.data.set([rgb[0], rgb[1], rgb[2], 255], 4 * i);
  }
  const off = new OffscreenCanvas(G, G);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w, h);
  if (points) {
    const N = labels.length;
    for (let i = 0; i < N; i++) {
      const px = ((points[2 * i] + EXTENT) / (2 * EXTENT)) * w;
      const py = ((EXTENT - points[2 * i + 1]) / (2 * EXTENT)) * h;
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, 2 * Math.PI);
      ctx.fillStyle = CLASS_COLORS[labels[i] % CLASS_COLORS.length];
      ctx.fill();
    }
  }
}

/** Line/scatter chart. series: [{x, y, color, label, dash, dots, hollow}]. */
export function chart(canvas, series, { xlabel = "", ylabel = "", xlog = false, ymin = null, ymax = null, xmin = null, xmax = null, marker = null, hlines = [], legend = "top-left" } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const fg = css("--fg"), muted = css("--muted"), grid = css("--grid");
  const m = { l: 46, r: 10, t: 10, b: 36 };
  const xs = series.flatMap((s) => s.x).filter(Number.isFinite);
  const ys = series.flatMap((s) => s.y).filter(Number.isFinite).concat(hlines.map((l) => l.y));
  const tx = xlog ? Math.log10 : (v) => v;
  let x0 = xmin ?? Math.min(...xs), x1 = xmax ?? Math.max(...xs);
  let y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const X = (v) => m.l + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0))) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - y0) / (y1 - y0)) * (h - m.t - m.b);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif";
  ctx.strokeStyle = grid; ctx.fillStyle = muted; ctx.lineWidth = 1;
  // axes ticks
  const yt = niceTicks(y0, y1, 5);
  yt.forEach((v) => {
    ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(w - m.r, Y(v)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(v), m.l - 5, Y(v) + 4);
  });
  const xt = xlog ? logTicks(x0, x1) : niceTicks(x0, x1, 6);
  xt.forEach((v) => {
    ctx.beginPath(); ctx.moveTo(X(v), m.t); ctx.lineTo(X(v), h - m.b); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(fmt(v), X(v), h - m.b + 14);
  });
  ctx.fillStyle = fg; ctx.textAlign = "center";
  ctx.fillText(xlabel, (m.l + w - m.r) / 2, h - 4);
  ctx.save(); ctx.translate(11, (m.t + h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore();
  hlines.forEach(({ y, label, color }) => {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = color || muted;
    ctx.beginPath(); ctx.moveTo(m.l, Y(y)); ctx.lineTo(w - m.r, Y(y)); ctx.stroke(); ctx.setLineDash([]);
    if (label) { ctx.fillStyle = color || muted; ctx.textAlign = "right"; ctx.fillText(label, w - m.r - 2, Y(y) - 4); }
  });
  if (marker !== null) {
    ctx.strokeStyle = fg; ctx.globalAlpha = 0.35;
    ctx.beginPath(); ctx.moveTo(X(marker), m.t); ctx.lineTo(X(marker), h - m.b); ctx.stroke(); ctx.globalAlpha = 1;
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b); ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 2;
    if (s.dots) {
      s.x.forEach((xv, i) => {
        if (!Number.isFinite(s.y[i])) return;
        ctx.beginPath(); ctx.arc(X(xv), Y(s.y[i]), 2.6, 0, 2 * Math.PI);
        ctx.globalAlpha = 0.55; s.hollow ? ctx.stroke() : ctx.fill(); ctx.globalAlpha = 1;
      });
      continue;
    }
    ctx.setLineDash(s.dash ? [6, 4] : []);
    ctx.beginPath();
    s.x.forEach((xv, i) => (i ? ctx.lineTo(X(xv), Y(s.y[i])) : ctx.moveTo(X(xv), Y(s.y[i]))));
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.restore();
  const named = series.filter((s) => s.label);
  if (!named.length) return;
  const lw = Math.max(...named.map((s) => ctx.measureText(s.label).width)) + 36, lh = 16;
  const lx = legend.endsWith("right") ? w - m.r - lw - 4 : m.l + 6;
  const ly = legend.startsWith("bottom") ? h - m.b - named.length * lh - 8 : m.t + 4;
  ctx.fillStyle = css("--card"); ctx.globalAlpha = 0.85;
  ctx.fillRect(lx, ly, lw, named.length * lh + 4); ctx.globalAlpha = 1;
  named.forEach((s, k) => {
    const yy = ly + 10 + k * lh;
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 2;
    if (s.dots) { ctx.beginPath(); ctx.arc(lx + 13, yy, 3, 0, 2 * Math.PI); s.hollow ? ctx.stroke() : ctx.fill(); }
    else { ctx.setLineDash(s.dash ? [5, 3] : []); ctx.beginPath(); ctx.moveTo(lx + 4, yy); ctx.lineTo(lx + 22, yy); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = css("--fg"); ctx.textAlign = "left"; ctx.fillText(s.label, lx + 28, yy + 4);
  });
}

function niceTicks(a, b, n) {
  const step = 10 ** Math.floor(Math.log10((b - a) / n));
  const err = ((b - a) / n) / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(a / s) * s; v <= b + 1e-9; v += s) out.push(+v.toFixed(10));
  return out;
}

function logTicks(a, b) {
  const out = [];
  for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) if (10 ** e >= a * 0.999 && 10 ** e <= b * 1.001) out.push(10 ** e);
  return out;
}

const fmt = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : +v.toPrecision(3)).toString();
