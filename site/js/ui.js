// Small UI helpers shared by the demos.

export const $ = (sel, root = document) => root.querySelector(sel);

/** Call fn on input events, at most once per animation frame. */
export function onInput(elements, fn) {
  let queued = false;
  const run = () => { queued = false; fn(); };
  for (const el of elements) el.addEventListener("input", () => { if (!queued) { queued = true; requestAnimationFrame(run); } });
}

/** Fill a <dl> with [label, value, title?] rows. */
export function stats(dl, rows) {
  dl.replaceChildren(...rows.flatMap(([k, v, title]) => {
    const dt = document.createElement("dt"); dt.innerHTML = k; if (title) dt.title = title;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
}

export const bits = (v) => `${v.toFixed(3)} bits`;
export const pct = (v) => `${(100 * v).toFixed(1)} %`;

/** Redraw on resize and on light/dark switches. */
export function onRedraw(fn) {
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(fn, 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
}
