// Loads the demo data and starts each demo when it scrolls into view.

import { initRegions } from "./demo-regions.js";
import { initQuotient } from "./demo-quotient.js";
import { initBias } from "./demo-bias.js";
import { initPermutation } from "./demo-permutation.js";

const DEMOS = { regions: initRegions, quotient: initQuotient, bias: initBias, permutation: initPermutation };

async function load(name) {
  const res = await fetch(`data/${name}.json`);
  if (!res.ok) throw new Error(`data/${name}.json: ${res.status}`);
  return res.json();
}

async function main() {
  const [points, networks, biasCells] = await Promise.all(["points", "networks", "bias_cells"].map(load));
  const data = {
    points: { X: Float32Array.from(points.X.flat()), y: Int32Array.from(points.y), N: points.N },
    networks,
    biasCells,
  };
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      const root = entry.target;
      try {
        DEMOS[root.dataset.demo](root, data);
        root.classList.add("ready");
      } catch (err) {
        root.querySelector(".status").textContent = `This demo failed to start: ${err.message}`;
        console.error(err);
      }
    }
  }, { rootMargin: "200px" });
  document.querySelectorAll("[data-demo]").forEach((el) => observer.observe(el));
}

document.querySelectorAll("button.copy").forEach((btn) => btn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(document.getElementById(btn.dataset.target).textContent);
  btn.textContent = "Copied";
  setTimeout(() => (btn.textContent = "Copy"), 1500);
}));

main().catch((err) => {
  document.querySelectorAll("[data-demo] .status").forEach((s) => (s.textContent = `Could not load the demo data: ${err.message}`));
  console.error(err);
});
