---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 28px;
    background: #ffffff;
    color: #1a1a2e;
    padding: 48px 60px;
  }
  h1 {
    font-size: 2.0em;
    color: #0f3460;
    border-bottom: 3px solid #e94560;
    padding-bottom: 10px;
    margin-bottom: 0.4em;
  }
  h2 {
    font-size: 1.5em;
    color: #16213e;
    margin-bottom: 0.3em;
  }
  h3 {
    font-size: 1.1em;
    color: #e94560;
    margin-bottom: 0.2em;
  }
  strong { color: #0f3460; }
  em { color: #555; }
  code {
    background: #f0f4ff;
    color: #c7254e;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.88em;
  }
  pre code {
    background: #1e1e2e;
    color: #cdd6f4;
    font-size: 0.78em;
    padding: 14px 18px;
    border-radius: 8px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
  }
  th {
    background: #0f3460;
    color: white;
    padding: 8px 12px;
  }
  td { padding: 7px 12px; border-bottom: 1px solid #dde; }
  tr:nth-child(even) td { background: #f7f9ff; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
  .box {
    background: #f0f4ff;
    border-left: 4px solid #0f3460;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .box-red {
    background: #fff0f2;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 14px 18px;
  }
  .label-sup {
    display: inline-block;
    background: #0f3460;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8em;
  }
  .label-unsup {
    display: inline-block;
    background: #e94560;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8em;
  }
  .arch-diagram {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    font-size: 0.78em;
    margin: 18px 0 10px 0;
  }
  .arch-input {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
  }
  .arch-node {
    background: #0f3460;
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
    white-space: nowrap;
  }
  .arch-node-gray {
    background: #546e7a;
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
    white-space: nowrap;
  }
  .arch-node-red {
    background: #e94560;
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
    white-space: nowrap;
  }
  .arch-arrow {
    font-size: 1.4em;
    color: #888;
    padding: 0 4px;
    line-height: 1;
  }
  .arch-skip {
    font-size: 0.72em;
    color: #e94560;
    text-align: center;
    margin-top: 2px;
  }
  .data-diagram {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 12px 0;
  }
  .data-box-sup {
    background: #eef2ff;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82em;
  }
  .data-box-unsup {
    background: #fff0f2;
    border: 2px solid #e94560;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82em;
  }
  .data-label {
    font-weight: bold;
    font-size: 0.9em;
    margin-bottom: 6px;
  }
  section.title-slide {
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: white;
  }
  section.title-slide h1 {
    color: white;
    border-bottom-color: #e94560;
    font-size: 1.9em;
  }
  section.title-slide h2 { color: #a0b4d0; font-size: 1.0em; }
  section.title-slide p { color: #c8d8e8; }
---

<!-- _class: title-slide -->

# Semi-Supervised 3D Shape Reconstruction with DeepSDF

## AI6131 · NTU MSAI · April 2026

<br>

*"Reducing annotation requirements for implicit neural shape learning"*

---

<!-- Slide 2 -->

# Research Question

<div class="box">

**Can we train a DeepSDF auto-decoder with fewer labelled SDF samples while maintaining reconstruction quality?**

</div>

### Motivation

- Generating SDF labels requires **dense normal-offset sampling** — expensive per shape
- Unlabelled points (uniform sphere probes) need no annotation, yet constrain the field via $\mathcal{L}_\text{eik}$
- Reducing the labelled fraction enables scaling to larger shape collections

### Variables under study

| Variable | Values tested |
|---|---|
| Supervision ratio | 100% → 50% → 10% → 5% |
| Eikonal regularization | On / Off |
| Positional encoding | On / Off (Fourier, details in experiment table) |

---

<!-- Slide 3 -->

# Background: DeepSDF Auto-Decoder

### Core idea

Each shape $i$ gets a latent code $\mathbf{z}_i \in \mathbb{R}^{256}$.  
A shared decoder $f_\theta(\mathbf{z}_i,\, \mathbf{x})$ maps **(latent, 3D point) → SDF value**.

<div class="arch-diagram">
  <div class="arch-input">
    <div class="arch-node">z<sub>i</sub> ∈ ℝ<sup>256</sup></div>
    <div class="arch-node-gray">x ∈ ℝ<sup>3</sup> <span style="font-size:0.8em;opacity:0.75">(or γ(x) ∈ ℝ<sup>6L</sup> with PE)</span></div>
  </div>
  <div class="arch-arrow">→</div>
  <div>
    <div class="arch-node">concat → Layers 1–4<br><span style="font-size:0.85em;opacity:0.8">ReLU, 512-d</span></div>
    <div class="arch-skip">⊕ skip (concat input)</div>
  </div>
  <div class="arch-arrow">→</div>
  <div>
    <div class="arch-node">Layers 5–7<br><span style="font-size:0.85em;opacity:0.8">ReLU, 512-d</span></div>
    <div class="arch-skip">layer 8: linear, no activation</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-node-red">f<sub>θ</sub>(z<sub>i</sub>, x)<br><span style="font-size:0.85em;opacity:0.9">SDF ∈ ℝ</span></div>
</div>

<div class="columns">
<div>

- **8-layer MLP**, hidden dim 512, skip at layer 4
- No output activation — raw SDF value
- Last-layer sphere-biased init: small weight std + bias = 0.1; interior layers use Xavier

</div>
<div>

### Training objective

$$\min_{\theta,\, \{\mathbf{z}_i\}} \sum_i \mathcal{L}_i(\theta, \mathbf{z}_i)$$

- **No encoder** — auto-decoder framework
- $\theta$ and all $\{\mathbf{z}_i\}$ optimized **jointly** with Adam
- This project: **train-set reconstruction only**  
  *(test-time latent opt. out of scope)*

</div>
</div>

---

<!-- Slide 4 -->

# Semi-Supervised Extension

### Semi-supervised objective

$$\mathcal{L} = \mathcal{L}_\text{sdf} + \lambda(t) \cdot \mathcal{L}_\text{eik}$$

*($\lambda_z \cdot \mathcal{L}_z$ latent regularization applied in all runs — standard DeepSDF component)*

<div class="data-diagram">
<div class="data-box-sup">
<div class="data-label" style="color:#0f3460">Supervised points — contribute to L_sdf + L_eik</div>

Surface samples **offset along normals** at ε ∈ {0.005, 0.01, 0.05},  
multiplier $j \in \{-2,-1,+1,+2\}$.  
SDF label: $s_j = j \times \varepsilon$ — **signed** approximate offset, not ray-cast GT.

$$\mathcal{L}_\text{sdf} = \frac{1}{N_s}\sum_{j \in S} \bigl|f_\theta(\mathbf{z}, \mathbf{x}_j) - s_j\bigr|$$

</div>
<div class="data-box-unsup">
<div class="data-label" style="color:#e94560">Unsupervised points — contribute to L_eik only</div>

Sampled **uniformly in the unit sphere** (rejection sampling).  
No SDF label — constrain the field geometry via the Eikonal term.

$$\mathcal{L}_\text{eik} = \frac{1}{N}\sum_j \bigl(\|\nabla_\mathbf{x} f\|_2 - 1\bigr)^2$$

</div>
</div>

### Eikonal warmup — $\lambda(t) = \lambda_\text{eik} \cdot \min\!\bigl(1,\, t / t_\text{warmup}\bigr)$

| Supervision | Warmup | Rationale |
|---|---|---|
| 100% or 50% | 100 ep | Enough supervised signal from start |
| 10% | 150 ep | Prevent L_eik dominating sparse labels |
| 5% | 200 ep | Maximum warmup for minimal supervision |
