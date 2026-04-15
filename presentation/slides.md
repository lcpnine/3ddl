---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 24px;
    background: #ffffff;
    color: #1a1a2e;
    padding: 48px 60px;
  }
  h1 {
    font-size: 1.8em;
    color: #0f3460;
    border-bottom: 3px solid #e94560;
    padding-bottom: 10px;
    margin-bottom: 0.4em;
  }
  h2 {
    font-size: 1.35em;
    color: #16213e;
    margin-bottom: 0.3em;
  }
  h3 {
    font-size: 1.0em;
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
    font-size: 1.7em;
  }
  section.title-slide h2 { color: #a0b4d0; font-size: 1.0em; }
  section.title-slide p { color: #c8d8e8; }
  section.small-text {
    font-size: 22px;
  }
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

---

# Research Variables

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

---

# Semi-Supervised Extension

### Eikonal warmup — $\lambda(t) = \lambda_\text{eik} \cdot \min\!\bigl(1,\, t / t_\text{warmup}\bigr)$

| Supervision | Warmup | Rationale |
|---|---|---|
| 100% or 50% | 100 ep | Enough supervised signal from start |
| 10% | 150 ep | Prevent L_eik dominating sparse labels |
| 5% | 200 ep | Maximum warmup for minimal supervision |

---

<!-- Slide 5 -->

# Experiment Design

Dataset: ShapeNet — 300 shapes (airplane / chair / table), 75% used for training (train_split=0.75) · 250K unsupervised + up to 250K supervised points per shape (supervised count = 250K × ratio) · ~6h per run on TC2

| Exp | Supervision | Eikonal | Positional Enc. | Seeds |
|-----|------------|---------|-----------------|-------|
| EXP-01 | 100% | No | No | 1 |
| EXP-02 | 100% | Yes | No | 1 |
| EXP-03 | 50% | Yes | No | 1 |
| EXP-04 | 10% | Yes | No | 3 |
| EXP-05 | 5% | Yes | No | 1 |
| EXP-06 | 10% | Yes | Yes (L=6) | 3 |

---

# Experiment Design

Dataset: ShapeNet — 300 shapes (airplane / chair / table), 75% used for training (train_split=0.75) · 250K unsupervised + up to 250K supervised points per shape (supervised count = 250K × ratio) · ~6h per run on TC2

| Exp | Supervision | Eikonal | Positional Enc. | Seeds |
|-----|------------|---------|-----------------|-------|
| EXP-07 | 5% | Yes | Yes (L=6) | 1 |
| EXP-08 | 10% | Yes | Yes (L=6) + $\mathcal{L}_\text{2nd}$ | 1 |
| EXP-09 | 100% | Yes | Yes (L=6) | 1 |
| EXP-10 | 100% | Yes | Yes (L=4) | 1 |
| EXP-11 | 10% | Yes | Yes (L=4) | 1 |
| EXP-12 | 5% | Yes | Yes (L=4) | 1 |

<div class="box">

**Primary comparisons**: EXP-01 → EXP-02 (Eikonal effect) · EXP-02 → EXP-04 → EXP-05 (label reduction) · EXP-04 vs EXP-06 (PE effect)

</div>

---

<!-- Slide 6 -->

# Preliminary Results

<div class="box-red">

**Preliminary** — evaluation bugs discovered after these runs; see following slides. Non-PE group: original evaluator (MC res=128). PE group: rerun with fixed evaluator (MC res=128). Cross-group CD/NC comparison is not valid.

</div>

### No positional encoding — original evaluator (MC res=128)

| Exp | Supervision | CD mean | NC mean | Seeds |
|-----|------------|---------|---------|-------|
| EXP-01 | 100% | 0.0593 | 0.5522 | 1 |
| EXP-02 | 100% + Eik | 0.0543 | 0.5920 | 1 |
| EXP-03 | 50% + Eik | 0.0534 | 0.5924 | 1 |
| EXP-04 | 10% + Eik | 0.0496 | 0.5987 | 3 |
| EXP-05 | 5% + Eik | 0.0509 | 0.5766 | 1 |

---

# Preliminary Results

### With positional encoding — rerun with fixed evaluator (MC res=128)

| Exp | Supervision | PE | CD mean | NC mean | Seeds |
|-----|------------|-----|---------|---------|-------|
| EXP-06 | 10% + Eik | L=6 | 0.1399 | 0.5080 | 3 |
| EXP-07 | 5% + Eik | L=6 | 0.1448 | 0.5074 | 1 |
| EXP-08 | 10% + Eik | L=6+$\mathcal{L}_\text{2nd}$ | 0.1443 | 0.5053 | 1 |
| EXP-09 | 100% + Eik | L=6 | 0.1450 | 0.5031 | 1 |
| EXP-10 | 100% + Eik | L=4 | 0.1401 | 0.5077 | 1 |
| EXP-11 | 10% + Eik | L=4 | 0.1427 | 0.5071 | 1 |
| EXP-12 | 5% + Eik | L=4 | 0.1400 | 0.5073 | 1 |

*Lower CD is better · Higher NC is better · Non-PE group: variable MC success (EXP-02: 261/300, EXP-03: 259/300, EXP-05: 295/300; EXP-04 seed42: 263/300) · EXP-04 per-seed CD means: {0.0609, 0.0443, 0.0436}; EXP-06 per-seed CD means: {0.1443, 0.1375, 0.1380}*

---

<!-- Slide 7 -->

# Evaluation Pipeline Review

<div class="box">

Before drawing conclusions from the results, I audited the evaluation pipeline. The main issue was incorrect shape→latent-index assignment: the evaluator used sorted alphabetical shape order, but training assigns latent indices in training-loop order, so each shape was decoded with the wrong latent code. Several secondary issues were also identified.

</div>

- Evaluation is now restricted to training shapes with the correct stored latent indices
- Secondary fixes applied: checkpoint selection criterion, train/val split ordering
- PE runs (EXP-06–12) have been re-evaluated with the corrected evaluator; non-PE results use the original

---

<!-- _class: small-text -->
<!-- Slide 8 -->

# Key Evaluation Issue: Latent Code Assignment

<div class="box">

DeepSDF is an auto-decoder: latent codes are learned one per training shape, indexed by training order. The evaluator must use the same shape→latent-index mapping that training used. If it uses a different ordering, it retrieves the wrong latent for each shape and the resulting CD/NC values do not reflect reconstruction quality.

</div>

<div class="columns">
<div>

### Before

- Evaluator iterated shapes in sorted alphabetical order
- Latent index assigned by sorted position, not training position
- Wrong latent retrieved for each shape → CD/NC reflects ordering mismatch, not shape quality

</div>
<div>

### After

- Evaluator loads `train_shapes.json` to reconstruct exact training-order index assignment
- Val shapes have no stored latent; TTO used if val split is evaluated
- Reported CD/NC reflects actual reconstruction of training shapes

</div>
</div>

*Reported metrics are train-set reconstruction quality — held-out generalization is not evaluated here*

---

<!-- Slide 9 -->

# Current Status and Next Step

<div class="columns">
<div>

### Established

- The 12-experiment design remains unchanged
- Checkpoints for the planned experiment matrix already exist
- The quantitative evaluation has been corrected for re-validation

</div>
<div>

### Not yet claimable

- Quantitative comparisons across experiments are still being re-validated
- No comparative conclusion is claimed until re-evaluation is complete

</div>
</div>

<div class="box">

**Next step**: Re-validate the existing checkpoints under the corrected evaluation, then report the quantitative comparison across experiments.

</div>
