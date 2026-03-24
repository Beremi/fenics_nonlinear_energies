# pLaplace_u3 Thesis Replications

Source of the original algorithms and published benchmark values:

- Michaela Bailová, *Variational methods for solving engineering problems*, PhD Thesis, Ostrava, 2023
- local source PDF: `BAI0012_FEI_P1807_1103V036_2023.pdf`
- this merged page combines the thesis/runbook description and the current canonical replication packet

## Thesis Problem Statement And Functionals

The thesis studies the nonlinear Dirichlet $p$-Laplacian problem

$$
-\Delta_p u = u^3 \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial \Omega,
$$

with

$$
\Delta_p u := \operatorname{div}\!\left(\lvert \nabla u \rvert^{p-2}\nabla u\right), \qquad p \in \left(\frac{4}{3}, 4\right).
$$

The weak form is:

$$
\int_\Omega \lvert \nabla u \rvert^{p-2}\nabla u \cdot \nabla v\,dx = \int_\Omega u^3 v\,dx
\qquad \forall v \in W_0^{1,p}(\Omega).
$$

The energy functional is

$$
J(u) = \frac{1}{p}\int_\Omega \lvert \nabla u \rvert^p\,dx - \frac{1}{4}\int_\Omega u^4\,dx.
$$

The thesis also uses the scale-invariant quotient

$$
I(u) = \frac{\|u\|_{1,p,0}}{\|u\|_{L^4(\Omega)}}.
$$

For the ray methods, the positive ray maximiser is

$$
t^*(w) = \left(\frac{A(w)}{B(w)}\right)^{\frac{1}{4-p}},
\qquad
A(w) = \int_\Omega \lvert \nabla w \rvert^p\,dx,
\qquad
B(w) = \int_\Omega w^4\,dx.
$$

## Thesis Geometries, Discretisation, And Seeds

- primary geometry: $\Omega = [0,\pi] \times [0,\pi]$
- secondary geometry: $\Omega = ([0,\pi] \times [0,\pi]) \setminus ((\pi/4,3\pi/4) \times (\pi/4,3\pi/4))$
- boundary condition: homogeneous Dirichlet on the full boundary, including the inner hole boundary
- discretisation: structured continuous $P_1$ finite elements on uniform right-triangle meshes with $h = \pi / 2^L$
- principal square seed: $\sin(x)\sin(y)$
- square OA2 seeds: $\sin(x)\sin(y)$, $10\sin(2x)\sin(y)$, $10\sin(x)\sin(2y)$, $4(x-y)\sin(x)\sin(y)$
- square-hole OA2 seeds: $\sin(x)\sin(y)$, $4|\sin(x)\sin(2y)|$, $4(x-y)\sin(x)\sin(y)$, $|4\sin(3x)\sin(3y)|$

## Thesis Algorithms And Current Repo Implementation

- `MPA`: classical polygonal-chain mountain-pass method
- `RMPA`: ray mountain-pass method using the analytic ray projection $t^*(w)$
- `OA1`: first-order descent on $I(u)$ with halving acceptance
- `OA2`: first-order descent on $I(u)$ with a 1D minimisation step on $[0, \delta]$

The repository implementation for this thesis packet lives under `src/problems/plaplace_u3/thesis/` and is backed by the same structured thesis geometries, exact $P_1$ quartic integration, and the current canonical summary at `artifacts/raw_results/plaplace_u3_thesis_full/summary.json`.

The section commands below rematerialize the current canonical thesis packet into dedicated experiment folders quickly. For a raw solver recomputation of the same families, use `experiments/runners/run_plaplace_u3_thesis_suite.py --only-table ...` with the table keys shown in each section.

![Current square sample state](../assets/plaplace_u3_thesis/plaplace_u3_sample_state.png)

PDF version: [sample state](../assets/plaplace_u3_thesis/plaplace_u3_sample_state.pdf)

## Validation Metric And Replication Status

The thesis validates computed solutions against a separate finite-element reference solution using the discrete $|u-\bar u|_{1,p,0}$ seminorm. In this repository packet, the direct thesis quantities such as $J$, $I$, and iteration counts are compared against the published tables, while the error columns use the repo's proxy reference policy documented in the canonical thesis report.

- canonical summary: `artifacts/raw_results/plaplace_u3_thesis_full/summary.json`
- canonical thesis report: `artifacts/reports/plaplace_u3_thesis/README.md`
- packet note: Hybrid refresh on 2026-03-23: thesis metadata recomputed with corrected tables; square-hole OA2 seeds skew and abs_sine_3x3 rerun with the documented execution overrides. Remaining rows still come from the previously promoted full packet.
- primary assignment rows passing: `159` / `185`
- status counts: `{'maxit': 52, 'completed': 183, 'failed': 1}`

### Stage Map

| stage | brief | pass | fail | secondary | total |
| --- | --- | --- | --- | --- | --- |
| Optional 1D Harness | Section 18; cheap stopping and direction sanity check on (0, π). | 2 | 18 | 1 | 21 |
| Stage A | Section 13 Stage A; principal branch on the square with RMPA. | 59 | 1 | 0 | 60 |
| Stage B | Section 13 Stage B; cross-check the same branch with OA1. | 30 | 0 | 30 | 60 |
| Stage C | Section 13 Stage C; compare method behavior and iteration counts. | 59 | 24 | 0 | 83 |
| Stage D | Section 13 Stage D; multiple branches on the square with OA2. | 7 | 1 | 0 | 8 |
| Stage E | Section 13 Stage E; multiple branches on the square-with-hole with OA2. | 4 | 0 | 0 | 4 |

### Table Map

| method | thesis targets |
| --- | --- |
| MPA | Table 5.6, Table 5.7, Table 5.12 |
| RMPA | Table 5.8, Table 5.9, Table 5.12, Table 5.13 |
| OA1 | Table 5.10, Table 5.11, Table 5.12, Table 5.13, Table 5.14 |
| OA2 | Table 5.14, Figure 5.13 |

### Current Table Coverage

| target | assignment section | pass | fail | secondary | total |
| --- | --- | --- | --- | --- | --- |
| table_5_2 | Section 18 / Table 5.2 | 1 | 9 | 0 | 10 |
| table_5_3 | Section 18 / Table 5.3 | 1 | 9 | 0 | 10 |
| table_5_2_drn_sanity | Section 18 / d^R_N sanity | 0 | 0 | 1 | 1 |
| table_5_8 | Section 14.1 / Table 5.8 | 29 | 1 | 0 | 30 |
| table_5_10 | Section 15.1 / Table 5.10 | 30 | 0 | 0 | 30 |
| table_5_9 | Section 14.2 / Table 5.9 | 30 | 0 | 0 | 30 |
| table_5_11 | Section 15.2 / Table 5.11 | 0 | 0 | 30 | 30 |
| table_5_6 | Section 16.1 / Table 5.6 | 13 | 11 | 0 | 24 |
| table_5_7 | Section 16.1 / Table 5.7 | 19 | 8 | 0 | 27 |
| table_5_13 | Section 16.2 / Table 5.13 | 27 | 5 | 0 | 32 |
| table_5_14 | Section 17.1 / Table 5.14 | 7 | 1 | 0 | 8 |
| figure_5_13 | Section 17.2 / Figure 5.13 | 4 | 0 | 0 | 4 |

## 1D Direction Study

This section merges the thesis 1D harness description with the current repo results for the same nonlinearity on $(0,\pi)$.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table table_5_2 --only-table table_5_3 --only-table table_5_2_drn_sanity \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/one_dimensional
```

| table | direction | p | thesis $J$ | repo $J$ | thesis error | repo error | repo iters | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| table_5_2 | d | 1.500 | <span style="color:#1d4ed8;"><em>0.7600</em></span> | <span style="color:#b91c1c;"><strong>0.9130</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0074</strong></span> | <span style="color:#b91c1c;"><strong>200</strong></span> | FAIL |
| table_5_2 | d | 1.667 | <span style="color:#1d4ed8;"><em>0.6200</em></span> | <span style="color:#b91c1c;"><strong>0.7559</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>105</strong></span> | FAIL |
| table_5_2 | d | 1.833 | <span style="color:#1d4ed8;"><em>0.5100</em></span> | <span style="color:#b91c1c;"><strong>0.6221</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>12</strong></span> | FAIL |
| table_5_2 | d | 2.000 | <span style="color:#1d4ed8;"><em>0.4100</em></span> | <span style="color:#b91c1c;"><strong>0.5082</strong></span> | <span style="color:#1d4ed8;"><em>0.0000</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | FAIL |
| table_5_2 | d | 2.167 | <span style="color:#1d4ed8;"><em>0.3300</em></span> | <span style="color:#b91c1c;"><strong>0.4109</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | FAIL |
| table_5_2 | d | 2.333 | <span style="color:#1d4ed8;"><em>0.2600</em></span> | <span style="color:#b91c1c;"><strong>0.3275</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0008</strong></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | FAIL |
| table_5_2 | d | 2.500 | <span style="color:#1d4ed8;"><em>0.1900</em></span> | <span style="color:#b91c1c;"><strong>0.2559</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0009</strong></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | FAIL |
| table_5_2 | d | 2.667 | <span style="color:#1d4ed8;"><em>0.1400</em></span> | <span style="color:#b91c1c;"><strong>0.1944</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0011</strong></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | FAIL |
| table_5_2 | d | 2.833 | <span style="color:#1d4ed8;"><em>0.1000</em></span> | <span style="color:#b91c1c;"><strong>0.1420</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0559</strong></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | FAIL |
| table_5_2 | d | 3.000 | <span style="color:#1d4ed8;"><em>0.1000</em></span> | <span style="color:#b91c1c;"><strong>0.0978</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0390</strong></span> | <span style="color:#b91c1c;"><strong>33</strong></span> | PASS |
| table_5_2_drn_sanity | d_rn | 2.000 | <span style="color:#1d4ed8;"><em>-</em></span> | <span style="color:#b91c1c;"><strong>0.5090</strong></span> | <span style="color:#1d4ed8;"><em>-</em></span> | <span style="color:#b91c1c;"><strong>-</strong></span> | <span style="color:#b91c1c;"><strong>200</strong></span> | secondary |
| table_5_3 | d_vh | 1.500 | <span style="color:#1d4ed8;"><em>0.7600</em></span> | <span style="color:#b91c1c;"><strong>0.9130</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>200</strong></span> | FAIL |
| table_5_3 | d_vh | 1.667 | <span style="color:#1d4ed8;"><em>0.6200</em></span> | <span style="color:#b91c1c;"><strong>0.7559</strong></span> | <span style="color:#1d4ed8;"><em>0.0000</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>40</strong></span> | FAIL |
| table_5_3 | d_vh | 1.833 | <span style="color:#1d4ed8;"><em>0.5100</em></span> | <span style="color:#b91c1c;"><strong>0.6221</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>11</strong></span> | FAIL |
| table_5_3 | d_vh | 2.000 | <span style="color:#1d4ed8;"><em>0.4100</em></span> | <span style="color:#b91c1c;"><strong>0.5082</strong></span> | <span style="color:#1d4ed8;"><em>0.0000</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | FAIL |
| table_5_3 | d_vh | 2.167 | <span style="color:#1d4ed8;"><em>0.3300</em></span> | <span style="color:#b91c1c;"><strong>0.4109</strong></span> | <span style="color:#1d4ed8;"><em>0.0001</em></span> | <span style="color:#b91c1c;"><strong>0.0007</strong></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | FAIL |
| table_5_3 | d_vh | 2.333 | <span style="color:#1d4ed8;"><em>0.2600</em></span> | <span style="color:#b91c1c;"><strong>0.3275</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0008</strong></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | FAIL |
| table_5_3 | d_vh | 2.500 | <span style="color:#1d4ed8;"><em>0.1900</em></span> | <span style="color:#b91c1c;"><strong>0.2559</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0009</strong></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | FAIL |
| table_5_3 | d_vh | 2.667 | <span style="color:#1d4ed8;"><em>0.1400</em></span> | <span style="color:#b91c1c;"><strong>0.1944</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0013</strong></span> | <span style="color:#b91c1c;"><strong>11</strong></span> | FAIL |
| table_5_3 | d_vh | 2.833 | <span style="color:#1d4ed8;"><em>0.1000</em></span> | <span style="color:#b91c1c;"><strong>0.1420</strong></span> | <span style="color:#1d4ed8;"><em>0.0013</em></span> | <span style="color:#b91c1c;"><strong>0.0014</strong></span> | <span style="color:#b91c1c;"><strong>21</strong></span> | FAIL |
| table_5_3 | d_vh | 3.000 | <span style="color:#1d4ed8;"><em>0.1000</em></span> | <span style="color:#b91c1c;"><strong>0.0978</strong></span> | <span style="color:#1d4ed8;"><em>0.0013</em></span> | <span style="color:#b91c1c;"><strong>0.0028</strong></span> | <span style="color:#b91c1c;"><strong>13</strong></span> | PASS |

## RMPA Square Principal-Branch Replication

The thesis uses the square benchmark as the main validation target for RMPA. The tables below merge the mesh-refinement and fixed-mesh tolerance studies.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table table_5_8 --only-table table_5_9 \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/rmpa_square
```

### Table 5.8 — refinement study

| $p$ | level | thesis $J$ | repo $J$ | thesis error | repo error | status |
| --- | --- | --- | --- | --- | --- | --- |
| 1.500 | 5 | <span style="color:#1d4ed8;"><em>4.9000</em></span> | <span style="color:#b91c1c;"><strong>4.9022</strong></span> | <span style="color:#1d4ed8;"><em>0.0139</em></span> | <span style="color:#b91c1c;"><strong>0.3293</strong></span> | PASS |
| 1.500 | 6 | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0035</em></span> | <span style="color:#b91c1c;"><strong>0.1634</strong></span> | PASS |
| 1.500 | 7 | <span style="color:#1d4ed8;"><em>4.4700</em></span> | <span style="color:#b91c1c;"><strong>4.8772</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0815</strong></span> | FAIL |
| 1.667 | 5 | <span style="color:#1d4ed8;"><em>4.4900</em></span> | <span style="color:#b91c1c;"><strong>4.4931</strong></span> | <span style="color:#1d4ed8;"><em>0.0083</em></span> | <span style="color:#b91c1c;"><strong>0.2463</strong></span> | PASS |
| 1.667 | 6 | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | PASS |
| 1.667 | 7 | <span style="color:#1d4ed8;"><em>4.4700</em></span> | <span style="color:#b91c1c;"><strong>4.4717</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0614</strong></span> | PASS |
| 1.833 | 5 | <span style="color:#1d4ed8;"><em>4.1400</em></span> | <span style="color:#b91c1c;"><strong>4.1362</strong></span> | <span style="color:#1d4ed8;"><em>0.0061</em></span> | <span style="color:#b91c1c;"><strong>0.2114</strong></span> | PASS |
| 1.833 | 6 | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | PASS |
| 1.833 | 7 | <span style="color:#1d4ed8;"><em>4.1100</em></span> | <span style="color:#b91c1c;"><strong>4.1146</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0527</strong></span> | PASS |
| 2.000 | 5 | <span style="color:#1d4ed8;"><em>3.8500</em></span> | <span style="color:#b91c1c;"><strong>3.8510</strong></span> | <span style="color:#1d4ed8;"><em>0.0049</em></span> | <span style="color:#b91c1c;"><strong>0.1932</strong></span> | PASS |
| 2.000 | 6 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | PASS |
| 2.000 | 7 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0482</strong></span> | PASS |
| 2.167 | 5 | <span style="color:#1d4ed8;"><em>3.6400</em></span> | <span style="color:#b91c1c;"><strong>3.6388</strong></span> | <span style="color:#1d4ed8;"><em>0.0041</em></span> | <span style="color:#b91c1c;"><strong>0.1837</strong></span> | PASS |
| 2.167 | 6 | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | PASS |
| 2.167 | 7 | <span style="color:#1d4ed8;"><em>3.6100</em></span> | <span style="color:#b91c1c;"><strong>3.6128</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0459</strong></span> | PASS |
| 2.333 | 5 | <span style="color:#1d4ed8;"><em>3.5000</em></span> | <span style="color:#b91c1c;"><strong>3.5023</strong></span> | <span style="color:#1d4ed8;"><em>0.0036</em></span> | <span style="color:#b91c1c;"><strong>0.1799</strong></span> | PASS |
| 2.333 | 6 | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | PASS |
| 2.333 | 7 | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4721</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0451</strong></span> | PASS |
| 2.500 | 5 | <span style="color:#1d4ed8;"><em>3.4500</em></span> | <span style="color:#b91c1c;"><strong>3.4515</strong></span> | <span style="color:#1d4ed8;"><em>0.0032</em></span> | <span style="color:#b91c1c;"><strong>0.1807</strong></span> | PASS |
| 2.500 | 6 | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | PASS |
| 2.500 | 7 | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4152</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0455</strong></span> | PASS |
| 2.667 | 5 | <span style="color:#1d4ed8;"><em>3.5100</em></span> | <span style="color:#b91c1c;"><strong>3.5115</strong></span> | <span style="color:#1d4ed8;"><em>0.0028</em></span> | <span style="color:#b91c1c;"><strong>0.1857</strong></span> | PASS |
| 2.667 | 6 | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| 2.667 | 7 | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4661</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0471</strong></span> | PASS |
| 2.833 | 5 | <span style="color:#1d4ed8;"><em>3.7400</em></span> | <span style="color:#b91c1c;"><strong>3.7384</strong></span> | <span style="color:#1d4ed8;"><em>0.0026</em></span> | <span style="color:#b91c1c;"><strong>0.1958</strong></span> | PASS |
| 2.833 | 6 | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | PASS |
| 2.833 | 7 | <span style="color:#1d4ed8;"><em>3.6800</em></span> | <span style="color:#b91c1c;"><strong>3.6780</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0503</strong></span> | PASS |
| 3.000 | 5 | <span style="color:#1d4ed8;"><em>4.2600</em></span> | <span style="color:#b91c1c;"><strong>4.2640</strong></span> | <span style="color:#1d4ed8;"><em>0.0023</em></span> | <span style="color:#b91c1c;"><strong>0.2127</strong></span> | PASS |
| 3.000 | 6 | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | PASS |
| 3.000 | 7 | <span style="color:#1d4ed8;"><em>4.1800</em></span> | <span style="color:#b91c1c;"><strong>4.1765</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0555</strong></span> | PASS |

### Table 5.9 — tolerance study

| $p$ | $\varepsilon$ | thesis $J$ | repo $J$ | thesis error | repo error | status |
| --- | --- | --- | --- | --- | --- | --- |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0035</em></span> | <span style="color:#b91c1c;"><strong>0.1634</strong></span> | PASS |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0035</em></span> | <span style="color:#b91c1c;"><strong>0.1634</strong></span> | PASS |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0033</em></span> | <span style="color:#b91c1c;"><strong>0.1638</strong></span> | PASS |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | PASS |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | PASS |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0020</em></span> | <span style="color:#b91c1c;"><strong>0.1231</strong></span> | PASS |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | PASS |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | PASS |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | PASS |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | PASS |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | PASS |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | PASS |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | PASS |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | PASS |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0011</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | PASS |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | PASS |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | PASS |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0011</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | PASS |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | PASS |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | PASS |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | PASS |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0023</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | PASS |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | PASS |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0019</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | PASS |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | PASS |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | PASS |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0013</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | PASS |

## OA1 Square Principal-Branch Replication

OA1 uses the scale-invariant functional $I(u)$. The thesis notes that Table 5.11 should be treated cautiously, so Table 5.10 remains the primary OA1 benchmark.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table table_5_10 --only-table table_5_11 \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/oa1_square
```

### Table 5.10 — refinement study

| $p$ | level | thesis $J$ | repo $J$ | thesis error | repo error | status |
| --- | --- | --- | --- | --- | --- | --- |
| 1.500 | 5 | <span style="color:#1d4ed8;"><em>4.9000</em></span> | <span style="color:#b91c1c;"><strong>4.9022</strong></span> | <span style="color:#1d4ed8;"><em>0.0136</em></span> | <span style="color:#b91c1c;"><strong>0.3293</strong></span> | PASS |
| 1.500 | 6 | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0033</em></span> | <span style="color:#b91c1c;"><strong>0.1634</strong></span> | PASS |
| 1.500 | 7 | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8772</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0815</strong></span> | PASS |
| 1.667 | 5 | <span style="color:#1d4ed8;"><em>4.4900</em></span> | <span style="color:#b91c1c;"><strong>4.4931</strong></span> | <span style="color:#1d4ed8;"><em>0.0083</em></span> | <span style="color:#b91c1c;"><strong>0.2463</strong></span> | PASS |
| 1.667 | 6 | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0020</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | PASS |
| 1.667 | 7 | <span style="color:#1d4ed8;"><em>4.4700</em></span> | <span style="color:#b91c1c;"><strong>4.4717</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0614</strong></span> | PASS |
| 1.833 | 5 | <span style="color:#1d4ed8;"><em>4.1400</em></span> | <span style="color:#b91c1c;"><strong>4.1362</strong></span> | <span style="color:#1d4ed8;"><em>0.0061</em></span> | <span style="color:#b91c1c;"><strong>0.2114</strong></span> | PASS |
| 1.833 | 6 | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | PASS |
| 1.833 | 7 | <span style="color:#1d4ed8;"><em>4.1100</em></span> | <span style="color:#b91c1c;"><strong>4.1146</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0527</strong></span> | PASS |
| 2.000 | 5 | <span style="color:#1d4ed8;"><em>3.8500</em></span> | <span style="color:#b91c1c;"><strong>3.8510</strong></span> | <span style="color:#1d4ed8;"><em>0.0049</em></span> | <span style="color:#b91c1c;"><strong>0.1932</strong></span> | PASS |
| 2.000 | 6 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | PASS |
| 2.000 | 7 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0482</strong></span> | PASS |
| 2.167 | 5 | <span style="color:#1d4ed8;"><em>3.6400</em></span> | <span style="color:#b91c1c;"><strong>3.6388</strong></span> | <span style="color:#1d4ed8;"><em>0.0041</em></span> | <span style="color:#b91c1c;"><strong>0.1837</strong></span> | PASS |
| 2.167 | 6 | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | PASS |
| 2.167 | 7 | <span style="color:#1d4ed8;"><em>3.6100</em></span> | <span style="color:#b91c1c;"><strong>3.6128</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0459</strong></span> | PASS |
| 2.333 | 5 | <span style="color:#1d4ed8;"><em>3.5000</em></span> | <span style="color:#b91c1c;"><strong>3.5023</strong></span> | <span style="color:#1d4ed8;"><em>0.0036</em></span> | <span style="color:#b91c1c;"><strong>0.1799</strong></span> | PASS |
| 2.333 | 6 | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | PASS |
| 2.333 | 7 | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4721</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0451</strong></span> | PASS |
| 2.500 | 5 | <span style="color:#1d4ed8;"><em>3.4500</em></span> | <span style="color:#b91c1c;"><strong>3.4515</strong></span> | <span style="color:#1d4ed8;"><em>0.0032</em></span> | <span style="color:#b91c1c;"><strong>0.1807</strong></span> | PASS |
| 2.500 | 6 | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | PASS |
| 2.500 | 7 | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4152</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0455</strong></span> | PASS |
| 2.667 | 5 | <span style="color:#1d4ed8;"><em>3.5100</em></span> | <span style="color:#b91c1c;"><strong>3.5115</strong></span> | <span style="color:#1d4ed8;"><em>0.0028</em></span> | <span style="color:#b91c1c;"><strong>0.1857</strong></span> | PASS |
| 2.667 | 6 | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| 2.667 | 7 | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4661</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0471</strong></span> | PASS |
| 2.833 | 5 | <span style="color:#1d4ed8;"><em>3.7400</em></span> | <span style="color:#b91c1c;"><strong>3.7384</strong></span> | <span style="color:#1d4ed8;"><em>0.0026</em></span> | <span style="color:#b91c1c;"><strong>0.1958</strong></span> | PASS |
| 2.833 | 6 | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | PASS |
| 2.833 | 7 | <span style="color:#1d4ed8;"><em>3.6800</em></span> | <span style="color:#b91c1c;"><strong>3.6780</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0503</strong></span> | PASS |
| 3.000 | 5 | <span style="color:#1d4ed8;"><em>4.2600</em></span> | <span style="color:#b91c1c;"><strong>4.2640</strong></span> | <span style="color:#1d4ed8;"><em>0.0024</em></span> | <span style="color:#b91c1c;"><strong>0.2127</strong></span> | PASS |
| 3.000 | 6 | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | PASS |
| 3.000 | 7 | <span style="color:#1d4ed8;"><em>4.1800</em></span> | <span style="color:#b91c1c;"><strong>4.1765</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0555</strong></span> | PASS |

### Table 5.11 — tolerance study (secondary / inconsistent in thesis)

| $p$ | $\varepsilon$ | thesis $J$ | repo $J$ | thesis error | repo error | status |
| --- | --- | --- | --- | --- | --- | --- |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.1634</strong></span> | secondary |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0035</em></span> | <span style="color:#b91c1c;"><strong>0.1635</strong></span> | secondary |
| 1.500 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.8800</em></span> | <span style="color:#b91c1c;"><strong>4.8823</strong></span> | <span style="color:#1d4ed8;"><em>0.0032</em></span> | <span style="color:#b91c1c;"><strong>0.1648</strong></span> | secondary |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.4700</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | secondary |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.1229</strong></span> | secondary |
| 1.667 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.4760</strong></span> | <span style="color:#1d4ed8;"><em>0.0019</em></span> | <span style="color:#b91c1c;"><strong>0.1232</strong></span> | secondary |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.1100</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | secondary |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | secondary |
| 1.833 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1190</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1055</strong></span> | secondary |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | secondary |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | secondary |
| 2.000 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.0965</strong></span> | secondary |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.6100</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | secondary |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | secondary |
| 2.167 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0918</strong></span> | secondary |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | secondary |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0011</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | secondary |
| 2.333 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0014</em></span> | <span style="color:#b91c1c;"><strong>0.0901</strong></span> | secondary |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | secondary |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | secondary |
| 2.500 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4225</strong></span> | <span style="color:#1d4ed8;"><em>0.0044</em></span> | <span style="color:#b91c1c;"><strong>0.0908</strong></span> | secondary |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | secondary |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0011</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | secondary |
| 2.667 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | secondary |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>3.6800</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | secondary |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0030</em></span> | <span style="color:#b91c1c;"><strong>0.0995</strong></span> | secondary |
| 2.833 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.6901</strong></span> | <span style="color:#1d4ed8;"><em>0.0030</em></span> | <span style="color:#b91c1c;"><strong>0.0996</strong></span> | secondary |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-05</strong></span> | <span style="color:#1d4ed8;"><em>4.1800</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | secondary |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.2000</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0065</em></span> | <span style="color:#b91c1c;"><strong>0.1091</strong></span> | secondary |
| 3.000 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.2000</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0065</em></span> | <span style="color:#b91c1c;"><strong>0.1092</strong></span> | secondary |

## Cross-Method Comparison: MPA, Iteration Counts, And Descent Directions

This section combines the MPA square tables with the cross-method and descent-direction comparison rows.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table table_5_6 --only-table table_5_7 --only-table table_5_13 \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/method_comparison
```

### MPA square branch

| table | $p$ | level | $\varepsilon$ | thesis $J$ | repo $J$ | thesis error | repo error | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| table_5_6 | 1.667 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4900</em></span> | <span style="color:#b91c1c;"><strong>4.5483</strong></span> | <span style="color:#1d4ed8;"><em>0.0084</em></span> | <span style="color:#b91c1c;"><strong>0.8137</strong></span> | FAIL |
| table_5_6 | 1.667 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.5367</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.7931</strong></span> | FAIL |
| table_5_6 | 1.667 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4700</em></span> | <span style="color:#b91c1c;"><strong>4.5334</strong></span> | <span style="color:#1d4ed8;"><em>0.0005</em></span> | <span style="color:#b91c1c;"><strong>0.7821</strong></span> | FAIL |
| table_5_6 | 1.833 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1400</em></span> | <span style="color:#b91c1c;"><strong>4.1818</strong></span> | <span style="color:#1d4ed8;"><em>0.0061</em></span> | <span style="color:#b91c1c;"><strong>0.4402</strong></span> | FAIL |
| table_5_6 | 1.833 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1228</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1910</strong></span> | PASS |
| table_5_6 | 1.833 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1100</em></span> | <span style="color:#b91c1c;"><strong>4.1102</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.1659</strong></span> | PASS |
| table_5_6 | 2.000 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8500</em></span> | <span style="color:#b91c1c;"><strong>3.8737</strong></span> | <span style="color:#1d4ed8;"><em>0.0049</em></span> | <span style="color:#b91c1c;"><strong>0.3993</strong></span> | FAIL |
| table_5_6 | 2.000 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8344</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.1202</strong></span> | PASS |
| table_5_6 | 2.000 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8285</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0619</strong></span> | PASS |
| table_5_6 | 2.167 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6400</em></span> | <span style="color:#b91c1c;"><strong>3.6023</strong></span> | <span style="color:#1d4ed8;"><em>0.0041</em></span> | <span style="color:#b91c1c;"><strong>0.4255</strong></span> | FAIL |
| table_5_6 | 2.167 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6218</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.1374</strong></span> | PASS |
| table_5_6 | 2.167 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6100</em></span> | <span style="color:#b91c1c;"><strong>3.6154</strong></span> | <span style="color:#1d4ed8;"><em>0.0003</em></span> | <span style="color:#b91c1c;"><strong>0.0930</strong></span> | PASS |
| table_5_6 | 2.500 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4500</em></span> | <span style="color:#b91c1c;"><strong>3.4517</strong></span> | <span style="color:#1d4ed8;"><em>0.0031</em></span> | <span style="color:#b91c1c;"><strong>0.1821</strong></span> | PASS |
| table_5_6 | 2.500 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4229</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0939</strong></span> | PASS |
| table_5_6 | 2.500 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4136</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.1393</strong></span> | PASS |
| table_5_6 | 2.667 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.5100</em></span> | <span style="color:#b91c1c;"><strong>3.5404</strong></span> | <span style="color:#1d4ed8;"><em>0.0028</em></span> | <span style="color:#b91c1c;"><strong>0.2994</strong></span> | FAIL |
| table_5_6 | 2.667 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.5008</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.2972</strong></span> | FAIL |
| table_5_6 | 2.667 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4700</em></span> | <span style="color:#b91c1c;"><strong>3.4661</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0481</strong></span> | PASS |
| table_5_6 | 2.833 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.7400</em></span> | <span style="color:#b91c1c;"><strong>3.7429</strong></span> | <span style="color:#1d4ed8;"><em>0.0026</em></span> | <span style="color:#b91c1c;"><strong>0.2084</strong></span> | PASS |
| table_5_6 | 2.833 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.7193</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1926</strong></span> | FAIL |
| table_5_6 | 2.833 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6800</em></span> | <span style="color:#b91c1c;"><strong>3.7076</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.1743</strong></span> | FAIL |
| table_5_6 | 3.000 | 5 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.2600</em></span> | <span style="color:#b91c1c;"><strong>4.2946</strong></span> | <span style="color:#1d4ed8;"><em>0.0024</em></span> | <span style="color:#b91c1c;"><strong>0.2409</strong></span> | FAIL |
| table_5_6 | 3.000 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1939</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1107</strong></span> | PASS |
| table_5_6 | 3.000 | 7 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1800</em></span> | <span style="color:#b91c1c;"><strong>4.1796</strong></span> | <span style="color:#1d4ed8;"><em>0.0002</em></span> | <span style="color:#b91c1c;"><strong>0.0656</strong></span> | PASS |
| table_5_7 | 1.667 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.5367</strong></span> | <span style="color:#1d4ed8;"><em>0.0021</em></span> | <span style="color:#b91c1c;"><strong>0.7931</strong></span> | FAIL |
| table_5_7 | 1.667 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.5367</strong></span> | <span style="color:#1d4ed8;"><em>0.0025</em></span> | <span style="color:#b91c1c;"><strong>0.7931</strong></span> | FAIL |
| table_5_7 | 1.667 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>4.4800</em></span> | <span style="color:#b91c1c;"><strong>4.5367</strong></span> | <span style="color:#1d4ed8;"><em>0.0033</em></span> | <span style="color:#b91c1c;"><strong>0.7931</strong></span> | FAIL |
| table_5_7 | 1.833 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1201</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1617</strong></span> | PASS |
| table_5_7 | 1.833 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.2006</strong></span> | <span style="color:#1d4ed8;"><em>0.0016</em></span> | <span style="color:#b91c1c;"><strong>0.5080</strong></span> | FAIL |
| table_5_7 | 1.833 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>4.1200</em></span> | <span style="color:#b91c1c;"><strong>4.1228</strong></span> | <span style="color:#1d4ed8;"><em>0.0034</em></span> | <span style="color:#b91c1c;"><strong>0.1910</strong></span> | PASS |
| table_5_7 | 2.000 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8385</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.1978</strong></span> | PASS |
| table_5_7 | 2.000 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8342</strong></span> | <span style="color:#1d4ed8;"><em>0.0012</em></span> | <span style="color:#b91c1c;"><strong>0.1167</strong></span> | PASS |
| table_5_7 | 2.000 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8324</strong></span> | <span style="color:#1d4ed8;"><em>0.0019</em></span> | <span style="color:#b91c1c;"><strong>0.0966</strong></span> | PASS |
| table_5_7 | 2.167 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6221</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.1402</strong></span> | PASS |
| table_5_7 | 2.167 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6202</strong></span> | <span style="color:#1d4ed8;"><em>0.0011</em></span> | <span style="color:#b91c1c;"><strong>0.1112</strong></span> | PASS |
| table_5_7 | 2.167 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.6200</em></span> | <span style="color:#b91c1c;"><strong>3.6180</strong></span> | <span style="color:#1d4ed8;"><em>0.0019</em></span> | <span style="color:#b91c1c;"><strong>0.0921</strong></span> | PASS |
| table_5_7 | 2.333 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.5076</strong></span> | <span style="color:#1d4ed8;"><em>0.0009</em></span> | <span style="color:#b91c1c;"><strong>0.2342</strong></span> | FAIL |
| table_5_7 | 2.333 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4811</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.1248</strong></span> | PASS |
| table_5_7 | 2.333 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4781</strong></span> | <span style="color:#1d4ed8;"><em>0.0004</em></span> | <span style="color:#b91c1c;"><strong>0.0903</strong></span> | PASS |
| table_5_7 | 2.500 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4229</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0939</strong></span> | PASS |
| table_5_7 | 2.500 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4229</strong></span> | <span style="color:#1d4ed8;"><em>0.0008</em></span> | <span style="color:#b91c1c;"><strong>0.0939</strong></span> | PASS |
| table_5_7 | 2.500 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.4200</em></span> | <span style="color:#b91c1c;"><strong>3.4229</strong></span> | <span style="color:#1d4ed8;"><em>0.0010</em></span> | <span style="color:#b91c1c;"><strong>0.0939</strong></span> | PASS |
| table_5_7 | 2.667 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4810</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.1143</strong></span> | PASS |
| table_5_7 | 2.667 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0938</strong></span> | PASS |
| table_5_7 | 2.667 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.4800</em></span> | <span style="color:#b91c1c;"><strong>3.4751</strong></span> | <span style="color:#1d4ed8;"><em>0.0007</em></span> | <span style="color:#b91c1c;"><strong>0.0937</strong></span> | PASS |
| table_5_7 | 2.833 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.7193</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1926</strong></span> | FAIL |
| table_5_7 | 2.833 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.7193</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1926</strong></span> | FAIL |
| table_5_7 | 2.833 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>3.6900</em></span> | <span style="color:#b91c1c;"><strong>3.7193</strong></span> | <span style="color:#1d4ed8;"><em>0.0018</em></span> | <span style="color:#b91c1c;"><strong>0.1926</strong></span> | FAIL |
| table_5_7 | 3.000 | 6 | <span style="color:#b91c1c;"><strong>1e-04</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1972</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1425</strong></span> | PASS |
| table_5_7 | 3.000 | 6 | <span style="color:#b91c1c;"><strong>1e-03</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1941</strong></span> | <span style="color:#1d4ed8;"><em>0.0006</em></span> | <span style="color:#b91c1c;"><strong>0.1104</strong></span> | PASS |
| table_5_7 | 3.000 | 6 | <span style="color:#b91c1c;"><strong>1e-02</strong></span> | <span style="color:#1d4ed8;"><em>4.1900</em></span> | <span style="color:#b91c1c;"><strong>4.1940</strong></span> | <span style="color:#1d4ed8;"><em>0.0015</em></span> | <span style="color:#b91c1c;"><strong>0.1092</strong></span> | PASS |

### Table 5.13 — direction comparison

| method | direction | $p$ | thesis iters | repo iters | thesis direction iters | status |
| --- | --- | --- | --- | --- | --- | --- |
| oa1 | d | 1.833 | <span style="color:#1d4ed8;"><em>9</em></span> | <span style="color:#b91c1c;"><strong>11</strong></span> | <span style="color:#1d4ed8;"><em>11</em></span> | PASS |
| oa1 | d | 2.000 | <span style="color:#1d4ed8;"><em>7</em></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | <span style="color:#1d4ed8;"><em>7</em></span> | PASS |
| oa1 | d | 2.167 | <span style="color:#1d4ed8;"><em>6</em></span> | <span style="color:#b91c1c;"><strong>10</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | PASS |
| oa1 | d | 2.333 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | PASS |
| oa1 | d | 2.500 | <span style="color:#1d4ed8;"><em>10</em></span> | <span style="color:#b91c1c;"><strong>11</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| oa1 | d | 2.667 | <span style="color:#1d4ed8;"><em>7</em></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | FAIL |
| oa1 | d | 2.833 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| oa1 | d | 3.000 | <span style="color:#1d4ed8;"><em>14</em></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| oa1 | d_vh | 1.833 | <span style="color:#1d4ed8;"><em>9</em></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| oa1 | d_vh | 2.000 | <span style="color:#1d4ed8;"><em>7</em></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | <span style="color:#1d4ed8;"><em>7</em></span> | PASS |
| oa1 | d_vh | 2.167 | <span style="color:#1d4ed8;"><em>6</em></span> | <span style="color:#b91c1c;"><strong>6</strong></span> | <span style="color:#1d4ed8;"><em>6</em></span> | PASS |
| oa1 | d_vh | 2.333 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| oa1 | d_vh | 2.500 | <span style="color:#1d4ed8;"><em>10</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | PASS |
| oa1 | d_vh | 2.667 | <span style="color:#1d4ed8;"><em>7</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>7</em></span> | PASS |
| oa1 | d_vh | 2.833 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| oa1 | d_vh | 3.000 | <span style="color:#1d4ed8;"><em>14</em></span> | <span style="color:#b91c1c;"><strong>10</strong></span> | <span style="color:#1d4ed8;"><em>14</em></span> | FAIL |
| rmpa | d | 1.833 | <span style="color:#1d4ed8;"><em>13</em></span> | <span style="color:#b91c1c;"><strong>18</strong></span> | <span style="color:#1d4ed8;"><em>19</em></span> | PASS |
| rmpa | d | 2.000 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| rmpa | d | 2.167 | <span style="color:#1d4ed8;"><em>12</em></span> | <span style="color:#b91c1c;"><strong>10</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | PASS |
| rmpa | d | 2.333 | <span style="color:#1d4ed8;"><em>9</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| rmpa | d | 2.500 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | <span style="color:#1d4ed8;"><em>7</em></span> | PASS |
| rmpa | d | 2.667 | <span style="color:#1d4ed8;"><em>10</em></span> | <span style="color:#b91c1c;"><strong>11</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| rmpa | d | 2.833 | <span style="color:#1d4ed8;"><em>16</em></span> | <span style="color:#b91c1c;"><strong>14</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | FAIL |
| rmpa | d | 3.000 | <span style="color:#1d4ed8;"><em>26</em></span> | <span style="color:#b91c1c;"><strong>31</strong></span> | <span style="color:#1d4ed8;"><em>19</em></span> | FAIL |
| rmpa | d_vh | 1.833 | <span style="color:#1d4ed8;"><em>13</em></span> | <span style="color:#b91c1c;"><strong>13</strong></span> | <span style="color:#1d4ed8;"><em>13</em></span> | PASS |
| rmpa | d_vh | 2.000 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>8</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| rmpa | d_vh | 2.167 | <span style="color:#1d4ed8;"><em>12</em></span> | <span style="color:#b91c1c;"><strong>12</strong></span> | <span style="color:#1d4ed8;"><em>12</em></span> | PASS |
| rmpa | d_vh | 2.333 | <span style="color:#1d4ed8;"><em>9</em></span> | <span style="color:#b91c1c;"><strong>9</strong></span> | <span style="color:#1d4ed8;"><em>9</em></span> | PASS |
| rmpa | d_vh | 2.500 | <span style="color:#1d4ed8;"><em>8</em></span> | <span style="color:#b91c1c;"><strong>7</strong></span> | <span style="color:#1d4ed8;"><em>8</em></span> | PASS |
| rmpa | d_vh | 2.667 | <span style="color:#1d4ed8;"><em>10</em></span> | <span style="color:#b91c1c;"><strong>12</strong></span> | <span style="color:#1d4ed8;"><em>10</em></span> | PASS |
| rmpa | d_vh | 2.833 | <span style="color:#1d4ed8;"><em>16</em></span> | <span style="color:#b91c1c;"><strong>16</strong></span> | <span style="color:#1d4ed8;"><em>16</em></span> | PASS |
| rmpa | d_vh | 3.000 | <span style="color:#1d4ed8;"><em>26</em></span> | <span style="color:#b91c1c;"><strong>20</strong></span> | <span style="color:#1d4ed8;"><em>26</em></span> | FAIL |

## Square Multiple-Solution Study (Table 5.14)

OA1 stays on the principal branch for the square seeds, while OA2 can recover distinct higher branches depending on the initialisation.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table table_5_14 \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/square_multibranch
```

![Square OA1/OA2 multibranch panel](../assets/plaplace_u3_thesis/square_multibranch_panel.png)

| seed | method | thesis $J$ | repo $J$ | thesis $I$ | repo $I$ | status |
| --- | --- | --- | --- | --- | --- | --- |
| sine | oa1 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>1.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | PASS |
| sine | oa2 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>1.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | PASS |
| sine_x2 | oa1 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>1.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | PASS |
| sine_x2 | oa2 | <span style="color:#1d4ed8;"><em>21.7400</em></span> | <span style="color:#b91c1c;"><strong>21.7443</strong></span> | <span style="color:#1d4ed8;"><em>3.0500</em></span> | <span style="color:#b91c1c;"><strong>3.0539</strong></span> | PASS |
| sine_y2 | oa1 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>1.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | PASS |
| sine_y2 | oa2 | <span style="color:#1d4ed8;"><em>21.7400</em></span> | <span style="color:#b91c1c;"><strong>21.7443</strong></span> | <span style="color:#1d4ed8;"><em>3.0500</em></span> | <span style="color:#b91c1c;"><strong>3.0539</strong></span> | PASS |
| skew | oa1 | <span style="color:#1d4ed8;"><em>3.8300</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>1.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | PASS |
| skew | oa2 | <span style="color:#1d4ed8;"><em>19.8000</em></span> | <span style="color:#b91c1c;"><strong>3.8278</strong></span> | <span style="color:#1d4ed8;"><em>2.9800</em></span> | <span style="color:#b91c1c;"><strong>1.9781</strong></span> | FAIL |

## Square-With-Hole OA2 Study (Figure 5.13)

This nonconvex domain is the sharpest multi-solution benchmark in the thesis packet and is the main extension case beyond the square.

```bash
./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --only-table figure_5_13 \
  --out-dir artifacts/raw_results/plaplace_u3_thesis_sections/square_hole
```

![Square-hole OA2 panel](../assets/plaplace_u3_thesis/square_hole_panel.png)

| seed | thesis $J$ | repo $J$ | thesis $I$ | repo $I$ | status |
| --- | --- | --- | --- | --- | --- |
| abs_sine_3x3 | <span style="color:#1d4ed8;"><em>71.2205</em></span> | <span style="color:#b91c1c;"><strong>71.2205</strong></span> | <span style="color:#1d4ed8;"><em>4.1083</em></span> | <span style="color:#b91c1c;"><strong>4.1083</strong></span> | PASS |
| abs_sine_y2 | <span style="color:#1d4ed8;"><em>90.3644</em></span> | <span style="color:#b91c1c;"><strong>90.3642</strong></span> | <span style="color:#1d4ed8;"><em>4.3603</em></span> | <span style="color:#b91c1c;"><strong>4.3603</strong></span> | PASS |
| sine | <span style="color:#1d4ed8;"><em>45.1821</em></span> | <span style="color:#b91c1c;"><strong>45.1816</strong></span> | <span style="color:#1d4ed8;"><em>3.6665</em></span> | <span style="color:#b91c1c;"><strong>3.6665</strong></span> | PASS |
| skew | <span style="color:#1d4ed8;"><em>36.0353</em></span> | <span style="color:#b91c1c;"><strong>36.0353</strong></span> | <span style="color:#1d4ed8;"><em>3.4650</em></span> | <span style="color:#b91c1c;"><strong>3.4650</strong></span> | PASS |

## Rebuild The Canonical Thesis Packet And This Page

Use the canonical summary to rebuild the assignment-facing report and this merged docs page without rerunning the full raw suite.

```bash
./.venv/bin/python -u experiments/analysis/generate_plaplace_u3_thesis_report.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --summary-label artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --out artifacts/reports/plaplace_u3_thesis/README.md
```

```bash
./.venv/bin/python -u experiments/analysis/generate_plaplace_u3_thesis_problem_page.py \
  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \
  --out docs/problems/pLaplace_u3_thesis_replications.md \
  --asset-dir docs/assets/plaplace_u3_thesis
```

## What Matches, What Is Partial, And What Does Not Match

- primary assignment rows passing the current thresholds: `159` / `185`
- secondary / partial rows: `61`
- mismatch rows: `44`

### What works

- Stage A and Stage B square principal-branch energies largely track the published thesis values.
- Stage E square-with-hole OA2 values currently match all four published seeds in the canonical packet.
- The merged page uses docs-owned assets and only repo-relative links.

### What is partial

| target | note |
| --- | --- |
| Section 18 / d^R_N sanity | secondary target |
| Section 14.1 / Table 5.8 | secondary target |
| Section 14.2 / Table 5.9 | secondary target |
| Section 15.1 / Table 5.10 | secondary target |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |
| Section 15.2 / Table 5.11 | Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target. |

### What does not match

| target | method | status | gap class |
| --- | --- | --- | --- |
| Section 18 / Table 5.2 | rmpa | maxit | 1D harness did not meet thesis stopping tolerance |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.2 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | maxit | 1D harness did not meet thesis stopping tolerance |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / Table 5.3 | rmpa | completed | Unresolved thesis mismatch |
| Section 18 / d^R_N sanity | rmpa | maxit | 1D harness did not meet thesis stopping tolerance |
| Section 14.1 / Table 5.8 | rmpa | maxit | Low-p RMPA target / branch sensitivity |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.6 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.1 / Table 5.7 | mpa | maxit | MPA convergence budget / step robustness |
| Section 16.2 / Table 5.13 | oa1 | completed | Unresolved thesis mismatch |
| Section 16.2 / Table 5.13 | oa1 | completed | Unresolved thesis mismatch |
| Section 16.2 / Table 5.13 | rmpa | completed | Unresolved thesis mismatch |
| Section 16.2 / Table 5.13 | rmpa | completed | Unresolved thesis mismatch |
| Section 16.2 / Table 5.13 | rmpa | completed | Unresolved thesis mismatch |
| Section 17.1 / Table 5.14 | oa2 | completed | OA2 square branch-selection mismatch |
