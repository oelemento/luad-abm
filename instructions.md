---

## 0) Design goals (speed + credibility)

* **Keep it small, vectorized, and cache‑friendly.** Single 2D lattice representing a 1 mm × 1 mm ROI at **\~10 µm/site** → **100 × 100 grid**. At most **1 agent/site**; Moore neighborhood interactions; no expensive pairwise searches.
* **Minimal yet expressive agents:** tumor/epithelium, CAF/αSMA⁺ fibroblast, CD8 T, CD4 T, Treg, macrophage (single agent class with a continuous M1↔M2 score), **TLS node** (static or slowly adapting), optional PMN‑MDSC as a “field effect” rather than thousands of agents.
* **Few fields:** ECM density/stiffness (CAF‑deposited), CXCL9/10 (T‑cell chemotaxis), CXCL13 (TLS attractant), TGFβ (EMT + suppression). Discrete diffusion by separable kernels (fast 2D convolution) each step.
* **Outputs by default:** time series CSVs, static figures (counts, distances, interaction heatmap), and a **movie** (MP4 or GIF).
* **Two presets (switch at CLI):** **G3 inflammatory** vs **G4 fibrotic** phenotypes (see Fig. 5 for composition/architecture differences; **Fig. 4f** shows reduced epithelial–immune proximity and elevated epithelial–αSMA proximity to match).&#x20;

---

## 1) Biological anchors from your paper → modeling constraints

* **Co‑expansion of CD8 and Tregs; functional rise in GZMB⁺ CD8 and CTLA‑4⁺ Tregs** across progression (Fig. 2c–h). → Model both **kill capability and suppression**; Treg presence reduces local CD8 kill probability.&#x20;
* **Macrophage shift to monocyte‑derived and M1↔M2 polarization; VISTA⁺/M2‑like suppression present** (Fig. 2l–m). → Single macrophage class with a continuous polarization variable controlling suppression vs stimulation.&#x20;
* **CAF/αSMA⁺ cuffs around tumor nests; immune exclusion; EMT increase** (Figs. 3–4). → ECM field around tumor lowers T‑cell speed/contact, increases EMT score in nearby tumor; **epithelial–immune adjacency drops while epithelial–αSMA rises** (Fig. 4f target).&#x20;
* **TLS expansion but functional drift (ICOS/GZMB↓; TIM‑3↑)** (Fig. 4d–e). → TLS node emits CXCL13/CXCL9/10; a **TLS “quality” state** lowers the fraction of high‑quality effectors over time.&#x20;
* **Two archetypes at patient level**: **G3 inflammatory** vs **G4 fibrotic/immune‑excluded** with higher EMT/fibrosis/angiogenesis and lower immune infiltration; **G4 misclassified as PNS on CT but biologically invasive** (Fig. 5c–e; Abstract/Discussion). → Two parameter presets and a simple **“solidity proxy” = ECM fraction**.&#x20;

---

## 2) Model architecture (Mesa)

**Core classes**

* `LUADModel(Model)`

  * Space: `MultiGrid(100, 100, torus=False)` (1 site ≈ 10 µm).
  * Scheduler: `StagedActivation` with stages: `fields → movement → interactions → state_updates`.
  * Fields (NumPy arrays): `ECM`, `CXCL9_10`, `CXCL13`, `TGFb`.
  * DataCollector: counts per type, mean distances (CD8→nearest tumor), **interaction matrix** (see §6), EMT fraction, ECM fraction (“solidity proxy”).
  * Params via YAML/JSON (see §3 presets and §7 knobs).

* `EpithelialTumorAgent`: state in {AT1‑like, AT2‑like, tumor}, **EMT∈\[0,1]**, MHC‑I, PD‑L1, proliferative flag.

* `CAF`: activation level; deposits ECM; secretes TGFβ.

* `CD8T`, `CD4T`, `Treg`: activation/exhaustion∈\[0,1]; chemotaxis sensitivity; local suppression susceptibility.

* `Macrophage`: polarization p∈\[−1,+1] (−1=M2, +1=M1); suppression strength rises as p→−1.

* `TLSNode`: static or slowly updating; emits CXCL13 and CXCL9/10; **quality q∈\[0,1]** that tilts T‑cell effector vs exhausted fate.

**Fast interaction rules (local only)**

* **Movement:** One‑step to a neighboring site, probability ∝ chemokine gradient × (1−ECM\_penalty).
* **Killing:** If CD8 adjacent to tumor → kill with prob = `base_kill × f(MHC‑I, PD‑L1) × (1−Treg_suppr) × (1−M2_suppr)`.
* **Suppression:** Treg/M2 reduce nearby CD8 effective kill probability; **no global fields** for suppression (keeps it fast).
* **EMT update:** Tumor EMT increases with local TGFβ/ECM; decreased by anti‑TGFβ intervention.
* **ECM update:** CAFs increment ECM locally; ECM decays slowly.
* **TLS quality drift:** q(t) decreases without intervention (mirrors Fig. 4d–e functional shift).&#x20;

---

## 3) Two ready‑to‑use presets (values are intentionally simple but separable)

> All counts are **initial agents**; adjust as needed without changing logic.

* **G3\_inflammatory.json**

  * Tumor: 1,200; CAF: 300; Macrophage: 250; CD8: 700; CD4: 500; Treg: 120; TLS nodes: 2
  * ECM init mean: 0.20; TLS quality q₀: 0.7; Chemokine gain: high; TGFβ gain: moderate

* **G4\_fibrotic.json**

  * Tumor: 1,200; CAF: 700; Macrophage: 300 (biased to M2); CD8: 250; CD4: 200; Treg: 80; TLS nodes: 1
  * ECM init mean: 0.50 (peritumoral ring); TLS quality q₀: 0.4; Chemokine gain: low; TGFβ gain: high

**Why these choices:** G4 gets more CAF/ECM and fewer lymphocytes; G3 gets more lymphocytes and better TLS quality, reflecting **Fig. 5c** and **Fig. 4c–f** patterns.&#x20;

---

## 4) Lightweight parameter defaults (numeric “good starters”)

*(These are deliberately simple for speed and stability; refine later as needed.)*

* **Grid & timing:** 100×100; **Δx ≈ 10 µm; Δt = 1 min**.
* **Movements (per minute):** CD8/CD4/Treg attempt 1 step; macrophage 1 step every 2–3 min; CAFs 1 step every \~10 min (mostly stationary).
* **Chemotaxis strength:** choose `β_chem` so a cell moves up‑gradient \~60–70% of the time when a gradient exists.
* **Diffusion/decay (per step):**

  * CXCL9/10: kernel `[[0,1,0],[1,4,1],[0,1,0]] / 8` with decay 1–2%/step
  * CXCL13: same kernel, decay 1%/step
  * TGFβ: same kernel, decay 0.5%/step
* **ECM:** CAF deposition +0.02/step in a 3×3 neighborhood; decay 0.001/step; cap at 1.0.
* **CD8 kill:** base 0.25 on adjacency, ×(1−0.5·PD‑L1) ×(1−Treg\_suppr) ×(1−M2\_suppr); IFNγ boost (optional) +0.05 if recent kill.
* **Suppression multipliers (local neighborhood):** Treg\_suppr = 0.15–0.35; M2\_suppr = 0.10–0.30 (higher in G4).
* **EMT update:** ΔEMT = `+0.02·(TGFβ+ECM)` near CAFs; −0.03 if anti‑TGFβ on.
* **Exhaustion:** CD8 exhaustion += 0.02 per minute when repeatedly contacting tumor or in suppressive neighborhoods; PD‑1 block halves accrual.
* **TLS quality drift:** q(t+1)=q(t)−0.002 baseline; **TLS‑boost** sets floor at 0.5 and increases effector seeding.

*(Your paper fixes **relative** trends and spatial patterns; these starting values are chosen to achieve them fast and stably. Calibrate to match **Fig. 2–4 medians/contrasts** rather than exact absolute counts.)*&#x20;

---

## 5) Interventions (fast toggles)

* **PD‑1/PD‑L1 blockade:** halve exhaustion accrual and add +0.1 to effective kill.
* **TIM‑3 blockade:** prevents TLS quality‑driven loss of effector function (raises q by +0.1 and slows its drift).
* **CTLA‑4/Treg modulation:** reduce Treg suppression by 50% or trim Treg count by 30%.
* **VISTA/CSF1R (TAM reprogramming):** bias macrophage polarization p toward +1 by +0.2 and reduce M2\_suppr by 40%.
* **Anti‑TGFβ / anti‑fibrotic:** lower CAF deposition by 50%, ECM penalty by 30%, and EMT Δ by 50%.
* **TLS‑boost (e.g., CXCL13/CD40 agonism proxy):** raise q toward 0.8; increase effector seeding rate from TLS.

**Pre‑canned combos to demonstrate non‑trivial context:**

* **G4:** Anti‑TGFβ + PD‑1; VISTA/CSF1R + PD‑1.
* **G3:** Treg modulation + PD‑1; TLS‑boost + TIM‑3.
  These map directly to **Fig. 2 (co‑expansion), Fig. 3 (EMT/CAF), Fig. 4 (TLS shift & exclusion), Fig. 5 (G3 vs G4)**.&#x20;

---

## 6) Quantitative readouts (what the model emits every N steps)

* **Counts:** agents per type; tumor viable count; EMT‑high fraction (EMT>0.6).
* **Distances:** distribution of **CD8→nearest tumor** cell distance (target shift: G4 right‑shifted).
* **Interaction matrix (Fig. 4f analogue):** for each tick, compute counts of **adjacent** pairs (epithelial↔immune, epithelial↔αSMA, epithelial↔epithelial, etc.) and normalize by permutation‑based expectation (quick trick: 10–20 label shuffles on the grid). Expect **↑ epithelial–αSMA** and **↓ epithelial–immune** in tumor vs normal/G3.&#x20;
* **Niche areas:** approximate using thresholds on ECM (tumor–stroma interface) and TLS coverage (compare **Fig. 4c**).&#x20;
* **“Solidity proxy”:** ECM fraction; report alongside EMT fraction to illustrate the **radiology–biology discordance** seen in **G4** (Abstract, **Fig. 5e**).&#x20;

---

## 7) Performance knobs (so it runs smoothly)

* Grid size (e.g., 80–140 per side), step frequency for diffusion (every step vs every 2–3 steps), number of shuffles for the interaction baseline (10 is fine), and whether to include MDSC as explicit agents (off by default—use a **global “suppressive background”** scalar early on to mimic Fig. 2j–k).&#x20;
* Use vectorized 2D convolutions for fields; pre‑allocate arrays; avoid Python loops over agents where possible (e.g., move agents by scanning their local 3×3 patch only).

---

## 8) File/folder layout (so you can build quickly)

```
luad_abm/
  config/
    G3_inflammatory.json
    G4_fibrotic.json
    interventions/*.json            # optional presets
  luad/
    model.py                        # LUADModel + fields engine
    agents.py                       # classes: Tumor, CAF, CD8, CD4, Treg, Macro, TLS
    rules.py                        # movement, interactions, state updates
    fields.py                       # kernels & diffusion/decay
    metrics.py                      # distances, interaction matrix, niches
    viz.py                          # matplotlib plotting + movie frames
    io.py                           # DataCollector dumps to CSV/NPZ
  runs/
    run.py                          # CLI: select preset + interventions + seed
  outputs/
    G3_baseline/
      frames/*.png
      timeseries.csv
      interactions.csv
      summary_plots/*.png
      movie.mp4 (or movie.gif)
```

**CLI examples**

```
python runs/run.py --preset config/G3_inflammatory.json --ticks 2000 --out outputs/G3_base
python runs/run.py --preset config/G4_fibrotic.json --add PD1 --add antiTGFb --ticks 2000 --out outputs/G4_PD1_antiTGFb
```

---

## 9) What the code should automatically save (no manual fiddling)

* **Static plots (PNG):**

  1. **Trajectories:** counts/time; EMT‑high fraction/time; ECM fraction/time.
  2. **Distance CDFs:** CD8→tumor for snapshot times.
  3. **Interaction heatmap:** tumor vs normal‑like vs intervention (Fig. 4f‑style).
  4. **TLS quality vs effector fraction** (dot/line overlay; compare **Fig. 4d–e**).&#x20;
* **Movie (MP4/GIF):** per‑tick raster with 5 overlays—tumor (PanCK proxy), CAF/ECM (αSMA/ECM), CD8, Tregs, macrophage shade by polarization, TLS nodes. Add a small side panel showing counts/time.

---

## 10) Minimal experiment set for the talk (clear, non‑trivial results)

1. **G3 baseline → Treg modulation + PD‑1**: expect higher effective kills near tumor, preserved CD8 proximity, and slower decline in effector fraction (consistent with **co‑expansion and regulation**, Fig. 2d–h).&#x20;
2. **G4 baseline → anti‑TGFβ + PD‑1**: expect **thinner CAF/ECM ring**, improved CD8→tumor contacts (distance CDF left‑shift), lower EMT fraction (reflects **Fig. 3 and Fig. 4f** barriers).&#x20;
3. **G4 baseline → VISTA/CSF1R + PD‑1**: macrophage polarization toward M1 reduces suppression; CD8 kills increase even if ECM remains. **Show complementarity** vs anti‑TGFβ arm (biology: **Fig. 2m**).&#x20;
4. **G3 baseline → TLS‑boost + TIM‑3**: maintain TLS “quality”; show slower drop of ICOS/GZMB proxy and higher durable effector fraction (compare **Fig. 4d–e**).&#x20;

---

## 11) Quick calibration checklist against your figures

* **Composition sanity:** trajectories of B/CD4/CD8/Treg and macrophage polarization resemble **Fig. 2** trends (qualitative).&#x20;
* **Spatial adjacency:** replicate **↓ epithelial–immune** and **↑ epithelial–αSMA** in tumor vs normal; replicate stronger exclusion in the **G4** preset (**Fig. 4f**, page 35).&#x20;
* **Niche areas:** ensure tumor–stroma interface expansion vs normal and larger TLS area in invasive settings (**Fig. 4c**, page 35).&#x20;
* **Patient‑level archetypes:** ECM fraction (“solidity proxy”) higher in G4, EMT fraction higher; immune fractions lower (cf. **Fig. 5c–e**).&#x20;

---

## 12) “Good enough today” parameter tuning order (fast)

1. **Movement/chemotaxis** until CD8 finds tumor borders in G3 but not in G4.
2. **ECM penalty** until epithelial–immune adjacency dips and epithelial–αSMA climbs (**Fig. 4f pattern**).&#x20;
3. **EMT dynamics** until ECM/TGFβ raises EMT fraction in G4.
4. **Suppression multipliers** so PD‑1 alone helps in G3 but is limited in G4 unless combined.

---

## 13) What to show on slides (straight from the outputs)

* **Panel A:** 2×2 snapshots (G3 vs G4; baseline vs combo), with **adjacency heatmaps** underneath (Fig. 4f analogue).&#x20;
* **Panel B:** **Distance CDF** (CD8→tumor) and **EMT fraction** before/after intervention.
* **Panel C:** **TLS quality vs effector fraction** trajectory plot (TLS‑boost/TIM‑3).
* **Panel D:** **Short movie clip** (10–20 s) of the G4 anti‑TGFβ+PD‑1 run showing ECM ring thinning and CD8 entry.

---

## 14) Known shortcuts (if you need even more speed)

* Turn off macrophage heterogeneity: keep polarization as a scalar on the **grid** instead of per agent.
* Compute the interaction matrix every **k** ticks (e.g., every 10th tick) and reuse.
* Diffuse fields every **2–3 ticks** without noticeable visual loss.
* Represent PMN‑MDSC as a **transient global suppression scalar** early on (motivated by their early spike then plateau in **Fig. 2j–k**).&#x20;

---

## 15) Notes on scientific grounding

* All modeling choices above are tied to your paper’s principal findings: **co‑expansion of effectors and regulators, macrophage polarization, CAF‑driven immune exclusion and EMT, TLS functional drift, and the G3/G4 archetypes**, with page‑level anchors to **Figs. 2–5** (pp. 33–37) and the **Abstract/Discussion** (pp. 2, 14–17). The simulation’s job is to **reproduce those patterns** and then test interception strategies on top.&#x20;

