# DrainLite: leakage-controlled residual emulation of a fixed conceptual sewer network for urban flood modelling

## Highlights

- Matched surface controls isolate sewer exchange from inlet-neighbourhood roughness.
- A road-aligned conceptual network connects 2,276 junctions to 221 receiving interfaces.
- Whole-event validation reduces coupled-label depth error from 9.266 to 2.169 mm.
- A training-event prior explains most of the repeatable fixed-network response.
- Static network fields add 0.048 +/- 0.002 mm MAE improvement across five sampling seeds.

## Abstract

Urban flood surrogates commonly reproduce the output of a coupled hydrodynamic model without exposing the drainage system as an explicit condition. This study considers a narrower problem: whether the surface-depth change caused by one specified sewer network can be emulated as a lightweight correction to a drainage-free hydrodynamic state. A road-aligned conceptual network was constructed for the 20 m Shenzhen benchmark and coupled bidirectionally to the two-dimensional Itzï solver through the Storm Water Management Model. Eight six-hour rainfall events were calculated under an original surface configuration (A), a surface configuration with coupling-neighbourhood roughness but no pipe exchange (B), and a bidirectionally coupled 1D-2D hydrodynamic configuration using Itzï and SWMM (C). The signed learning target was C-B.

DrainLite combines an event-excluded spatiotemporal residual prior with current rainfall, the contemporaneous B depth field, terrain and rasterised network attributes. Complete events were held out in turn. Five independent spatial sampling seeds were used for the principal dynamic and full-network models, while model complexity was fixed before fitting. Across 60,783,552 held-out active cell-times, B differed from C by 9.266 mm mean absolute error. The spatiotemporal prior attained 3.226 mm, the prior-plus-dynamic model attained 2.217 mm, and the full hybrid attained 2.169 mm. The static-network increment averaged 0.048 mm across sampling seeds and was smaller than the contribution of the prior and dynamic state. Displacing, block-shuffling or jointly permuting the network fields increased the error of the final hybrid, indicating that the small increment depended on spatial alignment.

The coupled calculations met the repository's topology, continuity and mass-balance conditions. Their mean routing continuity error was -0.091%, and their mean signed depth change was -7.860 mm. Comparison with public MIKE fields was metric dependent: coupling reduced final-volume error but increased full space-time depth error. MIKE was therefore retained as a descriptive external comparator, not a training label or independent validation dataset. The resulting method is a fixed-network residual emulator that requires the contemporaneous surface-only field; it is not an end-to-end rainfall-to-flood predictor and does not establish transfer to an unseen sewer layout.

**Keywords:** urban pluvial flooding; conceptual sewer network; Itzï; Storm Water Management Model; residual emulation; whole-event validation

## 1. Introduction

Short-duration urban flooding reflects the joint influence of rainfall, buildings, surface conveyance, local storage and underground drainage. Two-dimensional surface models resolve overland propagation, whereas one-dimensional network models describe flow through junctions and pipes. Their exchange through inlets can reduce surface storage, redistribute water and return surcharge to the street. Omitting this exchange does not make a surface calculation dynamically invalid, but it changes the system being represented.

The computational cost of coupled simulation remains a practical barrier to event ensembles and rapid forecasting. Neural operators offer a complementary route by learning mappings between forcing fields and hydrodynamic solutions. The latent autoregressive neural operator (LarNO) was developed for large-area urban flood prediction and demonstrated zero-shot evaluation across grid resolutions using a public Shenzhen benchmark (Cao et al., 2026). Its released inputs and checkpoint permit reproduction of the reference mapping, but the municipal drainage inventory represented in the MIKE calculations is not released as an independently changeable model input.

One response would be to retrain the complete neural operator with additional pipe channels. That strategy requires many matched network simulations and substantially more hardware than is available in the present study. A residual formulation provides a lower-cost alternative. If a drainage-free model supplies a surface state, a second model can estimate only the depth difference induced by a prescribed sewer system. This decomposition is useful only if the target is physically identifiable and the validation prevents information from the tested event entering the predictor.

Fixed terrain and a fixed network create a further difficulty. Their drainage response may recur at the same cells and times across events. A model can therefore appear skilful by recovering a climatological template, even if it makes little use of rainfall or pipe attributes. Random cell-wise splitting compounds this problem, because adjacent samples from one event enter both fitting and evaluation. A suitable test must retain entire rainfall events, compare against an event-excluded spatiotemporal prior, and perturb the network fields in the final model rather than in an earlier surrogate.

Here, DrainLite is formulated as a correction layer for a fixed conceptual network. Three questions are examined. First, does a connected road-aligned network produce a numerically controlled response that can be separated from ancillary roughness changes? Second, how much of this response is explained by a cross-event prior and by the current surface-rainfall state? Third, after those predictors are known, is the remaining contribution of correctly aligned static network information detectable? MIKE and the public LarNO checkpoint are analysed outside the residual-fitting pathway to clarify what the experiment does, and does not, establish.

![Figure 1. Residual learning with an offline archive. The prior requires C-B labels from other historical events at each cell and simulation time. Both the outer test event and each fitting row's own event are excluded from its prior. Current B, rainfall, terrain and fixed network fields enter the estimator; current-event C is used only for held-event scoring.](figures/fig01_workflow.png)

**Figure 1. Residual learning with an offline archive. The prior requires C-B labels from other historical events at each cell and simulation time. Both the outer test event and each fitting row's own event are excluded from its prior. Current B, rainfall, terrain and fixed network fields enter the estimator; current-event C is used only for held-event scoring.**

## 2. Data and methods

### 2.1 Study domain and event inclusion

The public 20 m benchmark comprises a 400 x 560 rectangular grid, corresponding to 8.0 x 11.2 km. The reporting mask contains 105,527 active cells (42.21 km2). Each event contains 72 five-minute fields spanning six hours. Water depth is stored in metres and rainfall as millimetres per five-minute interval. Cells at DEM elevations of at least 49.9 m form the numerical barrier mask and are excluded from active-cell statistics. This threshold is a modelling proxy, not an independently verified building inventory; it may also include elevated terrain or domain-exterior cells.

Paired local A/B/C simulations were available for events 1, 20 and 65-70. These eight events were defined by the availability of complete locally generated matched calculations, not by their DrainLite error. Of 17 public event directories with readable 20 m rainfall and MIKE arrays, 8 therefore entered the paired experiment. Event 79 was retained in the inventory as unreadable and excluded. Figure 9 and Appendix B place the selected events within the rainfall and reference-severity distribution. The design evaluates event transfer on one fixed terrain and network, not transfer between cities or sewer layouts.



Rainfall volume was recomputed from every raw five-minute field. Table 1 uses the full rectangular sum, whereas the event-inclusion plot uses the mean over active cells. These are different spatial supports. For event65, the active-cell mean is 24.61517 mm and the rectangle mean is 24.17178 mm; the rectangular total is 2165791.9 m3. Multiplying the active mean by 89.6 km2 is therefore invalid. The archived 576-step audit separates active-cell rainfall, reassigned barrier-cell rainfall and reconstructed model injection. Surveyed building and exterior areas cannot be separated with this threshold mask.

### 2.2 Construction and audit of the conceptual network

Surveyed sewer records were unavailable. Road-aligned candidates derived from OpenStreetMap geometry were converted to a directed network and conditioned for hydraulic use. Disconnected fragments and duplicate endpoint pairs were removed. Terrain-informed invert levels were assigned while maintaining cover, positive conduit slope and a downstream path to a receiving interface. The accepted network contains 2,276 junctions, 2,276 conduits and 221 NORMAL receiving interfaces. Pipe diameters range from 0.8 to 1.2 m; slopes range from 0.000499 to 0.100 m m-1.

Every junction reaches an outfall in the directed graph. The independent parser found no isolated junction, cycle, duplicate endpoint pair, reverse-slope conduit or pipe crown above a junction rim. These checks establish internal topological and geometric consistency. They do not establish correspondence with Shenzhen's surveyed municipal network. OpenStreetMap-derived geometry is attributed to OpenStreetMap contributors under the Open Database License.

![Figure 2. Conceptual-network maps. (a) Terrain, active-mask edge, selected link directions and outfall-connected terminal junctions; these are not surveyed river mouths or actual outfall coordinates. (b) Conduit diameter. (c) Junction degree. OpenStreetMap contributors provided the road geometry. Numeric slope and cover diagnostics are retained in the evidence supplement.](figures/fig02_network_audit.png)

**Figure 2. Conceptual-network maps. (a) Terrain, active-mask edge, selected link directions and outfall-connected terminal junctions; these are not surveyed river mouths or actual outfall coordinates. (b) Conduit diameter. (c) Junction degree. OpenStreetMap contributors provided the road geometry. Numeric slope and cover diagnostics are retained in the evidence supplement.**



Re-parsing all conduit inverts and offsets found 0 zero or negative slopes. There are 626 conduits (27.50%) within 0.0005 +/- 0.000002 m/m. This geometric count identifies concentration near the design floor; individual pre-adjustment slopes were not retained. None of the 221 synthetic NORMAL outfalls has its own coordinate record. Figure 2 marks their connecting terminal junctions instead. NORMAL is a normal-depth boundary, not an observed river or tidal hydrograph. The 0.8-1.2 m diameter range defines a controlled conceptual system, without a claim of survey-based calibration.

### 2.3 Matched physical calculations

Surface flow was solved with Itzï 25.4 using its damped partial-inertia formulation. The minimum water depth was 0.001 m, the Courant number 0.7, the partial-inertia coefficient 0.9 and the maximum surface step 1 s. The rectangular outer boundary was closed. Active cells used Manning's n = 0.015 except where stated below. Cells in the threshold-derived barrier mask were set to 50 m. Their rainfall was redistributed uniformly to active cells. This operation conserves the rectangular-domain rainfall total but changes its spatial allocation and cannot by itself identify real roof runoff. A uniform 1 mm h-1 effective loss represented unresolved interception, infiltration and other continuing losses; it was not fitted as a measured soil parameter.

The one-dimensional network was solved by SWMM 5.2.4 through PySWMM 2.1.0. Dynamic-wave routing used a 0.5 s maximum step, variable-step factor 0.2, 50 trials, slot surcharge representation and 0.0015 m head tolerance. SWMM node ponding was disabled because surface storage and surcharge exchange were represented by Itzï. The coupling relaxation and damping factors were 0.8 and 0.5. The executable entry point now rejects unreviewed Itzï, PySWMM and SWMM-toolkit versions before opening the native engine.

Scenario A used surface flow with Manning's n = 0.015. Scenario B remained surface-only but applied n = 0.012 in the same 3 x 3 inlet neighbourhoods used during coupling. Scenario C retained the B roughness field and activated Itzï's native DrainageSimulation interface to exchange water bidirectionally with SWMM. Hence B-A measures the local roughness perturbation, while C-B measures the pipe-coupling response:

<p style="text-align:center"><i>r</i>(t,x) = <i>h</i><sub>C</sub>(t,x) - <i>h</i><sub>B</sub>(t,x). &nbsp;&nbsp; (1)</p>

Negative r denotes local drainage relative to B; positive r is retained because network routing and surcharge may locally increase surface depth.

### 2.4 Physical-run acceptance criteria

The repository applies the following acceptance conditions; their historical pre-registration has not been independently established. Each array must be finite and have shape 72 x 400 x 560. Every junction requires a directed outfall path. Absolute SWMM flow-routing continuity error must be at most 2%, non-converging routing steps at most 2%, and absolute combined surface-network mass error at most 0.5% of event rainfall. SWMM flooding loss, warning count and error count must be zero. A signed continuity value is an accounting residual; acceptance uses its magnitude.

**Table 1. Physical quality of the eight paired events.**

| Event | Rain volume (10^6 m3) | \|B-A\| MAE (mm) | \|C-B\| MAE (mm) | Mean C-B (mm) | Routing continuity (%) | Non-converging (%) | Combined mass error (% rain) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event1 | 2.079 | 0.168 | 7.200 | -5.952 | -0.083 | 0.020 | -0.100 | Accepted |
| event20 | 1.673 | 0.101 | 3.322 | -2.673 | -0.178 | 0.000 | -0.032 | Accepted |
| event65 | 2.166 | 0.170 | 7.696 | -6.403 | -0.111 | 0.000 | -0.017 | Accepted |
| event66 | 2.305 | 0.181 | 8.431 | -7.086 | -0.101 | 0.000 | -0.015 | Accepted |
| event67 | 2.670 | 0.255 | 12.631 | -10.928 | -0.066 | 0.010 | -0.111 | Accepted |
| event68 | 2.552 | 0.236 | 11.914 | -10.101 | -0.060 | 0.040 | -0.097 | Accepted |
| event69 | 2.422 | 0.224 | 10.912 | -9.420 | -0.068 | 0.010 | -0.101 | Accepted |
| event70 | 2.578 | 0.240 | 12.023 | -10.319 | -0.063 | 0.010 | -0.098 | Accepted |

![Figure 3. Physical-run quality. Panel (a) reports routing continuity and non-converging steps; the 2% acceptance limits are specified in Section 2.4 rather than drawn on this magnified axis. Panel (b) compares the roughness effect B-A with the sewer effect C-B. Panel (c) shows final surface storage under B and C. Panel (d) gives final-volume discrepancy relative to MIKE. The logarithmic scale in panel (b) is required because the two effects differ by nearly two orders of magnitude.](figures/fig03_physical_quality.png)

**Figure 3. Physical-run quality. Panel (a) reports routing continuity and non-converging steps; the 2% acceptance limits are specified in Section 2.4 rather than drawn on this magnified axis. Panel (b) compares the roughness effect B-A with the sewer effect C-B. Panel (c) shows final surface storage under B and C. Panel (d) gives final-volume discrepancy relative to MIKE. The logarithmic scale in panel (b) is required because the two effects differ by nearly two orders of magnitude.**

### 2.5 Predictor fields and signed residual

The network was rasterised on the 20 m model grid. Eight mask and topology predictors describe inlet presence, pipe presence, inlet count, segment count, degree and local pipe/inlet densities. Five hydraulic predictors describe diameter, slope, full-flow capacity, cover and local capacity density. No row or column coordinate, target-side SWMM head, target-side pipe flow, synthetic outfall mask or distance-to-outfall field entered the principal models.

Ten non-network predictors describe the current B depth, current and cumulative rainfall, elevation, terrain slope, normalised time, sine and cosine time encodings, and 3 x 3 and 7 x 7 local depth means. The local means are mask-aware, so building and inactive cells do not depress neighbourhood values. The target is r in millimetres, and the corrected field is

<p style="text-align:center"><i>h</i><sub>DL</sub>(t,x) = max[<i>h</i><sub>B</sub>(t,x) + <i>r</i><sub>theta,mm</sub>(t,x)/1000, 0]. &nbsp;&nbsp; (2)</p>

No residual clipping is used in the primary experiment. The final maximum enforces non-negative water depth; the affected cell-time fraction is reported with the clipping sensitivity rather than being applied silently.

### 2.6 Whole-event validation and sampling repeats

Eight outer folds were formed by withholding one complete event. A spatiotemporal prior was calculated at every time and cell from the other seven events. For fitting rows, the prior also excluded the row's own event, so no target from that event contributed to its prior. MIKE arrays were never used in fitting, weighting, feature construction or model selection.

HistGradientBoostingRegressor used squared-error loss, learning rate 0.05, 31 leaves, L2 regularisation 0.01 and a pre-specified 140 boosting iterations. The fixed iteration count removes sensitivity to a single inner validation event. Each training event contributed 450 uniformly selected active cells per time step. The principal prior-plus-dynamic and full-network models were repeated for five pre-specified sampling seeds. Subgroup models and spatial controls used seed 1907 to limit redundant full-domain inference. Every held event was then predicted over all active cells and all 72 times.

### 2.7 Network controls and conditional evaluation

The final prior-plus-dynamic-plus-network model was subjected to three destructive controls. Primitive network fields were shifted by 20, 40, 80 and 160 m in each cardinal direction; 20 x 20-cell blocks were jointly shuffled in ten replicates; and all primitive network values were jointly permuted among active cells in five replicates. Derived density fields were regenerated after each transformation. A zero-static-fields control removed the explicit network fields but retained the event-excluded fixed-network prior. The latter is therefore a test of added static fields, not a claim that the complete model is network-free.

Errors were calculated over the full active domain, wet cells, cell-times with |C-B| above 1, 5 and 10 mm, negative and positive residual subsets, and bands within 20, 20-60, 60-100 and more than 100 m from inlets and pipes. This prevents dry or weak-effect cells from dominating interpretation.

### 2.8 Performance and computational metrics

Mean absolute error (MAE), root-mean-square error (RMSE), bias and critical success index (CSI) at 0.03 and 0.15 m were calculated on complete held events. Peak-map MAE compares the maximum depth at each cell. Global-peak depth and timing concern only the single largest depth. Surface volume and inundated-area errors were evaluated at every time and at six hours. Metrics were macro-averaged over events.

Runtime measurement includes uncached dynamic neighbourhood computation, feature construction, input assembly, estimator prediction and non-negativity post-processing. Static descriptors, the fitted model and the training-derived prior are prepared beforehand; input/output disk access is excluded. The workflow comparison sums this in-memory correction time and the recorded historical B simulation time and contrasts the sum with the recorded C time. It is not a newly timed integrated execution of both components.

## 3. Results

### 3.1 Coupling produced a resolved and numerically controlled response

All eight events met the physical acceptance conditions. Mean routing continuity error was -0.091%, and mean non-converging-step frequency was 0.011%. The mean |C-B| response was 9.266 mm, compared with 0.197 mm for |B-A|. The 47.1-fold difference shows that the learning target is dominated by sewer exchange rather than the inlet-neighbourhood roughness adjustment. Coupling reduced six-hour surface storage in every event, with a mean signed C-B depth of -7.860 mm.

![Figure 4. Active-domain surface-water volume for eight events. Grey is matched surface B, orange is coupled C and black is MIKE. Pale blue rainfall uses the right axis. The left axis is common in units but each right rainfall axis is event-scaled. A declining C volume after rainfall indicates drainage from the surface even where a single-cell maximum continues to rise.](figures/fig04_eight_event_hydrographs.png)

**Figure 4. Active-domain surface-water volume for eight events. Grey is matched surface B, orange is coupled C and black is MIKE. Pale blue rainfall uses the right axis. The left axis is common in units but each right rainfall axis is event-scaled. A declining C volume after rainfall indicates drainage from the surface even where a single-cell maximum continues to rise.**

![Figure 5. Cell-wise temporal maxima for event68. Panels (a-d) show max_t(MIKE), max_t(A), max_t(B) and max_t(C). Panel (e) is max_t(C)-max_t(B); panel (f) is max_t(B)-max_t(A). These differences do not measure instantaneous exchange or max_t(C-B). Supplementary Figure S1 compares the definitions at a common time. Display saturation is separate from untruncated statistics.](figures/fig05_event68_physical_maps.png)

**Figure 5. Cell-wise temporal maxima for event68. Panels (a-d) show max_t(MIKE), max_t(A), max_t(B) and max_t(C). Panel (e) is max_t(C)-max_t(B); panel (f) is max_t(B)-max_t(A). These differences do not measure instantaneous exchange or max_t(C-B). Supplementary Figure S1 compares the definitions at a common time. Display saturation is separate from untruncated statistics.**

### 3.2 The fixed-network prior accounts for most of the predictable structure

The matched surface field had 9.266 mm MAE relative to C. Dynamic predictors without a prior reduced the error to 5.273 mm. The event-excluded spatiotemporal prior was stronger, at 3.226 mm, showing that much of the drainage response recurred at the same cells and times on this fixed system. Adding the current surface-rainfall state to the prior reduced MAE to 2.217 mm. The full model reached 2.169 mm and RMSE 9.558 mm.

**Table 2. Whole-event model comparison; uncertainties refer to sampling seeds.**

| Model | Seeds | MAE (mm) | Seed SD | RMSE (mm) | CSI 0.03 m | CSI 0.15 m | Peak-map MAE (mm) | Final-volume error (10^3 m3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matched surface B | 5 | 9.266 | 0.000 | 41.421 | 0.889 | 0.633 | 16.835 | 699.5 |
| Spatiotemporal prior | 5 | 3.226 | 0.000 | 13.621 | 0.898 | 0.867 | 5.403 | 129.7 |
| Dynamic predictors, no prior | 1 | 5.273 | Not estimated | 19.635 | 0.896 | 0.731 | 8.284 | 40.5 |
| Prior + dynamic | 5 | 2.217 | 0.007 | 9.907 | 0.925 | 0.905 | 3.567 | 32.0 |
| Prior + dynamic + masks | 1 | 2.172 | Not estimated | 9.429 | 0.929 | 0.906 | 3.538 | 27.3 |
| Prior + dynamic + hydraulics | 1 | 2.193 | Not estimated | 9.477 | 0.927 | 0.905 | 3.553 | 27.3 |
| Prior + dynamic + all network fields | 5 | 2.169 | 0.008 | 9.558 | 0.929 | 0.906 | 3.516 | 29.1 |

![Figure 6. Held-event fields at maximum C surface volume; each row labels its exact time. All rows share fixed ranges: B depth 0-0.5 m and signed residual/error -200 to 200 mm. Saturation fractions and extrema are reported in the evidence supplement; statistics use untruncated arrays.](figures/fig06_fixed_network_residual_maps.png)

**Figure 6. Held-event fields at maximum C surface volume; each row labels its exact time. All rows share fixed ranges: B depth 0-0.5 m and signed residual/error -200 to 200 mm. Saturation fractions and extrema are reported in the evidence supplement; statistics use untruncated arrays.**

![Figure 7. Performance by predictor group. Circles and error bars show five-sampling-seed means and SD; diamonds denote a single fitted model or a deterministic baseline and have no estimated repeat uncertainty. Panels show MAE, RMSE and CSI. Single-seed subgroup estimates should not be ranked as five-seed averages.](figures/fig07_skill_and_sampling_seeds.png)

**Figure 7. Performance by predictor group. Circles and error bars show five-sampling-seed means and SD; diamonds denote a single fitted model or a deterministic baseline and have no estimated repeat uncertainty. Panels show MAE, RMSE and CSI. Single-seed subgroup estimates should not be ranked as five-seed averages.**

### 3.3 Static network information is detectable but secondary

Across five sampling seeds, adding all static network fields to the prior-plus-dynamic model changed MAE by 0.048 +/- 0.002 mm. This increment is small relative to the gains from the prior and dynamic state. It is also event dependent, as shown by the event-by-seed matrix in the evidence supplement.


The final-model controls nevertheless show that the increment is tied to spatial organisation. A 20 m shift increased MAE by 0.044 mm on average, and the penalty at 160 m was 0.139 mm. Zeroing the static fields increased MAE by 0.196 mm; block shuffling and group permutation increased it by 0.192 and 0.161 mm, respectively. These controls do not demonstrate transfer to another sewer layout, because the spatiotemporal prior still encodes the average response of the original network.

**Table 3. Inference-time perturbations of the fixed fitted hybrid; not retraining gains.**

| Inference-only perturbation | Runs | Mean penalty (mm) | SD across runs (mm) |
| --- | --- | --- | --- |
| zero_static_network_fields | 8 | 0.1962 | 0.1074 |
| shift_20m | 32 | 0.0438 | 0.0109 |
| shift_40m | 32 | 0.0617 | 0.0169 |
| shift_80m | 32 | 0.0949 | 0.0290 |
| shift_160m | 32 | 0.1389 | 0.0503 |
| block_shuffle | 80 | 0.1924 | 0.0836 |
| group_permutation | 40 | 0.1610 | 0.0473 |



The mean network increment is 0.047773 mm. Its SD across event-specific seed means is 0.037333 mm, compared with 0.001819 mm across seed-specific event means. An event-unit percentile bootstrap gives [0.023754, 0.071270] mm (20,000 resamples). This interval assumes exchangeable events; observation dates and storm-family membership have not been verified. It does not establish an engineering-significant gain or generalisation beyond this rainfall distribution.

### 3.4 Improvements persist where drainage is active

Full-domain means can be dominated by dry or weak-effect cells. Conditional evaluation gives a more demanding view. In the canonical-seed experiment, the full hybrid reduced MAE not only over all active cell-times but also for wet cells, locations with |C-B| above 5 mm, net-drainage cells, positive residual cells and the immediate inlet and pipe neighbourhoods. Errors remain larger in the strong-effect subsets, which identifies the residual transitions and positive-residual patches as the main unresolved structures.


![Figure 8. Conditional MAE. Panel (a) separates the full domain from wet cells. Panel (b) progressively restricts evaluation to stronger drainage effects. Panel (c) groups cells by distance to an inlet. The change in vertical scale between panels is intentional: strong-effect and near-inlet subsets are harder than the domain average.](figures/fig09_conditioned_performance.png)

**Figure 8. Conditional MAE. Panel (a) separates the full domain from wet cells. Panel (b) progressively restricts evaluation to stronger drainage effects. Panel (c) groups cells by distance to an inlet. The change in vertical scale between panels is intentional: strong-effect and near-inlet subsets are harder than the domain average.**

### 3.5 MIKE comparison depends on the quantity being evaluated

MIKE was not generated with the same disclosed conceptual network and is not an independent validation target for C. The comparison is therefore descriptive. Matched surface B has the lowest full space-time MAE to MIKE, whereas C and DrainLite substantially reduce six-hour volume discrepancy. The opposite ordering of these metrics means that coupling brings aggregate water volume closer to this external simulation without reproducing MIKE's complete spatial field.

**Table 4. External comparison with MIKE.**

| Field | MAE (mm) | RMSE (mm) | CSI 0.15 m | Peak-map MAE (mm) | Final-volume error (10^3 m3) |
| --- | --- | --- | --- | --- | --- |
| Matched surface B | 18.023 | 50.905 | 0.467 | 27.083 | 982.1 |
| Coupled label C | 21.258 | 62.509 | 0.311 | 34.608 | 310.6 |
| DrainLite full hybrid | 20.994 | 61.708 | 0.306 | 33.563 | 290.4 |



### 3.6 Recorded computation time and post-processing

Feature construction, input assembly, model prediction and post-processing together required 23.6 s per event on average for the canonical full hybrid. Adding recorded B time to in-memory correction time gives 297.7 s, compared with 3817.6 s for C. The ratio of these means is 12.82; the mean of eight event-wise ratios is 13.10 (event SD 2.84). Neither number includes disk I/O, prior preparation or a newly timed integrated run. It also shows why estimator-only timing would overstate acceleration.

The primary result uses no residual clipping. Sensitivity calculations at +/-500 and +/-1200 mm quantify the effect of optional bounds, and the recorded non-negativity count shows how often Eq. (2) changes an otherwise negative predicted depth. Event selection is shown alongside these computational diagnostics to keep performance claims connected to the available forcing set.

**Table 5. Runtime scope and post-processing, canonical seed 1907.**

| Quantity | Value | Scope |
| --- | --- | --- |
| Feature / assembly / estimator / postprocess (s) | 3.45 / 1.56 / 18.36 / 0.22 | Canonical seed 1907; mean of eight events |
| B / correction / B+correction / C (s) | 274.03 / 23.64 / 297.67 / 3817.57 | Historical physical runs + in-memory correction; excludes I/O |
| Mean event speed ratio; ratio of means | 13.098; 12.825 | Event-ratio SD 2.841; not integrated deployment timing |
| Residual bound none: MAE / depth-clamped cells | 2.1791 mm / 1.829% | Canonical seed only; fraction over active cell-times |
| Residual bound 500.0: MAE / depth-clamped cells | 2.2104 mm / 1.821% | Canonical seed only; fraction over active cell-times |
| Residual bound 1200.0: MAE / depth-clamped cells | 2.1796 mm / 1.829% | Canonical seed only; fraction over active cell-times |

![Figure 9. Event inclusion within the 17 locally readable rainfall/reference cases. The eight paired events tend to have higher rainfall; event77 has the largest reference maximum despite not being paired. This local subset is not the entire 80-event public benchmark.](figures/fig11_event_selection.png)

**Figure 9. Event inclusion within the 17 locally readable rainfall/reference cases. The eight paired events tend to have higher rainfall; event77 has the largest reference maximum despite not being paired. This local subset is not the entire 80-event public benchmark.**

### 3.7 Relation to the public LarNO checkpoint

The public LarNO checkpoint was executed independently for event 68. Its unclamped MAE was 2.409 mm, RMSE 5.119 mm and CSI at 0.15 m 0.955. Negative raw predictions comprised 18.15% of cells, so both raw and clamped statistics are retained.

![Figure 10. Independent public LarNO comparison for event68 at 3.25 h (frame 39, the MIKE global-depth peak). The top row shows MIKE, LarNO and signed error with ranges reaching the frame extrema, without high-end saturation. The lower maps isolate cells exceeding 0.5 m in either field; the last panel compares domain-maximum depth through time. Event-wide maxima are 2.795 m and 2.826 m. At the displayed time, 9.98% of all-grid LarNO predictions are negative; these are zeroed only for the depth display, not for the signed-error statistics. This is an upstream checkpoint audit, not a DrainLite result.](figures/fig12_larno_reproduction.png)

**Figure 10. Independent public LarNO comparison for event68 at 3.25 h (frame 39, the MIKE global-depth peak). The top row shows MIKE, LarNO and signed error with ranges reaching the frame extrema, without high-end saturation. The lower maps isolate cells exceeding 0.5 m in either field; the last panel compares domain-maximum depth through time. Event-wide maxima are 2.795 m and 2.826 m. At the displayed time, 9.98% of all-grid LarNO predictions are negative; these are zeroed only for the depth display, not for the signed-error statistics. This is an upstream checkpoint audit, not a DrainLite result.**

DrainLite was not appended to the public checkpoint. That checkpoint was trained against MIKE fields generated by MIKE Plus 2023 with 1D-2D drainage coupling (Cao et al., 2026, Section 4.1); applying C-B to it could count drainage twice. A defensible end-to-end LarNO-DrainLite calculation requires a LarNO model trained to drainage-free targets or paired neural-operator targets under matched network interventions.

### 3.8 Paired physical sensitivities for event 68

Five alternatives were calculated with paired B and C runs, changing one setting at a time: effective loss of 0 or 2 mm h-1, nearest-active-cell redistribution of building rainfall, exclusion of building rainfall, or inlet-neighbourhood Manning coefficient 0.015. The baseline retained 1 mm h-1 loss, global redistribution and coefficient 0.012. All alternatives produced finite 72-frame fields. Routing continuity magnitudes were below 0.2%; these diagnostics alone do not establish physical validity.

The mean drainage effect was 11.914 mm at baseline, 12.993 mm without effective loss and 10.874 mm at 2 mm h-1. Nearest-cell redistribution reduced it to 7.799 mm, and exclusion to 3.465 mm; the latter also removes rainfall volume and cannot isolate spatial routing alone. Changing inlet roughness gave 11.894 mm. Thus the existence of a drainage response persisted, while its magnitude depended substantially on rainfall treatment. C-to-MIKE MAE increased from 24.023 to 35.080 mm under nearest-cell redistribution, showing that local redistribution does not automatically improve agreement. This single-event experiment supports a sensitivity statement, not calibration or robustness of the trained residual model across physical configurations.


![Figure 11. Event68 physical settings shown as a ratio heatmap. Each cell prints the original measured quantity and its ratio to baseline. Colour compares each column with its own baseline, not different physical units. Exclude changes rainfall volume as well as allocation. No surrogate was retrained under these alternative settings.](figures/fig13_physical_sensitivity.png)

**Figure 11. Event68 physical settings shown as a ratio heatmap. Each cell prints the original measured quantity and its ratio to baseline. Colour compares each column with its own baseline, not different physical units. Exclude changes rainfall volume as well as allocation. No surrogate was retrained under these alternative settings.**

## 4. Discussion

### Physical evidence and unresolved uncertainty

The numerical label, the surrogate approximation and the incremental network predictors answer different questions. Small closure errors support internal accounting; they do not validate the conceptual network against actual drainage observations. Recorded exchanges are cumulative at 30-minute output times. The sign audit distinguishes water entering SWMM from return flow, but the available archives do not contain every routing-step storage and outfall term. A complete time-resolved joint ledger and a routed single-inlet/single-pipe benchmark remain required. The supplementary native-function check uses a prescribed-head node test double and does not substitute for that benchmark.

The selected events have mean active-cell rainfall 26.02 mm, versus 19.87 mm for the nine other readable local cases. Their missing paired labels reflect work not yet performed, rather than an established inability to simulate them. Event numbering is not evidence of independent weather. Validation is consequently conditional on these eight events.

Positive C-B is a difference between two evolving surface solutions. It can reflect redistribution as well as actual node return flow, so it must not be labelled surcharge without a local exchange record. The full model barely changes the positive-residual error relative to the dynamic hybrid. Static fields and contemporaneous B do not uniquely specify the internal network state. Non-negative depth also does not enforce water conservation; volume-error trajectories in the supplement quantify the remaining discrepancy.

The spatiotemporal prior encodes fixed location and time even though explicit coordinates are absent. Cross-configuration transfer, a coarser prior, shifted rainfall timing, multi-event roof-runoff alternatives and pipe-capacity/tailwater perturbations have not been tested here. These experiments are needed before claims of physical robustness or network intervention response.


### 4.1 What is learned on a fixed network

The results separate three sources of predictive skill. The event-excluded prior captures the repeatable cell-time response of one terrain-network system. Current surface depth and rainfall account for much of the event-specific departure from that response. Static pipe fields provide a smaller final adjustment. This hierarchy explains why a low-compute model can emulate C accurately while still offering limited evidence for network-conditioned generalisation.

The final-hybrid displacement and shuffle controls strengthen the interpretation of the static increment. Unlike earlier controls applied to a no-prior model, they perturb the exact estimator used for the main result. These tests expose sensitivity to inference-time distribution shifts; their penalties are not estimates of the benefit of adding network predictors during training. Yet the prior remains fixed during those tests. The experiment therefore supports aligned static information within the original system, not counterfactual prediction after redesigning that system.

### 4.2 Error structure and physical meaning

Conditioned metrics show why the 2 mm-scale domain average should not be read as uniform accuracy. Where |C-B| exceeds 5 or 10 mm, the residual is sharper and less frequent, and the error rises. Immediate inlet areas are similarly difficult because the exchange is concentrated within a small number of cells. Positive C-B patches are not necessarily numerical failures: bidirectional routing can return water or displace storage locally while total surface volume falls. Keeping a signed target is therefore essential.

### 4.3 External reference and model scope

The MIKE comparison constrains interpretation rather than certifying calibration. The conceptual network and undisclosed MIKE drainage representation differ in topology, inlet density, capacity and boundary treatment. Better final-volume agreement under C indicates lower storage discrepancy relative to MIKE, whereas poorer pixel-time MAE shows that the spatial redistribution is not the same. Both observations can be true.

Four boundaries remain. First, the sewer is road-derived rather than surveyed. Second, eight rainfall events share one terrain and network. Third, the surface edge is closed, building rainfall is globally redistributed and the 1 mm h-1 loss is an effective assumption. The event68 paired sensitivity experiment below quantifies their influence for one event; it does not establish robustness across events or retrained surrogates. Fourth, DrainLite requires the contemporaneous B field and should be described as a residual emulator, not an independent forecast model.

### 4.4 Implications for further development

The next physical dataset should vary network configuration as well as rainfall. Holding out a pipe-capacity, inlet-density or outlet-layout scenario would test whether the static fields act as intervention variables. Such an experiment should omit any prior derived from the target network. Dynamic node head, conduit flow, fullness and surcharge could then be rasterised as causal inputs if they are available at inference, or predicted by a separate network-state model. Only after a drainage-free LarNO upstream model is available should the two components be assessed as an end-to-end rainfall-to-flood system.

## 5. Conclusions

A connected conceptual sewer network was coupled through Itzï's native SWMM interface and evaluated using matched surface controls. Across eight events, |C-B| averaged 9.266 mm, compared with 0.197 mm for the roughness-only difference. All events met the repository's topology, continuity, convergence and combined mass-balance criteria.

Whole-event validation showed that the event-excluded spatiotemporal prior reduced MAE to 3.226 mm. Adding current surface and rainfall information reduced it to 2.217 mm, and adding all static network fields yielded 2.169 mm. The static increment was 0.048 +/- 0.002 mm across five sampling seeds. Final-model perturbations reveal sensitivity to input alignment, while retrained ablations quantify the smaller incremental predictive contribution, while conditional metrics show that improvements persist in wet, near-inlet and strong-effect cells.

The evidence supports rapid residual emulation for rainfall events on this fixed conceptual network. It does not establish generalisation to another pipe layout or an end-to-end LarNO-DrainLite predictor. That distinction is central to both the scientific contribution and the next experimental step.

## Data availability

Public LarNO rainfall, terrain, MIKE arrays and the upstream checkpoint retain their original provenance. Locally calculated A/B/C labels and static descriptors are provided in the V3 canonical dataset. Road-derived geometry requires attribution to © OpenStreetMap contributors and is subject to the Open Database License. The repository licence map identifies materials whose downstream redistribution terms remain to be confirmed with the source authors.

## Code availability

Network construction, physical simulation, residual fitting, controls, figures and read-only verification are available at https://github.com/Coucou2016/20260519-LarNO-DrainLite. The repository records exact package versions, Git LFS objects, SHA-256 manifests and the distinction between regeneration and verification.

## Declaration of competing interests

The declaration of competing interests requires confirmation by all authors before submission.

## Acknowledgements and author information

Author names, affiliations, contributions and funding information are **to be supplied before submission**.

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

Courty, L.G., Pedrozo-Acuna, A., Bates, P.D., 2017. Itzi (version 17.1): an open-source, distributed GIS model for dynamic flood simulation. Geoscientific Model Development 10, 1835-1847. https://doi.org/10.5194/gmd-10-1835-2017.

Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Annals of Statistics 29, 1189-1232.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations.

Rossman, L.A., Simon, M.A., 2022. Storm Water Management Model User's Manual Version 5.2. U.S. Environmental Protection Agency, Washington, DC.



## Reproducibility record

The V5 figure and evidence revision uses frozen V3 physical arrays and V4 seed-1907/full-five-seed outputs from source commit `d5fefaa0cf7a31cf7d29830ed5b070e1d2dac3a2`. The V3 tuned/clipped estimator is historical, not the current primary model. Current fitting uses `run_final_hybrid_major_revision.py` with fixed 140 iterations and no residual clipping. `audit_reviewer_v5_evidence.py` recomputes rainfall, signed residual, volume and event-seed diagnostics. V5 figure sources and input paths are recorded in the companion audit. No physical array was replaced for this revision.
