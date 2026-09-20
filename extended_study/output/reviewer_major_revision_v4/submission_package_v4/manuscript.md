# DrainLite: leakage-controlled residual emulation of a fixed conceptual sewer network for urban flood modelling

## Highlights

- Matched surface controls isolate sewer exchange from inlet-neighbourhood roughness.
- A road-aligned conceptual network connects 2,276 junctions to 221 receiving interfaces.
- Whole-event validation reduces coupled-label depth error from 9.266 to 2.169 mm.
- A training-event prior explains most of the repeatable fixed-network response.
- Static network fields add 0.048 +/- 0.002 mm MAE improvement across five sampling seeds.

## Abstract

Urban flood surrogates commonly reproduce the output of a coupled hydrodynamic model without exposing the drainage system as an explicit condition. This study considers a narrower problem: whether the surface-depth change caused by one specified sewer network can be emulated as a lightweight correction to a drainage-free hydrodynamic state. A road-aligned conceptual network was constructed for the 20 m Shenzhen benchmark and coupled bidirectionally to the two-dimensional Itzï solver through the Storm Water Management Model. Eight six-hour rainfall events were calculated under an original surface configuration (A), a surface configuration with coupling-neighbourhood roughness but no pipe exchange (B), and native Itzï-SWMM coupling (C). The signed learning target was C-B.

DrainLite combines an event-excluded spatiotemporal residual prior with current rainfall, the contemporaneous B depth field, terrain and rasterised network attributes. Complete events were held out in turn. Five independent spatial sampling seeds were used for the principal dynamic and full-network models, while model complexity was fixed before fitting. Across 60,783,552 held-out active cell-times, B differed from C by 9.266 mm mean absolute error. The spatiotemporal prior attained 3.226 mm, the prior-plus-dynamic model attained 2.217 mm, and the full hybrid attained 2.169 mm. The static-network increment averaged 0.048 mm across sampling seeds and was smaller than the contribution of the prior and dynamic state. Displacing, block-shuffling or jointly permuting the network fields increased the error of the final hybrid, indicating that the small increment depended on spatial alignment.

The coupled calculations met the predefined topology, continuity and mass-balance conditions. Their mean routing continuity error was -0.091%, and their mean signed depth change was -7.860 mm. Comparison with public MIKE fields was metric dependent: coupling reduced final-volume error but increased full space-time depth error. MIKE was therefore retained as a descriptive external comparator, not a training label or independent validation dataset. The resulting method is a fixed-network residual emulator that requires the contemporaneous surface-only field; it is not an end-to-end rainfall-to-flood predictor and does not establish transfer to an unseen sewer layout.

**Keywords:** urban pluvial flooding; conceptual sewer network; Itzï; Storm Water Management Model; residual emulation; whole-event validation

## 1. Introduction

Short-duration urban flooding reflects the joint influence of rainfall, buildings, surface conveyance, local storage and underground drainage. Two-dimensional surface models resolve overland propagation, whereas one-dimensional network models describe flow through junctions and pipes. Their exchange through inlets can reduce surface storage, redistribute water and return surcharge to the street. Omitting this exchange does not make a surface calculation dynamically invalid, but it changes the system being represented.

The computational cost of coupled simulation remains a practical barrier to event ensembles and rapid forecasting. Neural operators offer a complementary route by learning mappings between forcing fields and hydrodynamic solutions. The Large-scale Latent Autoregressive Neural Operator (LarNO) was developed for large-area urban flood prediction and demonstrated zero-shot evaluation across grid resolutions using a public Shenzhen benchmark (Cao et al., 2026). Its released inputs and checkpoint permit reproduction of the reference mapping, but the municipal drainage inventory represented in the MIKE calculations is not released as an independently changeable model input.

One response would be to retrain the complete neural operator with additional pipe channels. That strategy requires many matched network simulations and substantially more hardware than is available in the present study. A residual formulation provides a lower-cost alternative. If a drainage-free model supplies a surface state, a second model can estimate only the depth difference induced by a prescribed sewer system. This decomposition is useful only if the target is physically identifiable and the validation prevents information from the tested event entering the predictor.

Fixed terrain and a fixed network create a further difficulty. Their drainage response may recur at the same cells and times across events. A model can therefore appear skilful by recovering a climatological template, even if it makes little use of rainfall or pipe attributes. Random cell-wise splitting compounds this problem, because adjacent samples from one event enter both fitting and evaluation. A suitable test must retain entire rainfall events, compare against an event-excluded spatiotemporal prior, and perturb the network fields in the final model rather than in an earlier surrogate.

Here, DrainLite is formulated as a correction layer for a fixed conceptual network. Three questions are examined. First, does a connected road-aligned network produce a numerically controlled response that can be separated from ancillary roughness changes? Second, how much of this response is explained by a cross-event prior and by the current surface-rainfall state? Third, after those predictors are known, is the remaining contribution of correctly aligned static network information detectable? MIKE and the public LarNO checkpoint are analysed outside the residual-fitting pathway to clarify what the experiment does, and does not, establish.

![Figure 1. Experimental design. Scenarios B and C share their surface parameterisation, so C-B isolates the bidirectional sewer exchange. DrainLite receives the contemporaneous B field and predicts a signed residual. The held event is excluded from fitting and prior construction. MIKE and LarNO are external to the principal learning pathway.](figures/fig01_workflow.png)

**Figure 1. Experimental design. Scenarios B and C share their surface parameterisation, so C-B isolates the bidirectional sewer exchange. DrainLite receives the contemporaneous B field and predicts a signed residual. The held event is excluded from fitting and prior construction. MIKE and LarNO are external to the principal learning pathway.**

## 2. Data and methods

### 2.1 Study domain and event inclusion

The public 20 m benchmark comprises a 400 x 560 rectangular grid, corresponding to 8.0 x 11.2 km. The reporting mask contains 105,527 active cells (42.21 km2). Each event contains 72 five-minute fields spanning six hours. Water depth is stored in metres and rainfall as millimetres per five-minute interval. High building cells remain as hydraulic barriers but are excluded from active-cell performance statistics.

Paired local A/B/C simulations were available for events 1, 20 and 65-70. These eight events were defined by the availability of complete locally generated matched calculations, not by their DrainLite error. Of 17 public event directories with readable 20 m rainfall and MIKE arrays, 8 therefore entered the paired experiment. Event 79 was retained in the inventory as unreadable and excluded. Figure 11d and Appendix B place the selected events within the rainfall and reference-severity distribution. The design evaluates event transfer on one fixed terrain and network, not transfer between cities or sewer layouts.

### 2.2 Construction and audit of the conceptual network

Surveyed sewer records were unavailable. Road-aligned candidates derived from OpenStreetMap geometry were converted to a directed network and conditioned for hydraulic use. Disconnected fragments and duplicate endpoint pairs were removed. Terrain-informed invert levels were assigned while maintaining cover, positive conduit slope and a downstream path to a receiving interface. The accepted network contains 2,276 junctions, 2,276 conduits and 221 NORMAL receiving interfaces. Pipe diameters range from 0.8 to 1.2 m; slopes range from 0.000499 to 0.100 m m-1.

Every junction reaches an outfall in the directed graph. The independent parser found no isolated junction, cycle, duplicate endpoint pair, reverse-slope conduit or pipe crown above a junction rim. These checks establish internal topological and geometric consistency. They do not establish correspondence with Shenzhen's surveyed municipal network. OpenStreetMap-derived geometry is attributed to OpenStreetMap contributors under the Open Database License.

![Figure 2. Conceptual-network geometry and topology. Panel (a) places the road-aligned network over the digital elevation model. Panel (b) shows diameter and degree. Panel (c) verifies the positive design-slope distribution. Panel (d) relates length, cover and diameter. The panels test internal consistency; they are not a validation against surveyed pipes.](figures/fig02_network_audit.png)

**Figure 2. Conceptual-network geometry and topology. Panel (a) places the road-aligned network over the digital elevation model. Panel (b) shows diameter and degree. Panel (c) verifies the positive design-slope distribution. Panel (d) relates length, cover and diameter. The panels test internal consistency; they are not a validation against surveyed pipes.**

### 2.3 Matched physical calculations

Surface flow was solved with Itzï 25.4 using its damped partial-inertia formulation. The minimum water depth was 0.001 m, the Courant number 0.7, the partial-inertia coefficient 0.9 and the maximum surface step 1 s. The rectangular outer boundary was closed. Active cells used Manning's n = 0.015 except where stated below. Building cells were elevated barriers. Rain falling on those cells was globally redistributed over active cells to conserve rainfall volume. A uniform 1 mm h-1 effective loss represented unresolved interception, infiltration and other continuing losses; it was not fitted as a measured soil parameter.

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

![Figure 3. Physical-run quality. Panel (a) reports routing continuity and non-converging steps against the stated numerical criteria. Panel (b) compares the roughness effect B-A with the sewer effect C-B. Panel (c) shows final surface storage under B and C. Panel (d) gives final-volume discrepancy relative to MIKE. The logarithmic scale in panel (b) is required because the two effects differ by nearly two orders of magnitude.](figures/fig03_physical_quality.png)

**Figure 3. Physical-run quality. Panel (a) reports routing continuity and non-converging steps against the stated numerical criteria. Panel (b) compares the roughness effect B-A with the sewer effect C-B. Panel (c) shows final surface storage under B and C. Panel (d) gives final-volume discrepancy relative to MIKE. The logarithmic scale in panel (b) is required because the two effects differ by nearly two orders of magnitude.**

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

![Figure 5. Event 68 peak-depth comparison. Panels (a-d) show MIKE, A, B and C. Panel (e) gives C-B, and panel (f) gives B-A. The separate residual scales reflect their different magnitudes. Display clipping affects colour only; all cells remain in the statistics.](figures/fig05_event68_physical_maps.png)

**Figure 5. Event 68 peak-depth comparison. Panels (a-d) show MIKE, A, B and C. Panel (e) gives C-B, and panel (f) gives B-A. The separate residual scales reflect their different magnitudes. Display clipping affects colour only; all cells remain in the statistics.**

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

![Figure 6. Held-event residual fields at the time of maximum coupled surface volume. Column 1 is the contemporaneous B input; columns 2 and 3 are true and predicted C-B; column 4 is the resulting depth error. Each row uses its own percentile-based residual range, so spatial pattern should be compared within rows rather than by colour intensity between events.](figures/fig06_fixed_network_residual_maps.png)

**Figure 6. Held-event residual fields at the time of maximum coupled surface volume. Column 1 is the contemporaneous B input; columns 2 and 3 are true and predicted C-B; column 4 is the resulting depth error. Each row uses its own percentile-based residual range, so spatial pattern should be compared within rows rather than by colour intensity between events.**

![Figure 7. Whole-event performance. Panels (a-c) report MAE, RMSE and CSI at 0.15 m. Lines in panel (a) connect prior-based results for individual events under the canonical seed. Error bars show sampling-seed standard deviation where five repeats were run; subgroup models were intentionally evaluated only under the canonical seed and therefore have zero repeat error bars.](figures/fig07_skill_and_sampling_seeds.png)

**Figure 7. Whole-event performance. Panels (a-c) report MAE, RMSE and CSI at 0.15 m. Lines in panel (a) connect prior-based results for individual events under the canonical seed. Error bars show sampling-seed standard deviation where five repeats were run; subgroup models were intentionally evaluated only under the canonical seed and therefore have zero repeat error bars.**

### 3.3 Static network information is detectable but secondary

Across five sampling seeds, adding all static network fields to the prior-plus-dynamic model changed MAE by 0.048 +/- 0.002 mm. This increment is small relative to the gains from the prior and dynamic state. It is also event dependent, as shown by the canonical-seed values below.

**Table 3. Canonical-seed held-event errors and network increments (mm).**

| Held event | Prior MAE | Prior + dynamic | Full hybrid | Gain over prior | Network increment |
| --- | --- | --- | --- | --- | --- |
| event1 | 2.498 | 2.307 | 2.246 | 0.252 | 0.061663 |
| event20 | 3.365 | 2.227 | 2.121 | 1.244 | 0.106210 |
| event65 | 2.702 | 2.299 | 2.218 | 0.484 | 0.081149 |
| event66 | 2.537 | 2.367 | 2.293 | 0.245 | 0.073993 |
| event67 | 4.617 | 2.574 | 2.559 | 2.058 | 0.014278 |
| event68 | 3.637 | 2.048 | 2.048 | 1.590 | 0.000250 |
| event69 | 2.669 | 1.961 | 1.936 | 0.733 | 0.024358 |
| event70 | 3.779 | 2.045 | 2.013 | 1.766 | 0.032667 |

The final-model controls nevertheless show that the increment is tied to spatial organisation. A 20 m shift increased MAE by 0.044 mm on average, and the penalty at 160 m was 0.139 mm. Zeroing the static fields increased MAE by 0.196 mm; block shuffling and group permutation increased it by 0.192 and 0.161 mm, respectively. These controls do not demonstrate transfer to another sewer layout, because the spatiotemporal prior still encodes the average response of the original network.

![Figure 8. Controls applied to the final hybrid. Panel (a) shows the MAE penalty as correctly aligned fields are displaced. Panel (b) shows zero-field, block-shuffle and joint-permutation penalties. Panel (c) compares mask and hydraulic groups. Panel (d) gives the paired static-network increment for five sampling seeds. Positive values mean the aligned full-network model is more accurate.](figures/fig08_final_hybrid_network_controls.png)

**Figure 8. Controls applied to the final hybrid. Panel (a) shows the MAE penalty as correctly aligned fields are displaced. Panel (b) shows zero-field, block-shuffle and joint-permutation penalties. Panel (c) compares mask and hydraulic groups. Panel (d) gives the paired static-network increment for five sampling seeds. Positive values mean the aligned full-network model is more accurate.**

### 3.4 Improvements persist where drainage is active

Full-domain means can be dominated by dry or weak-effect cells. Conditional evaluation gives a more demanding view. In the canonical-seed experiment, the full hybrid reduced MAE not only over all active cell-times but also for wet cells, locations with |C-B| above 5 mm, net-drainage cells, positive residual cells and the immediate inlet and pipe neighbourhoods. Errors remain larger in the strong-effect subsets, which identifies the residual transitions and local surcharge patches as the main unresolved structures.

**Table 4. Conditional depth MAE (mm), macro-averaged over events.**

| Evaluation subset | Surface B | ST prior | Prior + dynamic | Full hybrid |
| --- | --- | --- | --- | --- |
| All active cell-times | 9.266 | 3.226 | 2.228 | 2.179 |
| Coupled depth >= 0.03 m | 33.676 | 12.218 | 8.372 | 8.214 |
| \|C-B\| > 5 mm | 62.153 | 19.765 | 13.372 | 13.008 |
| C-B < -5 mm | 78.864 | 24.109 | 15.602 | 15.122 |
| C-B > 5 mm | 14.544 | 7.131 | 6.845 | 6.838 |
| Within 20 m of inlet | 34.893 | 7.029 | 6.139 | 4.713 |
| Within 20 m of pipe | 25.716 | 6.414 | 4.598 | 4.182 |

![Figure 9. Conditional MAE. Panel (a) separates the full domain from wet cells. Panel (b) progressively restricts evaluation to stronger drainage effects. Panel (c) groups cells by distance to an inlet. The change in vertical scale between panels is intentional: strong-effect and near-inlet subsets are harder than the domain average.](figures/fig09_conditioned_performance.png)

**Figure 9. Conditional MAE. Panel (a) separates the full domain from wet cells. Panel (b) progressively restricts evaluation to stronger drainage effects. Panel (c) groups cells by distance to an inlet. The change in vertical scale between panels is intentional: strong-effect and near-inlet subsets are harder than the domain average.**

### 3.5 MIKE comparison depends on the quantity being evaluated

MIKE was not generated with the same disclosed conceptual network and is not an independent validation target for C. The comparison is therefore descriptive. Matched surface B has the lowest full space-time MAE to MIKE, whereas C and DrainLite substantially reduce six-hour volume discrepancy. The opposite ordering of these metrics means that coupling removes a more plausible aggregate water volume without reproducing MIKE's complete spatial field.

**Table 5. External comparison with MIKE.**

| Field | MAE (mm) | RMSE (mm) | CSI 0.15 m | Peak-map MAE (mm) | Final-volume error (10^3 m3) |
| --- | --- | --- | --- | --- | --- |
| Matched surface B | 18.023 | 50.905 | 0.467 | 27.083 | 982.1 |
| Coupled label C | 21.258 | 62.509 | 0.311 | 34.608 | 310.6 |
| DrainLite full hybrid | 20.994 | 61.708 | 0.306 | 33.563 | 290.4 |

![Figure 10. Metric-dependent comparison with MIKE. Panels (a-d) show full-field MAE, peak-map MAE, final-volume error and CSI at 0.15 m. Smaller values are better in panels (a-c); larger values are better in panel (d). No single field dominates all four quantities, so the comparison cannot be reduced to a single statement of agreement.](figures/fig10_mike_metric_tradeoff.png)

**Figure 10. Metric-dependent comparison with MIKE. Panels (a-d) show full-field MAE, peak-map MAE, final-volume error and CSI at 0.15 m. Smaller values are better in panels (a-c); larger values are better in panel (d). No single field dominates all four quantities, so the comparison cannot be reduced to a single statement of agreement.**

### 3.6 End-to-end timing and post-processing

Feature construction, input assembly, model prediction and post-processing together required 23.6 s per event on average for the canonical full hybrid. The observed B-plus-DrainLite workflow required 297.7 s, compared with 3817.6 s for C, giving a mean speed ratio of 13.1. This is a workflow measurement on one workstation rather than a hardware-normalised benchmark. It also shows why estimator-only timing would overstate acceleration.

The primary result uses no residual clipping. Sensitivity calculations at +/-500 and +/-1200 mm quantify the effect of optional bounds, and the recorded non-negativity count shows how often Eq. (2) changes an otherwise negative predicted depth. Event selection is shown alongside these computational diagnostics to keep performance claims connected to the available forcing set.

![Figure 11. Computational and inclusion diagnostics. Panel (a) decomposes the complete correction time. Panel (b) compares B, B plus DrainLite and C on a logarithmic axis. Panel (c) reports residual-clipping sensitivity; the primary result is the unbounded residual followed only by the physical non-negativity constraint. Panel (d) places paired events within the readable public rainfall-reference inventory.](figures/fig11_runtime_clipping_event_selection.png)

**Figure 11. Computational and inclusion diagnostics. Panel (a) decomposes the complete correction time. Panel (b) compares B, B plus DrainLite and C on a logarithmic axis. Panel (c) reports residual-clipping sensitivity; the primary result is the unbounded residual followed only by the physical non-negativity constraint. Panel (d) places paired events within the readable public rainfall-reference inventory.**

### 3.7 Relation to the public LarNO checkpoint

The public LarNO checkpoint was executed independently for event 68. Its unclamped MAE was 2.409 mm, RMSE 5.119 mm and CSI at 0.15 m 0.955. Negative raw predictions comprised 18.15% of cells, so both raw and clamped statistics are retained.

![Figure 12. Independent public-checkpoint reproduction for event 68. Reference and prediction share a depth scale; the third panel shows signed error. Percentile limits are used only for display. This calculation verifies execution of the upstream checkpoint and is not part of DrainLite fitting.](figures/fig12_larno_reproduction.png)

**Figure 12. Independent public-checkpoint reproduction for event 68. Reference and prediction share a depth scale; the third panel shows signed error. Percentile limits are used only for display. This calculation verifies execution of the upstream checkpoint and is not part of DrainLite fitting.**

DrainLite was not appended to the public checkpoint. That checkpoint was trained against MIKE fields that may already contain drainage effects; applying C-B to it could count drainage twice. A defensible end-to-end LarNO-DrainLite calculation requires a LarNO model trained to drainage-free targets or paired neural-operator targets under matched network interventions.

### 3.8 Paired physical sensitivities for event 68

Five alternatives were calculated with paired B and C runs, changing one setting at a time: effective loss of 0 or 2 mm h-1, nearest-active-cell redistribution of building rainfall, exclusion of building rainfall, or inlet-neighbourhood Manning coefficient 0.015. The baseline retained 1 mm h-1 loss, global redistribution and coefficient 0.012. All alternatives produced finite 72-frame fields. Routing continuity magnitudes were below 0.2%; these diagnostics alone do not establish physical validity.

The mean drainage effect was 11.914 mm at baseline, 12.993 mm without effective loss and 10.874 mm at 2 mm h-1. Nearest-cell redistribution reduced it to 7.799 mm, and exclusion to 3.465 mm; the latter also removes rainfall volume and cannot isolate spatial routing alone. Changing inlet roughness gave 11.894 mm. Thus the existence of a drainage response persisted, while its magnitude depended substantially on rainfall treatment. C-to-MIKE MAE increased from 24.023 to 35.080 mm under nearest-cell redistribution, showing that local redistribution does not automatically improve agreement. This single-event experiment supports a sensitivity statement, not calibration or robustness of the trained residual model across physical configurations.

**Table 6. Event68 paired physical sensitivities.**

| Case | Mean \|C-B\| (mm) | Final B-C volume (m3) | C vs MIKE MAE (mm) | Routing error (%) |
| --- | --- | --- | --- | --- |
| baseline | 11.914 | 856686 | 24.023 | -0.06 |
| loss_0mmh | 12.993 | 950270 | 24.423 | -0.053 |
| loss_2mmh | 10.874 | 773714 | 23.796 | -0.067 |
| building_nearest | 7.799 | 556868 | 35.080 | -0.097 |
| building_exclude | 3.465 | 252345 | 27.255 | -0.16 |
| inlet_manning_0p015 | 11.894 | 854512 | 24.054 | -0.06 |

![Figure 13. Paired event68 physical sensitivities. Panels (a-d) show mean absolute drainage response, final surface-volume reduction, coupled-field discrepancy from MIKE and signed SWMM routing continuity. Loss values are in mm per hour. Exclude removes building rainfall volume; nearest changes its spatial allocation. Numerical continuity is not evidence of surveyed-network accuracy.](figures/fig13_physical_sensitivity.png)

**Figure 13. Paired event68 physical sensitivities. Panels (a-d) show mean absolute drainage response, final surface-volume reduction, coupled-field discrepancy from MIKE and signed SWMM routing continuity. Loss values are in mm per hour. Exclude removes building rainfall volume; nearest changes its spatial allocation. Numerical continuity is not evidence of surveyed-network accuracy.**

## 4. Discussion

### 4.1 What is learned on a fixed network

The results separate three sources of predictive skill. The event-excluded prior captures the repeatable cell-time response of one terrain-network system. Current surface depth and rainfall account for much of the event-specific departure from that response. Static pipe fields provide a smaller final adjustment. This hierarchy explains why a low-compute model can emulate C accurately while still offering limited evidence for network-conditioned generalisation.

The final-hybrid displacement and shuffle controls strengthen the interpretation of the static increment. Unlike earlier controls applied to a no-prior model, they perturb the exact estimator used for the main result. Their penalties show that correctly aligned fields are preferable to displaced or destroyed fields. Yet the prior remains fixed during those tests. The experiment therefore supports aligned static information within the original system, not counterfactual prediction after redesigning that system.

### 4.2 Error structure and physical meaning

Conditioned metrics show why the 2 mm-scale domain average should not be read as uniform accuracy. Where |C-B| exceeds 5 or 10 mm, the residual is sharper and less frequent, and the error rises. Immediate inlet areas are similarly difficult because the exchange is concentrated within a small number of cells. Positive C-B patches are not necessarily numerical failures: bidirectional routing can return water or displace storage locally while total surface volume falls. Keeping a signed target is therefore essential.

### 4.3 External reference and model scope

The MIKE comparison constrains interpretation rather than certifying calibration. The conceptual network and undisclosed MIKE drainage representation differ in topology, inlet density, capacity and boundary treatment. Better final-volume agreement under C suggests that surface-only storage is excessive, whereas poorer pixel-time MAE shows that the spatial redistribution is not the same. Both observations can be true.

Four boundaries remain. First, the sewer is road-derived rather than surveyed. Second, eight rainfall events share one terrain and network. Third, the surface edge is closed, building rainfall is globally redistributed and the 1 mm h-1 loss is an effective assumption. The event68 paired sensitivity experiment below quantifies their influence for one event; it does not establish robustness across events or retrained surrogates. Fourth, DrainLite requires the contemporaneous B field and should be described as a residual emulator, not an independent forecast model.

### 4.4 Implications for further development

The next physical dataset should vary network configuration as well as rainfall. Holding out a pipe-capacity, inlet-density or outlet-layout scenario would test whether the static fields act as intervention variables. Such an experiment should omit any prior derived from the target network. Dynamic node head, conduit flow, fullness and surcharge could then be rasterised as causal inputs if they are available at inference, or predicted by a separate network-state model. Only after a drainage-free LarNO upstream model is available should the two components be assessed as an end-to-end rainfall-to-flood system.

## 5. Conclusions

A connected conceptual sewer network was coupled through Itzï's native SWMM interface and evaluated using matched surface controls. Across eight events, |C-B| averaged 9.266 mm, compared with 0.197 mm for the roughness-only difference. All events met the predefined topology, continuity, convergence and combined mass-balance criteria.

Whole-event validation showed that the event-excluded spatiotemporal prior reduced MAE to 3.226 mm. Adding current surface and rainfall information reduced it to 2.217 mm, and adding all static network fields yielded 2.169 mm. The static increment was 0.048 +/- 0.002 mm across five sampling seeds. Final-model displacement and destructive controls confirm that this small increment depends on correct network alignment, while conditional metrics show that improvements persist in wet, near-inlet and strong-effect cells.

The evidence supports rapid residual emulation for rainfall events on this fixed conceptual network. It does not establish generalisation to another pipe layout or an end-to-end LarNO-DrainLite predictor. That distinction is central to both the scientific contribution and the next experimental step.

## Data availability

Public LarNO rainfall, terrain, MIKE arrays and the upstream checkpoint retain their original provenance. Locally calculated A/B/C labels and static descriptors are provided in the V3 canonical dataset. Road-derived geometry requires attribution to © OpenStreetMap contributors and is subject to the Open Database License. The repository licence map identifies materials whose downstream redistribution terms remain to be confirmed with the source authors.

## Code availability

Network construction, physical simulation, residual fitting, controls, figures and read-only verification are available at https://github.com/Coucou2016/20260519-LarNO-DrainLite. The repository records exact package versions, Git LFS objects, SHA-256 manifests and the distinction between regeneration and verification.

## Declaration of competing interests

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

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

## Appendix A. Predictor groups

The full hybrid comprises one event-excluded spatiotemporal prior, ten dynamic or terrain predictors and thirteen static network predictors. Explicit coordinates, synthetic outfall fields, distance-to-outfall fields and target-side SWMM state are excluded. The primary residual is not clipped; only the reconstructed water depth is constrained to be non-negative.

## Appendix B. Public-event inventory

**Table 7. Locally readable public event inventory.**

| Event | Paired A/B/C | Mean 6 h rain (mm) | Maximum local 6 h rain (mm) | MIKE global peak (m) |
| --- | --- | --- | --- | --- |
| event1 | True | 23.52 | 30.00 | 2.708 |
| event20 | True | 18.68 | 26.16 | 1.837 |
| event65 | True | 24.62 | 32.45 | 2.204 |
| event66 | True | 25.95 | 28.69 | 2.228 |
| event67 | True | 30.20 | 38.53 | 3.573 |
| event68 | True | 28.64 | 32.72 | 2.795 |
| event69 | True | 27.53 | 36.29 | 2.971 |
| event70 | True | 29.01 | 32.09 | 2.976 |
| event71 | False | 16.88 | 21.53 | 2.450 |
| event72 | False | 16.00 | 18.28 | 2.223 |
| event73 | False | 16.21 | 17.93 | 2.221 |
| event74 | False | 23.02 | 26.30 | 2.010 |
| event75 | False | 16.73 | 21.35 | 1.773 |
| event76 | False | 22.12 | 29.16 | 1.953 |
| event77 | False | 23.33 | 29.77 | 3.905 |
| event78 | False | 22.13 | 25.28 | 3.154 |
| event80 | False | 22.42 | 24.79 | 3.337 |


## Supplement. Complete metric disclosure

All metrics use equal event weights. Repeated-seed variability is not an event-population confidence interval. Deterministic B and prior baselines are repeated identically across seeds. Single-seed subgroup results cannot establish a seed-averaged ranking.

**Table 8. Complete calculated metrics.**

| Reference | Model | Metric (unit in name) | Mean | Seed SD |
| --- | --- | --- | --- | --- |
| coupled_label | surface_matched | mae_mm | 9.26593 | 0 |
| coupled_label | surface_matched | rmse_mm | 41.4214 | 0 |
| coupled_label | surface_matched | csi_0p03 | 0.889172 | 0 |
| coupled_label | surface_matched | csi_0p15 | 0.633089 | 0 |
| coupled_label | surface_matched | peak_map_mae_mm | 16.8351 | 0 |
| coupled_label | surface_matched | global_peak_bias_mm | 5.83929 | 0 |
| coupled_label | surface_matched | global_peak_abs_error_mm | 5.83929 | 0 |
| coupled_label | surface_matched | top0p1_peak_map_mae_mm | 81.4476 | 0 |
| coupled_label | surface_matched | final_volume_bias_m3 | 699525 | 0 |
| coupled_label | surface_matched | final_volume_abs_error_m3 | 699525 | 0 |
| coupled_label | surface_matched | mean_abs_volume_error_m3 | 331783 | 0 |
| coupled_label | surface_matched | max_abs_volume_error_m3 | 699525 | 0 |
| coupled_label | surface_matched | mean_abs_area_error_0p03_m2 | 665574 | 0 |
| coupled_label | surface_matched | mean_abs_area_error_0p15_m2 | 1.02613e+06 | 0 |
| coupled_label | surface_matched | global_peak_time_error_min | 1.25 | 0 |
| coupled_label | surface_matched | residual_mae_mm | 9.26593 | 0 |
| mike_external | surface_matched | mae_mm | 18.0229 | 0 |
| mike_external | surface_matched | rmse_mm | 50.9055 | 0 |
| mike_external | surface_matched | csi_0p03 | 0.630448 | 0 |
| mike_external | surface_matched | csi_0p15 | 0.46697 | 6.20634e-17 |
| mike_external | surface_matched | peak_map_mae_mm | 27.0826 | 0 |
| mike_external | surface_matched | global_peak_bias_mm | 929.911 | 0 |
| mike_external | surface_matched | global_peak_abs_error_mm | 929.911 | 0 |
| mike_external | surface_matched | top0p1_peak_map_mae_mm | 331.047 | 0 |
| mike_external | surface_matched | final_volume_bias_m3 | 982098 | 0 |
| mike_external | surface_matched | final_volume_abs_error_m3 | 982098 | 0 |
| mike_external | surface_matched | mean_abs_volume_error_m3 | 379414 | 0 |
| mike_external | surface_matched | max_abs_volume_error_m3 | 982098 | 0 |
| mike_external | surface_matched | mean_abs_area_error_0p03_m2 | 2.47473e+06 | 0 |
| mike_external | surface_matched | mean_abs_area_error_0p15_m2 | 898556 | 0 |
| mike_external | surface_matched | global_peak_time_error_min | 166.875 | 0 |
| mike_external | surface_matched | residual_mae_mm | 18.0229 | 0 |
| coupled_label | prior_only | mae_mm | 3.22551 | 0 |
| coupled_label | prior_only | rmse_mm | 13.6212 | 0 |
| coupled_label | prior_only | csi_0p03 | 0.897883 | 0 |
| coupled_label | prior_only | csi_0p15 | 0.866708 | 0 |
| coupled_label | prior_only | peak_map_mae_mm | 5.40299 | 0 |
| coupled_label | prior_only | global_peak_bias_mm | -2.98023e-05 | 0 |
| coupled_label | prior_only | global_peak_abs_error_mm | 3.56314 | 0 |
| coupled_label | prior_only | top0p1_peak_map_mae_mm | 51.9807 | 0 |
| coupled_label | prior_only | final_volume_bias_m3 | 58237.2 | 8.13477e-12 |
| coupled_label | prior_only | final_volume_abs_error_m3 | 129733 | 0 |
| coupled_label | prior_only | mean_abs_volume_error_m3 | 84947.9 | 0 |
| coupled_label | prior_only | max_abs_volume_error_m3 | 156017 | 0 |
| coupled_label | prior_only | mean_abs_area_error_0p03_m2 | 602466 | 0 |
| coupled_label | prior_only | mean_abs_area_error_0p15_m2 | 204622 | 0 |
| coupled_label | prior_only | global_peak_time_error_min | 0 | 0 |
| coupled_label | prior_only | residual_mae_mm | 3.22551 | 0 |
| mike_external | prior_only | mae_mm | 20.0495 | 0 |
| mike_external | prior_only | rmse_mm | 58.2834 | 0 |
| mike_external | prior_only | csi_0p03 | 0.594847 | 0 |
| mike_external | prior_only | csi_0p15 | 0.348403 | 0 |
| mike_external | prior_only | peak_map_mae_mm | 31.8815 | 0 |
| mike_external | prior_only | global_peak_bias_mm | 924.071 | 0 |
| mike_external | prior_only | global_peak_abs_error_mm | 924.071 | 0 |
| mike_external | prior_only | top0p1_peak_map_mae_mm | 521.183 | 0 |
| mike_external | prior_only | final_volume_bias_m3 | 340810 | 0 |
| mike_external | prior_only | final_volume_abs_error_m3 | 340810 | 0 |
| mike_external | prior_only | mean_abs_volume_error_m3 | 201036 | 0 |
| mike_external | prior_only | max_abs_volume_error_m3 | 497865 | 6.50781e-11 |
| mike_external | prior_only | mean_abs_area_error_0p03_m2 | 1.80129e+06 | 0 |
| mike_external | prior_only | mean_abs_area_error_0p15_m2 | 507679 | 0 |
| mike_external | prior_only | global_peak_time_error_min | 168.125 | 0 |
| mike_external | prior_only | residual_mae_mm | 20.0495 | 0 |
| coupled_label | dynamic_no_prior | mae_mm | 5.2731 | Not estimated |
| coupled_label | dynamic_no_prior | rmse_mm | 19.6354 | Not estimated |
| coupled_label | dynamic_no_prior | csi_0p03 | 0.896304 | Not estimated |
| coupled_label | dynamic_no_prior | csi_0p15 | 0.731109 | Not estimated |
| coupled_label | dynamic_no_prior | peak_map_mae_mm | 8.28399 | Not estimated |
| coupled_label | dynamic_no_prior | global_peak_bias_mm | -13.7837 | Not estimated |
| coupled_label | dynamic_no_prior | global_peak_abs_error_mm | 13.7837 | Not estimated |
| coupled_label | dynamic_no_prior | top0p1_peak_map_mae_mm | 98.9838 | Not estimated |
| coupled_label | dynamic_no_prior | final_volume_bias_m3 | 40525.9 | Not estimated |
| coupled_label | dynamic_no_prior | final_volume_abs_error_m3 | 40525.9 | Not estimated |
| coupled_label | dynamic_no_prior | mean_abs_volume_error_m3 | 28118.2 | Not estimated |
| coupled_label | dynamic_no_prior | max_abs_volume_error_m3 | 74175.3 | Not estimated |
| coupled_label | dynamic_no_prior | mean_abs_area_error_0p03_m2 | 458853 | Not estimated |
| coupled_label | dynamic_no_prior | mean_abs_area_error_0p15_m2 | 166219 | Not estimated |
| coupled_label | dynamic_no_prior | global_peak_time_error_min | 6.875 | Not estimated |
| coupled_label | dynamic_no_prior | residual_mae_mm | 5.2731 | Not estimated |
| mike_external | dynamic_no_prior | mae_mm | 19.9415 | Not estimated |
| mike_external | dynamic_no_prior | rmse_mm | 58.4645 | Not estimated |
| mike_external | dynamic_no_prior | csi_0p03 | 0.622872 | Not estimated |
| mike_external | dynamic_no_prior | csi_0p15 | 0.323108 | Not estimated |
| mike_external | dynamic_no_prior | peak_map_mae_mm | 32.5008 | Not estimated |
| mike_external | dynamic_no_prior | global_peak_bias_mm | 910.288 | Not estimated |
| mike_external | dynamic_no_prior | global_peak_abs_error_mm | 910.288 | Not estimated |
| mike_external | dynamic_no_prior | top0p1_peak_map_mae_mm | 542.949 | Not estimated |
| mike_external | dynamic_no_prior | final_volume_bias_m3 | 323099 | Not estimated |
| mike_external | dynamic_no_prior | final_volume_abs_error_m3 | 335568 | Not estimated |
| mike_external | dynamic_no_prior | mean_abs_volume_error_m3 | 268802 | Not estimated |
| mike_external | dynamic_no_prior | max_abs_volume_error_m3 | 594428 | Not estimated |
| mike_external | dynamic_no_prior | mean_abs_area_error_0p03_m2 | 2.27827e+06 | Not estimated |
| mike_external | dynamic_no_prior | mean_abs_area_error_0p15_m2 | 774093 | Not estimated |
| mike_external | dynamic_no_prior | global_peak_time_error_min | 161.25 | Not estimated |
| mike_external | dynamic_no_prior | residual_mae_mm | 19.9415 | Not estimated |
| coupled_label | hybrid_dynamic | mae_mm | 2.21681 | 0.00742009 |
| coupled_label | hybrid_dynamic | rmse_mm | 9.90684 | 0.140526 |
| coupled_label | hybrid_dynamic | csi_0p03 | 0.924667 | 0.000681433 |
| coupled_label | hybrid_dynamic | csi_0p15 | 0.905264 | 0.000374287 |
| coupled_label | hybrid_dynamic | peak_map_mae_mm | 3.5671 | 0.015373 |
| coupled_label | hybrid_dynamic | global_peak_bias_mm | -0.659359 | 0.861521 |
| coupled_label | hybrid_dynamic | global_peak_abs_error_mm | 4.63431 | 0.635414 |
| coupled_label | hybrid_dynamic | top0p1_peak_map_mae_mm | 48.387 | 1.33281 |
| coupled_label | hybrid_dynamic | final_volume_bias_m3 | -12677.1 | 4740.27 |
| coupled_label | hybrid_dynamic | final_volume_abs_error_m3 | 31967.1 | 1292.89 |
| coupled_label | hybrid_dynamic | mean_abs_volume_error_m3 | 20238.2 | 551.527 |
| coupled_label | hybrid_dynamic | max_abs_volume_error_m3 | 56937.5 | 3514.27 |
| coupled_label | hybrid_dynamic | mean_abs_area_error_0p03_m2 | 284498 | 11143 |
| coupled_label | hybrid_dynamic | mean_abs_area_error_0p15_m2 | 68763.8 | 1409 |
| coupled_label | hybrid_dynamic | global_peak_time_error_min | 1.125 | 0.279508 |
| coupled_label | hybrid_dynamic | residual_mae_mm | 2.21681 | 0.0074201 |
| mike_external | hybrid_dynamic | mae_mm | 20.9919 | 0.0103367 |
| mike_external | hybrid_dynamic | rmse_mm | 61.7205 | 0.0517232 |
| mike_external | hybrid_dynamic | csi_0p03 | 0.603618 | 0.00060361 |
| mike_external | hybrid_dynamic | csi_0p15 | 0.305034 | 0.000489771 |
| mike_external | hybrid_dynamic | peak_map_mae_mm | 33.5181 | 0.0288058 |
| mike_external | hybrid_dynamic | global_peak_bias_mm | 923.412 | 0.861521 |
| mike_external | hybrid_dynamic | global_peak_abs_error_mm | 923.412 | 0.861521 |
| mike_external | hybrid_dynamic | top0p1_peak_map_mae_mm | 535.001 | 3.31014 |
| mike_external | hybrid_dynamic | final_volume_bias_m3 | 269896 | 4740.27 |
| mike_external | hybrid_dynamic | final_volume_abs_error_m3 | 286709 | 3787.79 |
| mike_external | hybrid_dynamic | mean_abs_volume_error_m3 | 248244 | 763.366 |
| mike_external | hybrid_dynamic | max_abs_volume_error_m3 | 549376 | 2033.87 |
| mike_external | hybrid_dynamic | mean_abs_area_error_0p03_m2 | 1.94959e+06 | 14379.4 |
| mike_external | hybrid_dynamic | mean_abs_area_error_0p15_m2 | 706655 | 2350.18 |
| mike_external | hybrid_dynamic | global_peak_time_error_min | 167 | 0.279508 |
| mike_external | hybrid_dynamic | residual_mae_mm | 20.9919 | 0.0103366 |
| coupled_label | hybrid_mask | mae_mm | 2.17178 | Not estimated |
| coupled_label | hybrid_mask | rmse_mm | 9.42876 | Not estimated |
| coupled_label | hybrid_mask | csi_0p03 | 0.928757 | Not estimated |
| coupled_label | hybrid_mask | csi_0p15 | 0.906041 | Not estimated |
| coupled_label | hybrid_mask | peak_map_mae_mm | 3.53834 | Not estimated |
| coupled_label | hybrid_mask | global_peak_bias_mm | -0.784814 | Not estimated |
| coupled_label | hybrid_mask | global_peak_abs_error_mm | 4.45163 | Not estimated |
| coupled_label | hybrid_mask | top0p1_peak_map_mae_mm | 53.6794 | Not estimated |
| coupled_label | hybrid_mask | final_volume_bias_m3 | -6005.98 | Not estimated |
| coupled_label | hybrid_mask | final_volume_abs_error_m3 | 27339.1 | Not estimated |
| coupled_label | hybrid_mask | mean_abs_volume_error_m3 | 19538 | Not estimated |
| coupled_label | hybrid_mask | max_abs_volume_error_m3 | 50977.5 | Not estimated |
| coupled_label | hybrid_mask | mean_abs_area_error_0p03_m2 | 260146 | Not estimated |
| coupled_label | hybrid_mask | mean_abs_area_error_0p15_m2 | 64204.2 | Not estimated |
| coupled_label | hybrid_mask | global_peak_time_error_min | 0.625 | Not estimated |
| coupled_label | hybrid_mask | residual_mae_mm | 2.17178 | Not estimated |
| mike_external | hybrid_mask | mae_mm | 20.9981 | Not estimated |
| mike_external | hybrid_mask | rmse_mm | 61.7902 | Not estimated |
| mike_external | hybrid_mask | csi_0p03 | 0.602454 | Not estimated |
| mike_external | hybrid_mask | csi_0p15 | 0.30538 | Not estimated |
| mike_external | hybrid_mask | peak_map_mae_mm | 33.5639 | Not estimated |
| mike_external | hybrid_mask | global_peak_bias_mm | 923.286 | Not estimated |
| mike_external | hybrid_mask | global_peak_abs_error_mm | 923.286 | Not estimated |
| mike_external | hybrid_mask | top0p1_peak_map_mae_mm | 546.455 | Not estimated |
| mike_external | hybrid_mask | final_volume_bias_m3 | 276567 | Not estimated |
| mike_external | hybrid_mask | final_volume_abs_error_m3 | 292447 | Not estimated |
| mike_external | hybrid_mask | mean_abs_volume_error_m3 | 247733 | Not estimated |
| mike_external | hybrid_mask | max_abs_volume_error_m3 | 550123 | Not estimated |
| mike_external | hybrid_mask | mean_abs_area_error_0p03_m2 | 1.94505e+06 | Not estimated |
| mike_external | hybrid_mask | mean_abs_area_error_0p15_m2 | 701012 | Not estimated |
| mike_external | hybrid_mask | global_peak_time_error_min | 167.5 | Not estimated |
| mike_external | hybrid_mask | residual_mae_mm | 20.9981 | Not estimated |
| coupled_label | hybrid_hydraulic | mae_mm | 2.19271 | Not estimated |
| coupled_label | hybrid_hydraulic | rmse_mm | 9.47748 | Not estimated |
| coupled_label | hybrid_hydraulic | csi_0p03 | 0.927269 | Not estimated |
| coupled_label | hybrid_hydraulic | csi_0p15 | 0.905321 | Not estimated |
| coupled_label | hybrid_hydraulic | peak_map_mae_mm | 3.55321 | Not estimated |
| coupled_label | hybrid_hydraulic | global_peak_bias_mm | 0.278711 | Not estimated |
| coupled_label | hybrid_hydraulic | global_peak_abs_error_mm | 3.47567 | Not estimated |
| coupled_label | hybrid_hydraulic | top0p1_peak_map_mae_mm | 52.3708 | Not estimated |
| coupled_label | hybrid_hydraulic | final_volume_bias_m3 | -6191.02 | Not estimated |
| coupled_label | hybrid_hydraulic | final_volume_abs_error_m3 | 27269.6 | Not estimated |
| coupled_label | hybrid_hydraulic | mean_abs_volume_error_m3 | 19664.8 | Not estimated |
| coupled_label | hybrid_hydraulic | max_abs_volume_error_m3 | 51731.5 | Not estimated |
| coupled_label | hybrid_hydraulic | mean_abs_area_error_0p03_m2 | 276219 | Not estimated |
| coupled_label | hybrid_hydraulic | mean_abs_area_error_0p15_m2 | 67095.1 | Not estimated |
| coupled_label | hybrid_hydraulic | global_peak_time_error_min | 1.25 | Not estimated |
| coupled_label | hybrid_hydraulic | residual_mae_mm | 2.19271 | Not estimated |
| mike_external | hybrid_hydraulic | mae_mm | 20.9967 | Not estimated |
| mike_external | hybrid_hydraulic | rmse_mm | 61.8028 | Not estimated |
| mike_external | hybrid_hydraulic | csi_0p03 | 0.603376 | Not estimated |
| mike_external | hybrid_hydraulic | csi_0p15 | 0.305122 | Not estimated |
| mike_external | hybrid_hydraulic | peak_map_mae_mm | 33.5442 | Not estimated |
| mike_external | hybrid_hydraulic | global_peak_bias_mm | 924.35 | Not estimated |
| mike_external | hybrid_hydraulic | global_peak_abs_error_mm | 924.35 | Not estimated |
| mike_external | hybrid_hydraulic | top0p1_peak_map_mae_mm | 547.537 | Not estimated |
| mike_external | hybrid_hydraulic | final_volume_bias_m3 | 276382 | Not estimated |
| mike_external | hybrid_hydraulic | final_volume_abs_error_m3 | 291943 | Not estimated |
| mike_external | hybrid_hydraulic | mean_abs_volume_error_m3 | 248133 | Not estimated |
| mike_external | hybrid_hydraulic | max_abs_volume_error_m3 | 550539 | Not estimated |
| mike_external | hybrid_hydraulic | mean_abs_area_error_0p03_m2 | 1.95999e+06 | Not estimated |
| mike_external | hybrid_hydraulic | mean_abs_area_error_0p15_m2 | 704167 | Not estimated |
| mike_external | hybrid_hydraulic | global_peak_time_error_min | 166.875 | Not estimated |
| mike_external | hybrid_hydraulic | residual_mae_mm | 20.9967 | Not estimated |
| coupled_label | hybrid_all | mae_mm | 2.16904 | 0.0075742 |
| coupled_label | hybrid_all | rmse_mm | 9.55794 | 0.127621 |
| coupled_label | hybrid_all | csi_0p03 | 0.928614 | 0.000381742 |
| coupled_label | hybrid_all | csi_0p15 | 0.906413 | 0.000390592 |
| coupled_label | hybrid_all | peak_map_mae_mm | 3.51563 | 0.0213327 |
| coupled_label | hybrid_all | global_peak_bias_mm | -1.42884 | 0.573357 |
| coupled_label | hybrid_all | global_peak_abs_error_mm | 5.17611 | 0.961956 |
| coupled_label | hybrid_all | top0p1_peak_map_mae_mm | 49.9754 | 2.27002 |
| coupled_label | hybrid_all | final_volume_bias_m3 | -8607.36 | 3711.69 |
| coupled_label | hybrid_all | final_volume_abs_error_m3 | 29050.3 | 2161.13 |
| coupled_label | hybrid_all | mean_abs_volume_error_m3 | 19793.9 | 527.95 |
| coupled_label | hybrid_all | max_abs_volume_error_m3 | 53050.6 | 1610.6 |
| coupled_label | hybrid_all | mean_abs_area_error_0p03_m2 | 264162 | 5784.47 |
| coupled_label | hybrid_all | mean_abs_area_error_0p15_m2 | 63905.1 | 1538.94 |
| coupled_label | hybrid_all | global_peak_time_error_min | 1.125 | 0.927025 |
| coupled_label | hybrid_all | residual_mae_mm | 2.16904 | 0.00757421 |
| mike_external | hybrid_all | mae_mm | 20.9936 | 0.00938279 |
| mike_external | hybrid_all | rmse_mm | 61.7082 | 0.0689164 |
| mike_external | hybrid_all | csi_0p03 | 0.602089 | 0.000306814 |
| mike_external | hybrid_all | csi_0p15 | 0.305971 | 0.000470568 |
| mike_external | hybrid_all | peak_map_mae_mm | 33.5627 | 0.027146 |
| mike_external | hybrid_all | global_peak_bias_mm | 922.642 | 0.573357 |
| mike_external | hybrid_all | global_peak_abs_error_mm | 922.642 | 0.573357 |
| mike_external | hybrid_all | top0p1_peak_map_mae_mm | 541.714 | 4.09353 |
| mike_external | hybrid_all | final_volume_bias_m3 | 273966 | 3711.69 |
| mike_external | hybrid_all | final_volume_abs_error_m3 | 290418 | 2936.28 |
| mike_external | hybrid_all | mean_abs_volume_error_m3 | 247714 | 689.745 |
| mike_external | hybrid_all | max_abs_volume_error_m3 | 547794 | 1391.58 |
| mike_external | hybrid_all | mean_abs_area_error_0p03_m2 | 1.92883e+06 | 12189.8 |
| mike_external | hybrid_all | mean_abs_area_error_0p15_m2 | 702500 | 2816.26 |
| mike_external | hybrid_all | global_peak_time_error_min | 167 | 0.927025 |
| mike_external | hybrid_all | residual_mae_mm | 20.9936 | 0.00938279 |
