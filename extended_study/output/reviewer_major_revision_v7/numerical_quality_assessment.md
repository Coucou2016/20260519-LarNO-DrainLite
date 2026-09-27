# Numerical quality assessment

Global routing continuity errors range from -0.178% to -0.060%, and reported nonconverging steps from 0.00% to 0.04%. Continuity error is a numerical water-budget residual, not a disconnected-pipe count. A nonzero residual is not automatically a model failure, but small global values do not establish local accuracy. In event68 the reported node N01520 has a 63.01% inflow-normalised local imbalance. Its rounded cumulative inflow is about 114 m3, corresponding to approximately 71.8 m3 of imbalance; this is not 63% of whole-network water. The same node's inflow table reports 170.331% because that table uses outflow as its denominator. These are different normalisations of the same budget discrepancy, not conflicting results. The existing aggregate screening therefore does not close the numerical validation. The reported local errors must be checked against node-level exchanges and shorter-step results before the labels can be described as locally validated.

| Event | Global balance (%) | Nonconverging steps (%) | Top local node | Local balance (% inflow) | Assessment |
| --- | --- | --- | --- | --- | --- |
| event1 | -0.083 | 0.02 | N00184 | 47.07 | Local review required |
| event20 | -0.178 | 0.0 | N00184 | 69.6 | Local review required |
| event65 | -0.111 | 0.0 | N00184 | 65.45 | Local review required |
| event66 | -0.101 | 0.0 | N00184 | 58.75 | Local review required |
| event67 | -0.066 | 0.01 | N01520 | 65.04 | Local review required |
| event68 | -0.06 | 0.04 | N01520 | 63.01 | Local review required |
| event69 | -0.068 | 0.01 | N01520 | 52.93 | Local review required |
| event70 | -0.063 | 0.01 | N01520 | 56.86 | Local review required |

## Interpretation and required follow-up

The EPA manual (Section 8.5) treats continuity as a numerical diagnostic, calls for investigation of excessive local errors at relevant nodes, and recommends time series at one-minute or shorter resolution for initial instability screening. Its example of a 10% excessive global error is not a universal safe threshold. The repository limits of 2% routing/nonconvergence and 0.5% combined error are project screens, not proof of physical validity. A printed 0.00% is rounded and does not prove zero failed iterations.

A nonconverging-step percentage measures steps that fail the iteration stopping criterion; it is neither a percent of wrong depths nor an estimate of mass loss. The current report prints two decimal places and cannot recover exact failed-step counts or their timing.

Next acceptance: instrument inflow, outflow, storage and signed exchange at the flagged nodes; repeat event68 and the event with the largest local discrepancy with reduced maximum routing steps (0.5, 0.25, 0.125 s) and reviewed head tolerance/iteration settings; compare node hydrographs, local budgets and surface labels. Do not alter the physical network merely to improve a diagnostic. No convergence rerun was executed by this audit, and no label or learned prediction has been replaced.

## Sources

- [EPA manual, Sections 8.5 and 9.1](https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf)
- [Engine 5.2.4 local ranking formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/stats.c)
- [Engine 5.2.4 inflow-table formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/statsrpt.c)
