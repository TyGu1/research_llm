# Adaptive-attack evaluation — smoke

- samples: **30** questions from **2** CUAD dev contracts (seed 42, min 5 answerable questions per contract)
- task types: {'entity_specific': 11, 'framework_reasoning': 6, 'compliance_regulatory': 10, 'procedural': 3}; category groups: {'date': 4, 'clause': 21, 'name': 2, 'party': 3}
- the sensitive document per query is a 4000-char window of the contract around the first gold span (window contains a gold span for 30/30)
- all intervals: cluster bootstrap by contract, 300 resamples, 95%

### Rung 0 — leak scan of the transmitted content (no attacker model)

| Arm | n | Gold span in transmitted text | Party name | Any date | Any amount |
|---|---|---|---|---|---|
| Topic bottleneck | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Topic bottleneck + vocab exclusion | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Entity substitution | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| SanText+ ($\varepsilon$=1) | 6 | 16.7% [0.0, 33.3] | 0.0% [0.0, 0.0] | 83.3% [66.7, 100.0] | 16.7% [0.0, 33.3] |

### Rungs a, b — reconstruction attacks (per query)

| Arm | Rung | Attack | Obs. | n | SVR | Any gold span | Party name | Category named | Situation |
|---|---|---|---|---|---|---|---|---|---|
| Topic bottleneck | a | Direct (paper prompt) | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.333 [0.332, 0.334] |
| Topic bottleneck | a | Expert persona (paper prompt) | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.309 [0.298, 0.319] |
| Topic bottleneck | a | ICL, 3 solved examples | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.324 [0.297, 0.352] |
| Topic bottleneck | a | ICL, 20 solved examples | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 83.3% [66.7, 100.0] | 0.361 [0.335, 0.387] |
| Topic bottleneck | a | ICL, examples from the same contract | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.343 [0.336, 0.351] |
| Topic bottleneck | b1 | Category inversion (pipeline-aware); category top-1 66.7% [66.7, 66.7], top-3 100.0% [100.0, 100.0] | Q+ | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.262 [0.260, 0.264] |
| Topic bottleneck + vocab exclusion | a | Direct (paper prompt) | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.325 [0.315, 0.335] |
| Topic bottleneck + vocab exclusion | a | Expert persona (paper prompt) | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.345 [0.338, 0.352] |
| Topic bottleneck + vocab exclusion | a | ICL, 3 solved examples | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.358 [0.330, 0.386] |
| Topic bottleneck + vocab exclusion | a | ICL, 20 solved examples | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.315 [0.286, 0.344] |
| Topic bottleneck + vocab exclusion | a | ICL, examples from the same contract | Q | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.310 [0.259, 0.361] |
| Topic bottleneck + vocab exclusion | b1 | Category inversion (pipeline-aware); category top-1 83.3% [66.7, 100.0], top-3 100.0% [100.0, 100.0] | Q+ | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.255 [0.228, 0.283] |
| Topic bottleneck + vocab exclusion | b2 | Absence fingerprint on Q; excluded-word recall 11.0% [9.7, 12.2] | Q+ | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.280 [0.225, 0.335] |
| Topic bottleneck + vocab exclusion | b2 | Absence fingerprint on S0 text (upper bound); excluded-word recall 45.1% [44.4, 45.8] | S0 | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.410 [0.379, 0.442] |
| Entity substitution | a | Placeholder re-identification | SUB-text | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 33.3% [33.3, 33.3] | 0.337 [0.324, 0.351] |
| Entity substitution | a | ICL reconstruction (Pang et al. recipe) | DP/SUB-text | 6 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 16.7% [0.0, 33.3] | 0.358 [0.341, 0.374] |
| SanText+ ($\varepsilon$=1) | a | Direct reconstruction | DP-text | 6 | 50.0% [16.7, 83.3] | 66.7% [33.3, 100.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.605 [0.575, 0.635] |
| SanText+ ($\varepsilon$=1) | a | ICL reconstruction (Pang et al. recipe) | DP/SUB-text | 6 | 50.0% [16.7, 83.3] | 66.7% [33.3, 100.0] | 0.0% [0.0, 0.0] | 66.7% [66.7, 66.7] | 0.586 [0.579, 0.594] |

### SVR by value type (strongest realistic attack per arm)

| Arm (strongest realistic attack) | Parties | Dates | Document name | Clause text |
|---|---|---|---|---|
| Topic bottleneck — Direct (paper prompt) | – (n=0) | 0.0% [0.0, 0.0] (n=2) | – (n=0) | 0.0% [0.0, 0.0] (n=4) |
| Topic bottleneck + vocab exclusion — Direct (paper prompt) | – (n=0) | 0.0% [0.0, 0.0] (n=2) | – (n=0) | 0.0% [0.0, 0.0] (n=4) |
| Entity substitution — Placeholder re-identification | – (n=0) | 0.0% [0.0, 0.0] (n=2) | – (n=0) | 0.0% [0.0, 0.0] (n=4) |
| SanText+ ($\varepsilon$=1) — Direct reconstruction | – (n=0) | 50.0% [0.0, 100.0] (n=2) | – (n=0) | 50.0% [25.0, 75.0] (n=4) |

### Rungs b3, c — embedding linkage (no attacker LLM)

| Arm | Category top-1 | Category top-5 | Contract top-1 | Contract top-5 | Contract top-1, k=3 | k=5 | k=all |
|---|---|---|---|---|---|---|---|
| Topic bottleneck | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 33.3% [33.3, 33.3] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | – | 100.0% [100.0, 100.0] |
| Topic bottleneck + vocab exclusion | 83.3% [66.7, 100.0] | 83.3% [66.7, 100.0] | 50.0% [0.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | – | 100.0% [100.0, 100.0] |
| chance | 2.4% | 12.2% | 50.0% | 250.0% | 50.0% | 50.0% | 50.0% |

### Rung c — multi-query aggregation with the LLM attacker

| Arm | k | n contracts | Any gold span of the contract | SVR over the contract's spans | Party name |
|---|---|---|---|---|---|
| Topic bottleneck | 3 | 4 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Topic bottleneck | all | 4 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Topic bottleneck + vocab exclusion | 3 | 4 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Topic bottleneck + vocab exclusion | all | 4 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

### Utility — answer quality against the oracle (same model answers the plain question with the full window)

| Arm | n | BERTScore F1 vs oracle | Semantic similarity | Gold-span overlap |
|---|---|---|---|---|
| Topic bottleneck | 6 | 0.910 [0.907, 0.913] | 0.852 [0.848, 0.855] | 83.3% [66.7, 100.0] |
| Topic bottleneck + vocab exclusion | 6 | 0.905 [0.904, 0.905] | 0.856 [0.846, 0.866] | 83.3% [66.7, 100.0] |
| Entity substitution | 6 | 0.823 [0.821, 0.824] | 0.610 [0.560, 0.659] | 16.7% [0.0, 33.3] |
| SanText+ ($\varepsilon$=1) | 6 | 0.857 [0.849, 0.866] | 0.597 [0.593, 0.601] | 16.7% [0.0, 33.3] |
