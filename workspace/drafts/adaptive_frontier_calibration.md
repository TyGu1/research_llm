# Frontier-attacker calibration (claude-opus-5-5) — main, arm plain

Same (sample, attack) pairs; attacker = Gemma-4-31B (local) vs claude-opus-5-5 (`claude -p`). n = 36 samples over 36 contracts. 95% cluster-bootstrap CIs by contract.

| Attack | n | Gemma SVR | Frontier SVR | Gemma any-gold | Frontier any-gold | Gemma party | Frontier party | Gemma cat. named | Frontier cat. named | Frontier refusals |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | 36 | 5.6% [0.0, 13.9] | 3.7% [0.0, 10.2] | 5.6% [0.0, 13.9] | 5.6% [0.0, 13.9] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 58.3% [44.4, 75.0] | 72.2% [58.3, 86.1] | 2 |
| icl20 | 36 | 6.5% [0.0, 14.8] | 6.5% [0.0, 14.8] | 8.3% [0.0, 19.4] | 8.3% [0.0, 19.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 58.3% [41.7, 75.0] | 75.0% [61.1, 88.9] | 1 |
| category_inversion (category top-1: Gemma 52.8% [36.1, 69.4], frontier 52.8% [36.1, 69.4]) | 36 | 2.8% [0.0, 8.3] | 8.3% [0.0, 19.4] | 2.8% [0.0, 8.3] | 8.3% [0.0, 19.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 88.9% [77.8, 97.2] | 80.6% [66.7, 91.7] | 0 |
