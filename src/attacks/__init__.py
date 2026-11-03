"""
Adaptive attacks against the topic-bottleneck pipeline (and its baselines).

Observation sets (what the attacker is given) are explicit everywhere:
  Q    the transmitted proxy questions (exactly what the cloud receives)
  Q+   Q plus knowledge of the pipeline (prompts, topic step, the 41 CUAD categories)
  S0   the local S0 text — NEVER transmitted; upper bound only, always labelled
  Qxk  Q for k queries from the same contract
  Pool a candidate population of contracts/categories the attacker can enumerate
"""
