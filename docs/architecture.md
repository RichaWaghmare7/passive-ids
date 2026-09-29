# Architecture notes

This file exists so the "why" behind each module isn't lost once the
implementation starts filling in the TODOs.

## Why bounded memory has two lanes (flow_state/)

A confirmed 5-tuple flow table is capacity-capped, because a flood or
scan is itself a resource-exhaustion attempt against the detector, not
just the target. Provisional entries (cheap, short TTL) filter out the
one-off packets a spoofed flood generates before they earn an
expensive full entry. But the per-source-IP aggregate (HLL + Count-Min)
must run unconditionally on every packet, regardless of whether the
confirmed table has room -- it's what DDoS and port-scan detection
actually read from, and it's exactly during a flood that the confirmed
table is under the most pressure. Degrading the wrong lane under load
would blind the detector at the worst possible moment.

## Why calibration is separate from models (calibration/)

Six different algorithm families producing six different native output
scales (a Hoeffding Tree's class probability, Isolation Forest's raw
anomaly score, LightGBM's sigmoid output) cannot be averaged together
meaningfully. Isotonic regression maps each one onto a comparable 0-1
scale first -- and it's fit once per model *instance*, not once per
threat class, since c2_beaconing's Isolation Forest and
exfiltration's Isolation Forest are unrelated fitted models on
unrelated feature distributions.

## Why alerting is multi-label, not a single risk score (alerting/)

A flow can trigger reconnaissance and DGA activity at the same time.
A weighted-sum fusion into one number hides that. `Detection` objects
are a list per alert; `Summary.overall_score` exists only as a sort
key for triage dashboards, always labeled with the method used to
compute it (noisy-OR or max), never presented as a calibrated
probability of a specific attack.

## Why thresholds and SLAs are templates, not tuned values (config/)

Both `thresholds.yaml` and `sla_matrix.yaml` need real data this
project doesn't have yet: labeled positive/negative examples per
class (for PR curves) and measured alert arrival rates (for the
Erlang-C staffing calculation). Filling these in with guessed numbers
would make them look authoritative when they aren't. `null` is the
honest placeholder until `scripts/compute_pr_curves.py` and a pilot
deployment produce real ones.

## Why verdict collection samples sub-threshold flows, not just alerts (verdicts/)

Only reviewing alerts that crossed a threshold means false negatives
are structurally invisible -- there is no ground truth for a flow that
was never shown to anyone. `sample_reason` distinguishes `alerted`
(precision signal) from `random` (unbiased recall signal) from
`boundary` (uncertainty-sampled, most useful for retraining but biased
by design -- must not be mixed into the recall estimate).

## Model learning cadence (models/)

Online learners (Hoeffding Tree, and the Welford/Markov feature layers
feeding the others) update per-instance with no downtime. Batch models
(Isolation Forest x2, LightGBM, XGBoost) are only ever replaced through
`feedback/retrain.py` + canary deployment -- never mutated while
serving live traffic.
