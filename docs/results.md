# Stored results

Every result file in `runs/baseline/` is written by one command in `docs/reproducing.md` and carries an
`_eval_provenance` block (commit hash and dirty flag of the evaluating code). Regret values are in decades
of base-10 BER; "positive" in a contrast means learning is better, unless the file says otherwise. All
contrasts are paired cluster-bootstrap differences over units (95 % percentile intervals).

| File | Written by | Content | Manuscript section |
|------|-----------|---------|--------------------|
| `states.csv` | `dcbias.generate` | One row per simulated state: exogenous state, latent device parameters (`u_*`), label bias `L1`, power-minimising label `L2`, oracle bias `O_L1` (test states), diagnostics | Problem and Simulator |
| `provenance.json` | `dcbias.generate` | Commit, configuration and its hash, seeds, library versions, label diagnostics | Problem and Simulator |
| `results_main.json` | `evaluate main` | Mean regret of every controller on the four test sets, selection, hyperparameters, confirmatory contrast, seed spread, power and FEC label, latency and size | Results: optimum versus ceiling, confirmatory test, out-of-distribution behaviour, power and FEC, cost |
| `results_sensing.json` | `evaluate sensing` | Thermal-family versus waveform-family regret over the sensor-noise grid, with stress columns and an ablation | Sensing |
| `results_learning.json` | `evaluate learning` | Learned-controller regret against the number of training units | Cost, latency, learning curve and seeds |
| `results_b8.json` | `evaluate b8` | Cost-regret frontier of the online golden-section search | Cost |
| `results_misspec.json` | `evaluate misspec` | Model-based controllers built on a wrong LED model, against the best learned controller | Model misspecification |
| `results_sweeps.json` | `dcbias.sweep_summary` | Per-sweep summary of the one-factor-at-a-time runs in `runs/sweep-*/` | Sweeps |
| `results_shift.json` | `evaluate shift` | Device-shift arm: every controller built on the nominal device and deployed on LED twins; cost of each shift; wide-prior B7; SNR-slope penalties | Device shift |
| `results_slopes.json` | `evaluate slopes` | SNR slope of BER at the oracle bias, per test split | Power and FEC, supplement |
| `results_shifttails.json` | `dcbias.shift_tails` | Median, 90th and 99th percentile of the MLP regret per twin, tail share, seed range | Supplement |
| `results_b7cal.json` | `evaluate b7cal` | B7 calibrated on 0, 10, 20 or 40 characterised units | Learning curve, supplement |
| `results_b7wide_imin.json`, `results_b7wide_extras.json` | `evaluate b7wide_imin`, `dcbias.b7_wide_extras` | Wide-prior B7 on the two turn-on-clip twins and its FEC-relevant contrasts | Device shift |
| `results_counted.json`, `results_counted_bands.json` | `dcbias.counted`, `dcbias.counted_bands` | Bit-error counting on the stress shifts and its band-resolved read-out | Counted bits on the stress shifts |
| `results_floor_sweepref.json` | `dcbias.floor_sweepref` | Sweep contrast against the 50-unit reference, noise floor of the regret metric, oracle-bias displacement per twin | Sweeps, device shift |
| `paper_numbers.tex` | `dcbias.macros` | LaTeX macros `\vnum{<file>/<path>}` for every number the manuscript cites | all |

`runs/sweep-<key>=<value>/` holds, for each sweep value, `results_main.json` (same format as the baseline file)
and `provenance.json`. The sweep named `led.model=tsonev_cubic` also holds `results_misspec.json`.

## Controller names in the result keys

`B0_static`, `B1a_k*`, `B1b_k*`, `B1c_kfit`, `B2_zhang_midpoint`, `B3_centre`, `B4_lut1d_C`, `B5_lut2d_Ir_T`,
`B6a_lut_state`, `B6b_model_grid`, `B7_bayes`, `B7w_bayes_wave`, `ML_{ridge,rf,hgb,mlp}_{state,state_obs}`.
A suffix `@seedmean` is the mean over five training seeds, `@seed<k>` one seed; `B7_bayes_wide` is the
wide-prior variant and `B6b_model_grid@device` the model grid built on the true deployed device (an upper reference).

## Test splits

`test_id` (in distribution), `ood_T` (heat-sink temperature above the training range), `ood_SNR`
(reference SNR outside the range) and `ood_sigma` (signal level outside the range). `fec_relevant` rows are
states whose oracle minimum BER meets the HD-FEC threshold.

## Decision criteria in `results_main.json`

* `C1`: the best learned controller beats the best non-learning controller on `test_id` by at least the equivalence margin (0.10 decades), with a bootstrap interval above zero.
* `C2`: the interval lower bound is above zero on every out-of-distribution test set.
* `C3`: the violation-rate difference between the learned and the non-learning controller (constraint non-inferiority) has an interval upper bound within `evaluation.c3_margin` (0.01, in `dcbias/config.json`).

`framing` states which row of the decision table the outcome falls in.
