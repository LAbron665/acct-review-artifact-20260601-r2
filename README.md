# ACCT Anonymous Artifact

This directory contains a local reference implementation for **Agent-Time
Counterfactual Credit Transport (ACCT)**.  The implementation is intentionally
small and identity-scrubbed so it can be verified as a local ZIP bundle and
mirrored through an anonymous review link:
`https://anonymous.4open.science/r/acctreviewartifact20260601r2`.

## Layout

- `acct/credit.py`: counterfactual influence, magnitude and directional
  temporal credit transport, counterfactual-salience transport-only
  redistribution, return-equivalent salience redistribution, and a sampled
  agent-time Shapley diagnostic baseline.
- `acct/critics.py`: quadratic ridge reward model for learned-critic diagnostics.
- `acct/envs.py`: fast local sparse-reward cooperative diagnostics.
- `acct/learners.py`: tabular policy-gradient harness.
- `experiments/run_local_experiments.py`: runs local benchmark seeds.
- `experiments/plot_results.py`: regenerates the legacy paper figure and
  summary CSV.
- `experiments/run_medium_benchmark_suite.py`: runs the medium local suite
  (MPE2 PPO/MAPPO at 600 episodes, LBF/RWARE at 1000 episodes, fixed 16-seed
  order) and writes unified final-return, AUC, threshold, bootstrap-CI, and
  win-rate CSVs.
- `experiments/plot_paper_figures.py`: regenerates the polished method,
  local-result, medium-benchmark, and appendix diagnostic figures used by the
  paper.
- `experiments/run_mpe_delayed_experiment.py`: runs a lightweight delayed-reward
  MPE2 Simple Spread diagnostic.
- `experiments/run_mpe_ppo_experiment.py`: runs the clipped PPO-style MPE2
  diagnostic.
- `experiments/run_mpe_mappo_experiment.py`: runs a compact centralized-value
  MAPPO-style MPE2 diagnostic with delayed rewards, GAE, and conservative
  learned-credit hybrid shaping.
- `experiments/analyze_mpe_counterfactual_quality.py`: trains the compact
  MAPPO-style shared-GAE model and reports held-out centralized critic fit plus
  counterfactual-credit alignment diagnostics.
- `experiments/analyze_learned_critic_sample_sweep.py`: sweeps local learned
  reward-model sample sizes and reports held-out fit plus learned-credit
  downstream returns.
- `experiments/analyze_influence_perturbation.py`: perturbs exact
  counterfactual influence by scale or additive noise and compares raw AT-CF
  with ACCT transport on the local diagnostic tasks.
- `experiments/run_epymarl_smoke.py`: runs a tiny external-EPyMARL MAPPO/LBF
  smoke test when an EPyMARL checkout is supplied by the caller.
- `experiments/run_epymarl_comparison.py`: runs a tiny paired EPyMARL MAPPO vs.
  learned-head ACCT LBF comparison for matched seeds.
- `experiments/analyze_mpe_statistics.py`: computes paired bootstrap intervals
  and seed-wise win rates for MPE2/LBF/RWARE diagnostic CSV files.
- `experiments/audit_paper_claims.py`: verifies that key numeric claims in the
  LaTeX draft match the artifact CSV files after paper-style rounding.
- `experiments/audit_anonymity.py`: scans text sources for identity-sensitive
  local path or Overleaf remote residue and writes `results/anonymity_audit.csv`.
- `experiments/audit_dependency_licenses.py`: writes a dependency license
  metadata CSV from `requirements*.txt` and installed Python package metadata.
- `experiments/audit_bibliography.py`: checks manuscript citation keys against
  the BibTeX database and verifies locators for arXiv-style entries.
- `experiments/audit_scope_guard.py`: checks that the manuscript keeps explicit
  local-evidence caveats and does not introduce unsupported SOTA or benchmark
  dominance claims.
- `experiments/audit_submission_readiness.py`: writes a non-failing
  reviewer-facing report of final-submission blockers such as missing public
  anonymous URL state, public-code checklist state, large-scale benchmark
  caveats, release-repository dry-run evidence, and finalization-report
  matching after the anonymous URL is present.
- `experiments/check_latex_submission.py`: checks the compiled LaTeX log and
  `.aux` metadata for unresolved references/citations, overfull boxes, and the
  NeurIPS-style main-text page budget.
- `experiments/verify_anonymous_artifact.py`: extracts the anonymous ZIP into a
  clean temporary directory, checks for required files and identity-sensitive
  strings, validates all `MANIFEST.md` SHA-256 entries, and runs smoke
  verification commands from the extracted copy.
- `frameworks/epymarl_acct_adapter.py`: shape-safe ACCT advantage adapter for
  EPyMARL-style PPO/actor-critic tensors, with synthetic smoke tests.
- `frameworks/patch_epymarl_acct.py`: applies the adapter to an external
  EPyMARL checkout for proxy-influence and learned-Q-head ACCT learner smoke
  tests.
- `frameworks/README_EPymarl_ACCT.md`: inspected EPyMARL insertion points and
  a minimal patch sketch for future full-framework experiments.
- `requirements_epymarl_smoke.txt`: optional dependency list for the external
  EPyMARL smoke test.
- `experiments/run_lbf_ppo_experiment.py`: runs a tiny PPO-style Level-Based
  Foraging stress test.
- `experiments/run_rware_ppo_experiment.py`: runs a tiny PPO-style RWARE stress
  test; the current local run is a sparse-exploration negative result.
- `experiments/analyze_credit_localization.py`: computes credit mass and
  coverage on known reward-relevant agent-time masks.
- `experiments/analyze_learned_salience_pruning.py`: checks whether dense
  learned counterfactual salience ranks reward-relevant agent-time pairs ahead
  of distractors under top-fraction pruning diagnostics.
- `experiments/analyze_pruned_learned_acct.py`: runs the local top-fraction
  learned-pruned-ACCT training diagnostic and paired bootstrap summaries.
- `experiments/analyze_pruned_fraction_sensitivity.py`: reruns learned-pruned
  ACCT with nearby top-fraction thresholds to check pruning sensitivity.
- `experiments/analyze_distractor_scaling.py`: increases irrelevant
  agent-time decisions while holding the reward-relevant pairs fixed.
- `experiments/analyze_local_statistics.py`: computes paired bootstrap
  intervals and win rates for local diagnostic comparisons.
- `experiments/analyze_learning_efficiency.py`: computes full-curve sampled
  and greedy AUC plus episodes-to-threshold summaries from the local diagnostic
  learning curves.
- `experiments/analyze_query_budget.py`: computes per-trajectory
  reward/critic-query budgets and transport arithmetic counts for local credit
  rules.
- `experiments/analyze_data_budget.py`: summarizes training episodes,
  evaluation episodes, and auxiliary critic-fit trajectories for the reported
  diagnostics.
- `experiments/analyze_acct_sensitivity.py`: runs a small lambda/beta
  sensitivity sweep for ACCT transport on the local diagnostic tasks.
- `experiments/summarize_mpe_results.py`: summarizes the MPE2 diagnostic table.
- `package_anonymous_artifact.py`: creates a zip bundle with source, tests, raw
  CSV logs, and a SHA-256 manifest for anonymous review upload.
- `prepare_anonymous_release_repo.py`: rebuilds the verified ZIP into a fresh
  Git repository with anonymous commit metadata, ready to push to a review-only
  GitHub repository before creating an anonymous.4open.science link.
- `publish_anonymous_release_repo.py`: checks the prepared release repository
  for leaks/residue, then either pushes to a caller-supplied fresh Git remote or
  uses an authenticated GitHub CLI to create and push a fresh review-only repo.
- `finalize_anonymous_url.py`: validates an anonymous.4open.science URL and
  inserts it into the paper/checklist so final readiness can be audited.
- `Makefile`: short reviewer-facing targets for smoke verification, paper-claim
  auditing, packaging, and clean ZIP verification.
- `THIRD_PARTY.md`: dependency license-audit notes and interpretation.
- `tests/test_artifact_verifier.py`: unit tests for manifest SHA-256
  verification in the anonymous artifact checker.
- `tests/test_credit.py`: unit tests for conservation, uniform and directional
  fallback behavior, counterfactual action replacement, beta-zero and additive
  consistency cases, temporal masking, sampled Shapley attribution, agent-time
  counterfactual baselines, and actor-credit shape compatibility.

## Quick Verification

From this directory, reviewers can run the lightweight checks without
retraining the full local experiment matrix:

```bash
make artifact-smoke PYTHON=python3
make release-check PYTHON=python3
make release-repo PYTHON=python3
python3 publish_anonymous_release_repo.py --remote <fresh-review-only-git-url> --dry-run
python3 publish_anonymous_release_repo.py --github-repo <owner/fresh-review-only-repo> --dry-run
python3 finalize_anonymous_url.py --url https://anonymous.4open.science/r/<review-id> --dry-run
make submission-readiness PYTHON=python3
```

`artifact-smoke` runs the unit tests and recomputes the anonymity audit, stored query-budget,
data-budget, learning-efficiency, dependency-license metadata, and MAPPO
statistics summaries from the included CSV logs. `release-check` additionally compiles and audits the LaTeX
draft, checks the main-text page budget, audits paper claims in the full
repository checkout, checks bibliography consistency and claim-scope guardrails,
rebuilds the anonymous ZIP, and verifies the ZIP from a clean extraction directory. In an extracted ZIP without the paper source, use
`artifact-smoke`. `release-repo` creates
`dist/acct_anonymous_release_repo`, initializes it as a clean Git repository,
and writes `dist/anonymous_release_repo_report.json`. The publish script can
then dry-run the final leak/metadata checks, push that prepared repository to a
caller-supplied fresh review-only Git remote, or create and push a fresh GitHub
repository when `gh auth login` has already been completed. `submission-readiness` writes
`results/submission_readiness.csv`; after the anonymous.4open.science link
exists, `finalize_anonymous_url.py` updates the manuscript and checklist in one
step.

## Reproduce Local Results

```bash
cd code
PYTHONPATH=. python3 -m unittest discover tests
PYTHONPATH=. python3 experiments/run_local_experiments.py --episodes 800 --seeds 32
PYTHONPATH=. MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp python3 experiments/plot_results.py
PYTHONPATH=. .venv/bin/python experiments/run_medium_benchmark_suite.py --skip-existing
PYTHONPATH=. MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp .venv/bin/python experiments/plot_paper_figures.py
PYTHONPATH=. .venv/bin/python experiments/run_mpe_delayed_experiment.py --episodes 300 --seeds 8
PYTHONPATH=. .venv/bin/python experiments/summarize_mpe_results.py
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/mpe_delayed_results.csv --out results/mpe_delayed_stat_tests.csv --baseline-method shared --episode 300
PYTHONPATH=. .venv/bin/python experiments/run_mpe_ppo_experiment.py --episodes 300 --seeds 8
PYTHONPATH=. .venv/bin/python experiments/summarize_mpe_results.py --input results/mpe_ppo_results.csv --summary results/mpe_ppo_summary.csv --methods ppo_shared ppo_learned_agent_time_cf ppo_learned_agent_shapley ppo_learned_acct ppo_learned_directional_acct
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/mpe_ppo_results.csv --out results/mpe_ppo_stat_tests.csv --baseline-method ppo_shared --episode 300
PYTHONPATH=. .venv/bin/python experiments/run_mpe_mappo_experiment.py --episodes 300 --seeds 8 --hybrid-weight 0.05
PYTHONPATH=. .venv/bin/python experiments/summarize_mpe_results.py --input results/mpe_mappo_results.csv --summary results/mpe_mappo_summary.csv --methods mappo_shared_gae mappo_learned_agent_time_cf mappo_learned_acct mappo_gae_plus_agent_time_cf mappo_gae_plus_acct
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/mpe_mappo_results.csv --out results/mpe_mappo_stat_tests.csv --baseline-method mappo_shared_gae --episode 300
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_counterfactual_quality.py --episodes 300 --seeds 8 --holdout 16
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_counterfactual_quality.py --episodes 600 --seeds 8 --holdout 16 --out results/mpe_counterfactual_quality_600.csv --summary-out results/mpe_counterfactual_quality_600_summary.csv
PYTHONPATH=. .venv/bin/python experiments/analyze_learned_critic_sample_sweep.py --episodes 400 --seeds 8 --critic-samples 128 512 2048 8192 --holdout 512
PYTHONPATH=. .venv/bin/python experiments/analyze_influence_perturbation.py --episodes 400 --seeds 16
PYTHONPATH=. .venv/bin/python experiments/run_lbf_ppo_experiment.py --episodes 500 --seeds 8
PYTHONPATH=. .venv/bin/python experiments/summarize_mpe_results.py --input results/lbf_ppo_results.csv --summary results/lbf_ppo_summary.csv --methods lbf_ppo_shared lbf_ppo_learned_agent_time_cf lbf_ppo_learned_acct
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/lbf_ppo_results.csv --out results/lbf_ppo_stat_tests.csv --baseline-method lbf_ppo_shared --episode 500
PYTHONPATH=. .venv/bin/python experiments/run_rware_ppo_experiment.py --episodes 300 --seeds 8 --max-steps 100
PYTHONPATH=. .venv/bin/python experiments/summarize_mpe_results.py --input results/rware_ppo_results.csv --summary results/rware_ppo_summary.csv --methods rware_ppo_shared rware_ppo_learned_agent_time_cf rware_ppo_learned_acct
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/rware_ppo_results.csv --out results/rware_ppo_stat_tests.csv --baseline-method rware_ppo_shared --episode 300
PYTHONPATH=. .venv/bin/python experiments/analyze_credit_localization.py --seeds 32 --samples 256
PYTHONPATH=. .venv/bin/python experiments/analyze_learned_salience_pruning.py
PYTHONPATH=. .venv/bin/python experiments/analyze_pruned_learned_acct.py
PYTHONPATH=. .venv/bin/python experiments/analyze_pruned_fraction_sensitivity.py
PYTHONPATH=. .venv/bin/python experiments/analyze_distractor_scaling.py
PYTHONPATH=. .venv/bin/python experiments/analyze_local_statistics.py --episode 200
PYTHONPATH=. .venv/bin/python experiments/analyze_query_budget.py
PYTHONPATH=. .venv/bin/python experiments/analyze_data_budget.py
PYTHONPATH=. .venv/bin/python experiments/analyze_learning_efficiency.py
PYTHONPATH=. .venv/bin/python experiments/analyze_acct_sensitivity.py --episodes 400 --seeds 8
PYTHONPATH=. .venv/bin/python experiments/check_latex_submission.py
PYTHONPATH=. .venv/bin/python experiments/audit_anonymity.py
PYTHONPATH=. .venv/bin/python experiments/audit_paper_claims.py
PYTHONPATH=. .venv/bin/python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl
PYTHONPATH=. .venv/bin/python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl --apply-acct-patch --use-acct --out results/epymarl_acct_smoke.json
PYTHONPATH=. .venv/bin/python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl --apply-acct-patch --use-acct --acct-influence-source learned_q_head --standardise-rewards False --acct-residual-weight 0.1 --out results/epymarl_acct_qhead_smoke.json
PYTHONPATH=. .venv/bin/python experiments/run_epymarl_comparison.py --epymarl-root /path/to/epymarl --t-max 60 --time-limit 5 --seeds 0 1 2 --standardise-rewards False
PYTHONPATH=. .venv/bin/python experiments/analyze_mpe_statistics.py --input results/epymarl_lbf_comparison.csv --out results/epymarl_lbf_comparison_stat_tests.csv --baseline-method epymarl_mappo --episode 60 --methods epymarl_acct_learned_qhead
PYTHONPATH=. .venv/bin/python package_anonymous_artifact.py
PYTHONPATH=. .venv/bin/python experiments/verify_anonymous_artifact.py
PYTHONPATH=. .venv/bin/python prepare_anonymous_release_repo.py --force
PYTHONPATH=. .venv/bin/python publish_anonymous_release_repo.py --remote <fresh-review-only-git-url> --dry-run
PYTHONPATH=. .venv/bin/python publish_anonymous_release_repo.py --github-repo <owner/fresh-review-only-repo> --dry-run
PYTHONPATH=. .venv/bin/python finalize_anonymous_url.py --url https://anonymous.4open.science/r/<review-id> --dry-run
PYTHONPATH=. .venv/bin/python experiments/audit_submission_readiness.py
```

The legacy local figure is written to
`../Formatting_Instructions_For_NeurIPS_2026/Figure/acct_local_results.pdf`.
The polished paper figures are written to the same `Figure/` directory as
`acct_method_schematic.pdf`, `acct_local_results_polished.pdf`,
`acct_medium_benchmarks.pdf`, and `acct_diagnostics_appendix.pdf`.
The anonymous artifact bundle is written to `dist/acct_anonymous_artifact.zip`
and includes a manifest with SHA-256 hashes. The clean extraction verifier
checks every manifest entry against the extracted file contents, and writes its
report to `dist/artifact_verification.json`.
The release-repository preparation step writes
`dist/acct_anonymous_release_repo` and commits the extracted artifact with
`Anonymous Authors <anonymous@example.com>` metadata. To create the final
double-anonymous link, push that directory to a fresh review-only GitHub
repository, then anonymize that repository through
`https://anonymous.4open.science/`. That service expects an accessible GitHub
repository URL, not a local directory or ZIP upload. The publish helper can
perform the final local leak/metadata checks and push once the fresh remote URL
is available; with GitHub CLI authentication, it can instead create and push the
fresh review-only repo via `--github-repo <owner/name>`.
After anonymous.4open.science returns a review URL, the finalization helper
updates the paper's artifact sentence and the NeurIPS open-code checklist answer
and writes `dist/finalize_anonymous_url_report.json`; `submission-readiness --strict`
requires that non-dry-run report to match the manuscript URL before declaring
the package final-ready. The current manuscript is finalized with
`https://anonymous.4open.science/r/acctreviewartifact20260601r2`.
The MPE2, LBF, and RWARE diagnostics require the optional PyTorch/PettingZoo/MPE2/LBF/RWARE
dependencies listed in `requirements.txt`; set `SDL_VIDEODRIVER=dummy` on
headless macOS if your shell does not inherit the default set by the MPE scripts.
The EPyMARL smoke command additionally requires an external EPyMARL checkout and
the packages in `requirements_epymarl_smoke.txt`.  It writes only an anonymous
summary JSON and should not be interpreted as benchmark evidence.  The patched
ACCT smoke mode uses a policy-confidence proxy influence only to verify learner
plumbing; the learned-Q-head smoke mode trains a tiny actual-action value head
and then evaluates counterfactual action replacements, but it is still not a
tuned benchmark.  The paired EPyMARL/LBF comparison is likewise a tiny
integration stress test: under `t_max=60` and three matched seeds, the learned
head path trains but does not improve return over MAPPO.

## Scope

The code is a local diagnostic artifact, not a full PyMARL2 or EPyMARL fork.
The ACCT functions are framework-agnostic and can be inserted before a MAPPO or
value-decomposition actor/value update when a centralized critic can provide
counterfactual Q estimates.
The included EPyMARL material is limited to an adapter scaffold plus a tiny
MAPPO/LBF smoke run summary, a patched proxy-ACCT smoke run summary, and a
patched learned-Q-head smoke run summary, plus a tiny paired MAPPO-vs-ACCT LBF
stress test that is negative for ACCT under this budget; it does not include
tuned EPyMARL ACCT results.
The included baselines are deliberately simple: shared return, terminal-only
counterfactual credit, uniform temporal transport, counterfactual-salience
transport-only redistribution, exact agent-time counterfactual influence,
sampled agent-time Shapley credit, directional ACCT, and learned counterparts
from a small ridge reward model.
