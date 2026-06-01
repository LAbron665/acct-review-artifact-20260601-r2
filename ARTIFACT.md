# Artifact Notes

This artifact is prepared for anonymous review packaging and mirrored at
`https://anonymous.4open.science/r/acctreviewartifact20260601r2`.  It can also
be verified as a local ZIP bundle and staged as a review-only Git repository
with anonymous commit metadata.

## Claims Supported by Current Runs

- Unit tests pass for ACCT credit conservation, directional residual transport,
  counterfactual replacement, agent-time counterfactual baseline behavior, and
  learned-critic credit shapes.
- Local diagnostics run for 32 seeds and 800 training episodes.
- The generated local learning-curve figure comes from
  `results/local_experiments.csv`; the polished figure bundle also reads the
  medium benchmark and diagnostic CSV files.
- The local diagnostics include a sampled agent-time Shapley attribution
  baseline with 16 reference-trajectory permutations per update.
- The local diagnostics include a uniform temporal-transport ablation to
  separate temporal spreading from counterfactual salience.
- The local diagnostics include exact and learned counterfactual-salience
  transport-only baselines to separate normalized residual redistribution from
  ACCT's added instantaneous counterfactual influence.
- The local diagnostics include exact and learned return-equivalent
  counterfactual transport proxies that normalize the observed team return
  over agent-time salience; these are local diagnostics for the redistribution
  question, not implementations of TAR^2.
- Learned-critic diagnostics use a quadratic ridge reward model fitted from
  uniformly sampled local trajectories.
- A lightweight delayed-reward MPE2 Simple Spread diagnostic runs for 8 seeds
  and 300 training episodes with a learned centralized Q critic.
- A clipped PPO-style MPE2 diagnostic runs over the same environment, seeds,
  and episode budget, including learned sampled-Shapley and directional-ACCT
  credit baselines.
- A compact centralized-value MAPPO-style MPE2 diagnostic runs with delayed
  rewards, return-to-go value fitting, GAE, and conservative learned-credit
  hybrid shaping; in the current run hybrid AT-CF slightly improves greedy
  return, while hybrid ACCT is near the shared-GAE baseline and does not improve
  sampled return.
- Held-out MPE2 counterfactual-quality diagnostics at 300 and 600 training
  episodes report centralized Q/V return-to-go fit and the alignment between
  learned counterfactual influence, ACCT credit, and shared GAE; the longer run
  improves observed-action return fit but still finds weak influence alignment,
  so it supports the paper's caution about learned-credit quality rather than a
  benchmark-performance claim.
- A local learned-critic sample-size sweep fits the quadratic reward model with
  128, 512, 2048, and 8192 uniformly sampled trajectories. It reports held-out
  reward-model MSE/R2 and shows that learned ACCT has higher Sample@200 than
  raw learned AT-CF at every tested sample size on both local tasks, while final
  greedy return is saturated for both methods.
- An influence-perturbation diagnostic scales exact counterfactual influence by
  0.25, 1.0, and 4.0 or adds Gaussian noise with standard deviation 0.10 and
  0.25. Over 16 seeds and 400 training episodes, ACCT gives larger Sample@200
  than raw perturbed AT-CF in the weak-scale and moderate-noise settings, while
  both methods saturate final greedy return on the local tasks.
- A tiny PPO-style Level-Based Foraging stress test runs on a 3x3 two-agent
  cooperative collection task; it is included as an external-environment stress
  test, not as a headline performance claim.
- A tiny PPO-style RWARE stress test runs on `rware-tiny-2ag-easy-v2`; the
  current 300-episode local run obtains zero final sample and greedy return for
  all compared methods, so it is reported as a sparse-exploration negative
  result rather than evidence for ACCT.
- The medium benchmark-suite runner writes separate `medium_*` raw logs plus
  `results/medium_benchmark_summary.csv` and
  `results/medium_benchmark_stat_tests.csv`.  It keeps the scope local while
  broadening the evidence package to MPE2 PPO/MAPPO at 600 episodes and
  LBF/RWARE at 1000 episodes with 16 seeds when the suite is run.
- The polished figure generator writes the ACCT schematic, local learning
  curves, medium benchmark summary, and appendix diagnostics to the NeurIPS
  figure directory, with `results/figure_manifest.csv` recording the produced
  figure files.
- Credit-localization diagnostics report relevant credit mass, relevant-pair
  coverage, and irrelevant-pair activation on the two local tasks with known
  reward-relevant agent-time masks.
- Learned-salience pruning diagnostics report whether the dense learned
  counterfactual influence ranks reward-relevant agent-time pairs ahead of
  distractors after retaining only the largest salience entries.
- A local learned-pruned-ACCT training diagnostic applies top-fraction pruning
  inside learned ACCT and reports paired Sample@200 summaries against learned
  ACCT and learned AT-CF; it is diagnostic evidence, not a new headline method.
- A nearby-fraction sensitivity diagnostic checks top-0.10, top-0.20, and
  top-0.30 pruning so the top-fraction result is not presented as a single
  unexamined threshold.
- A distractor-scaling stress diagnostic increases irrelevant agent-time
  decisions while holding the same four reward-relevant pairs fixed, testing
  whether the local ACCT mechanism survives controlled credit distraction.
- Paired bootstrap summaries report confidence intervals and seed-wise win
  rates for Sample@200 comparisons in the local diagnostics.
- Full-curve learning-efficiency summaries report sampled-return AUC,
  greedy-return AUC, and the first episode at which each local run reaches
  sampled return 1.5, reducing dependence on a single selected evaluation
  point.
- Query-budget summaries report per-trajectory credit-specific reward/critic
  query counts, replacement counts, Shapley permutation counts, and transport
  arithmetic entries for the local diagnostic rules.
- Data-budget summaries report training episodes, evaluation episodes, and
  auxiliary critic-fit trajectories for the reported diagnostics, so learned
  counterfactual-credit claims can be interpreted separately from exact/oracle
  diagnostics and extra offline critic data.
- Dependency-license metadata is written to
  `results/dependency_license_audit.csv` from `requirements*.txt` and installed
  Python package metadata; missing optional packages are recorded rather than
  treated as failures.
- MPE2/LBF paired bootstrap summaries report confidence intervals and seed-wise
  win rates against the corresponding shared-return or shared-GAE baselines.
- ACCT sensitivity diagnostics report lambda/residual-weight sweeps and
  directional transport variants on both local tasks for 8 seeds and 400
  training episodes.
- The paper-claim audit regenerates the key rounded numeric snippets from CSV
  files and checks that they appear in the LaTeX draft.
- The anonymity audit scans text sources for identity-sensitive local path or
  Overleaf remote residue and writes `results/anonymity_audit.csv`.
- The bibliography audit checks that manuscript citation keys are defined in
  BibTeX, that BibTeX entries are cited, and that arXiv-style entries include a
  DOI, URL, or eprint locator.
- The scope-guard audit checks that local-evidence caveats remain in the
  manuscript and that unsupported SOTA or benchmark dominance claims are absent.
- The submission-readiness audit writes
  `results/submission_readiness.csv` with a report-only distinction between a
  verified local artifact and a final submission package. It checks public
  anonymous URL state, checklist public-code language, large-scale benchmark
  caveats, the prepared release repository report, the publish dry-run report,
  and the post-upload finalization report when an anonymous URL is present.
- The LaTeX submission audit compiles the draft and checks unresolved
  references/citations, overfull boxes, and the main-text page budget from the
  generated `.aux` and `.log` files.
- `make artifact-smoke` runs unit tests and recomputes the anonymity audit,
  stored query-budget, data-budget, learning-efficiency, dependency-license metadata, and MAPPO statistics summaries from
  included logs; `make release-check` also compiles and audits the LaTeX draft,
  checks bibliography consistency and claim-scope guardrails, rebuilds the ZIP,
  and verifies the clean extraction in the full checkout.
- The anonymous artifact verifier extracts the release ZIP into `/private/tmp`,
  checks for required files and identity-sensitive strings, validates every
  `MANIFEST.md` SHA-256 entry against the extracted file contents, and runs
  smoke commands from the extracted copy.
- `prepare_anonymous_release_repo.py` and `make release-repo` rebuild the
  verified ZIP into a fresh Git repository with anonymous commit metadata. This
  is the staging tree intended to be pushed to a review-only GitHub repository
  before creating an anonymous.4open.science link.
- `publish_anonymous_release_repo.py` performs final local leak/residue checks
  on that prepared repository, then pushes it to a caller-supplied fresh
  review-only Git remote or creates and pushes a fresh GitHub repository through
  an authenticated GitHub CLI.
- `finalize_anonymous_url.py` validates the anonymous.4open.science URL and
  updates the paper artifact sentence plus the NeurIPS open-code checklist
  answer after the upload link exists.
- A lightweight EPyMARL adapter maps EPyMARL-style PPO/actor-critic tensors to
  ACCT advantages and includes synthetic shape/mask smoke tests. This is an
  integration scaffold, not a completed EPyMARL benchmark result.
- `results/epymarl_smoke.json` records one tiny external-EPyMARL MAPPO/LBF run
  that completed locally (`t_max=20`, `time_limit=5`). This is framework-entry
  smoke evidence only, not benchmark evidence.
- `results/epymarl_acct_smoke.json` records one patched external-EPyMARL run
  with `use_acct=True`. The patch uses a policy-confidence proxy influence to
  verify that ACCT advantage transport can enter the EPyMARL PPO actor-loss
  path; it is not a counterfactual-Q benchmark.
- `results/epymarl_acct_qhead_smoke.json` records one patched external-EPyMARL
  run with `acct_influence_source=learned_q_head`. The smoke trains a tiny
  actual-action value head and evaluates counterfactual action replacements;
  it is stronger than the proxy plumbing check, but still not a tuned benchmark
  result.
- `results/epymarl_lbf_comparison.csv` records a three-seed paired
  external-EPyMARL MAPPO-vs-learned-head-ACCT LBF stress test (`t_max=60`,
  `time_limit=5`). In this tiny run, MAPPO obtains final sample return
  `0.278 +/- 0.255`, learned-head ACCT obtains `0.000 +/- 0.000`, and both
  methods obtain zero final greedy return. This is negative integration
  evidence for the current learned-head path, not a performance claim.

## Claims Not Yet Supported

- No SMAC or SMACv2 result is included. The included RWARE result is only a
  tiny negative stress test, not a tuned benchmark claim.
- No full PyMARL2/EPyMARL/MAPPO benchmark integration is included; the EPyMARL
  files, smoke JSONs, and tiny paired comparison are integration evidence, not
  tuned benchmark evidence.
- No full framework comparison to AREL, ATA, STAS, TAR^2, MAPPO, or
  value-factorization baselines is included; the included PPO-style and
  MAPPO-style runs are compact local sanity checks rather than tuned benchmark
  implementations.
- The manuscript cites only the anonymous.4open.science review URL. Direct
  GitHub repository URLs should not be cited in the double-blind paper.

## Anonymization Checklist

- No author names are present in source files.
- No institution names are present in source files.
- Generated caches are ignored by `.gitignore`.
- For public review, use the anonymous.4open.science mirror rather than a direct
  GitHub URL.
- Run `PYTHONPATH=. .venv/bin/python package_anonymous_artifact.py` to create
  `dist/acct_anonymous_artifact.zip` with a SHA-256 manifest before upload.
- Run `PYTHONPATH=. .venv/bin/python experiments/verify_anonymous_artifact.py`
  to verify the ZIP from a clean extraction directory before upload, including
  manifest hash validation.
- Run `PYTHONPATH=. .venv/bin/python prepare_anonymous_release_repo.py --force`
  or `make release-repo PYTHON=.venv/bin/python` to create a clean local Git
  repository at `dist/acct_anonymous_release_repo` with anonymous author and
  committer metadata.
- After creating a fresh review-only Git remote, run
  `PYTHONPATH=. .venv/bin/python publish_anonymous_release_repo.py --remote <fresh-review-only-git-url> --dry-run`
  to recheck local release metadata before the actual push; omit `--dry-run` to
  push.
- Alternatively, after `gh auth login`, run
  `PYTHONPATH=. .venv/bin/python publish_anonymous_release_repo.py --github-repo <owner/fresh-review-only-repo> --dry-run`
  and then rerun without `--dry-run` to create and push a fresh review-only
  GitHub repository. Add `--visibility private` if the anonymous.4open.science
  login session can read private repositories; otherwise use the default public
  visibility and cite only the anonymous.4open.science URL in the paper.
- Run `PYTHONPATH=. .venv/bin/python experiments/audit_submission_readiness.py`
  to write `results/submission_readiness.csv`; use `--strict` only for a final
  pre-submission gate after the anonymous review URL has been created.
- Push `dist/acct_anonymous_release_repo` to a fresh review-only GitHub
  repository and then anonymize that repository through
  `https://anonymous.4open.science/`.
- anonymous.4open.science requires an accessible GitHub repository URL; it does
  not create the GitHub repository from a local directory or ZIP file.
- After anonymous.4open.science returns a new review URL, run
  `PYTHONPATH=. .venv/bin/python finalize_anonymous_url.py --url https://anonymous.4open.science/r/<review-id>`
  and then rerun `make release-check PYTHON=.venv/bin/python` plus
  `PYTHONPATH=. .venv/bin/python experiments/audit_submission_readiness.py --strict`.
  The readiness audit requires a non-dry-run `dist/finalize_anonymous_url_report.json`
  whose URL matches the manuscript.
- Alternatively, run `make release-check PYTHON=.venv/bin/python` from the full
  checkout, or `make artifact-smoke PYTHON=python3` from a clean extracted ZIP.
- The finalization helper performs the post-upload manuscript/checklist URL
  replacement; avoid editing those fields manually unless the helper fails.
