PYTHON ?= python3
LATEXMK ?= latexmk
PAPER_TEX ?= ../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex
PYTHONPYCACHEPREFIX ?= /tmp/acct_pycache
MPLCONFIGDIR ?= /tmp/acct_mpl
XDG_CACHE_HOME ?= /tmp/acct_cache

export PYTHONPATH := .
export PYTHONPYCACHEPREFIX
export MPLCONFIGDIR
export XDG_CACHE_HOME

.PHONY: test artifact-smoke data-budget medium-benchmarks paper-figures latex-audit paper-audit submission-readiness package verify release-repo finalize-anonymous-url release-check

test:
	"$(PYTHON)" -m unittest discover tests

artifact-smoke: test
	"$(PYTHON)" experiments/audit_anonymity.py
	"$(PYTHON)" experiments/analyze_query_budget.py
	"$(PYTHON)" experiments/analyze_data_budget.py
	"$(PYTHON)" experiments/analyze_learning_efficiency.py
	"$(PYTHON)" experiments/audit_dependency_licenses.py
	"$(PYTHON)" experiments/analyze_learned_salience_pruning.py
	"$(PYTHON)" experiments/analyze_mpe_statistics.py \
		--input results/mpe_mappo_results.csv \
		--out dist/_verification_mpe_mappo_stat_tests.csv \
		--baseline-method mappo_shared_gae \
		--episode 300

data-budget:
	"$(PYTHON)" experiments/analyze_data_budget.py

medium-benchmarks:
	"$(PYTHON)" experiments/run_medium_benchmark_suite.py --skip-existing

paper-figures:
	"$(PYTHON)" experiments/plot_paper_figures.py

latex-audit: paper-figures
	"$(LATEXMK)" -cd -pdf -interaction=nonstopmode -halt-on-error "$(PAPER_TEX)"
	"$(PYTHON)" experiments/check_latex_submission.py

paper-audit: data-budget
	"$(PYTHON)" experiments/audit_anonymity.py
	"$(PYTHON)" experiments/audit_bibliography.py
	"$(PYTHON)" experiments/audit_scope_guard.py
	"$(PYTHON)" experiments/analyze_learned_salience_pruning.py
	"$(PYTHON)" experiments/audit_paper_claims.py

submission-readiness:
	"$(PYTHON)" experiments/audit_submission_readiness.py

package:
	"$(PYTHON)" package_anonymous_artifact.py

verify:
	"$(PYTHON)" experiments/verify_anonymous_artifact.py

release-repo: package
	"$(PYTHON)" prepare_anonymous_release_repo.py --force

finalize-anonymous-url:
	"$(PYTHON)" finalize_anonymous_url.py --url "$(ANONYMOUS_URL)"

release-check: latex-audit paper-audit test package verify
