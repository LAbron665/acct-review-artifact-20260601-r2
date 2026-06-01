# EPyMARL ACCT Adapter Notes

This directory contains a shape-safe adapter for inserting ACCT-style credits
into EPyMARL without vendoring or modifying EPyMARL source code in this
artifact.

## Inspected EPyMARL Insertion Points

The adapter was written after inspecting EPyMARL's learner interfaces in a
temporary clone of `uoe-agents/epymarl`.

- `src/learners/ppo_learner.py`: `advantages` is returned by
  `train_critic_sequential(...)` and then used in `surr1 = ratios * advantages`.
- `src/learners/actor_critic_learner.py`: `advantages` is returned by
  `train_critic_sequential(...)` and then used in
  `advantages * log_pi_taken`.

The lowest-risk ACCT insertion point is immediately after the existing
`advantages = advantages.detach()` line.  At that point EPyMARL already has
`mask`, policy probabilities `pi`, taken actions, and critic residuals.  A
counterfactual Q replacement head can produce signed influence with shape
`[batch, time, agents]`; the adapter then returns an EPyMARL-shaped advantage
tensor.

## Minimal Patch Sketch

```python
from frameworks.epymarl_acct_adapter import acct_epymarl_advantages

# after EPyMARL computes `advantages = advantages.detach()`
if getattr(self.args, "use_acct", False):
    acct_adv = acct_epymarl_advantages(
        influence=counterfactual_influence,  # [B, T, N]
        td_residuals=advantages,             # [B, T, N]
        mask=mask,                           # [B, T, N]
        gamma=self.args.gamma,
        lam=getattr(self.args, "acct_lambda", 0.90),
        residual_weight=getattr(self.args, "acct_residual_weight", 0.5),
        transport_mode=getattr(self.args, "acct_transport_mode", "absolute"),
        standardize="masked",
    )
    advantages = acct_adv.detach()
```

The included adapter smoke tests use synthetic tensors with the same rank and
mask conventions as EPyMARL learners.  This is an integration scaffold, not a
claim that an EPyMARL benchmark has been run.

## Local Smoke Test

`experiments/run_epymarl_smoke.py` runs a tiny external-EPyMARL MAPPO/LBF smoke
test when provided with an EPyMARL checkout:

```bash
PYTHONPATH=. python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl
```

The recorded local summary in `results/epymarl_smoke.json` completed on EPyMARL
commit `cbc38c09588064eab978501d0f12c2cf58fa7fc2` with
`env_args.key=lbforaging:Foraging-5x5-2p-1f-v3`, `time_limit=5`, and `t_max=20`.
This verifies that the inspected EPyMARL MAPPO learner stack can execute on the
local machine.  It is not a tuned baseline and is not reported as a benchmark
result.

The artifact also includes a patched ACCT smoke command:

```bash
PYTHONPATH=. python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl --apply-acct-patch --use-acct --out results/epymarl_acct_smoke.json
```

`frameworks/patch_epymarl_acct.py` copies the adapter into the external
checkout, adds ACCT defaults to `mappo.yaml`, and inserts the ACCT call after
`advantages = advantages.detach()` in the PPO and actor-critic learners.  This
smoke mode deliberately uses a policy-confidence proxy influence, so it verifies
that the ACCT transport tensor can enter EPyMARL's actor-loss path but does not
replace the future learned counterfactual Q replacement head.

The learned-head smoke command exercises a slightly stronger path:

```bash
PYTHONPATH=. python experiments/run_epymarl_smoke.py --epymarl-root /path/to/epymarl --apply-acct-patch --use-acct --acct-influence-source learned_q_head --standardise-rewards False --acct-residual-weight 0.1 --out results/epymarl_acct_qhead_smoke.json
```

In this mode, the patch trains a tiny actual-action value head on the same
target returns used by the centralized critic, then queries replacement actions
to form counterfactual influence.  It is an integration smoke test for the
counterfactual-head pathway, not a tuned EPyMARL benchmark.

The paired comparison command runs the same patched checkout for a tiny matched
seed stress test:

```bash
PYTHONPATH=. python experiments/run_epymarl_comparison.py --epymarl-root /path/to/epymarl --t-max 60 --time-limit 5 --seeds 0 1 2 --standardise-rewards False
```

The recorded local CSVs compare MAPPO with the learned-head ACCT path over
three seeds on `lbforaging:Foraging-5x5-2p-1f-v3`.  MAPPO obtains nonzero
sample return in two seeds, while the learned-head ACCT path obtains zero
sample and greedy return in all three.  We include this as a negative
integration diagnostic for the current smoke head, not as a benchmark result.

## Required Future Work

- Replace the tiny smoke-test value head with a tuned recurrent counterfactual
  Q replacement head in EPyMARL's PPO/MAPPO critic.
- Run tuned EPyMARL baselines and ACCT variants on MPE/LBF/RWARE under the same
  config and seed budget.
- Report only completed EPyMARL runs in the paper.
