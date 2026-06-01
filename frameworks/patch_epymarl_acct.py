"""Patch an external EPyMARL checkout with ACCT smoke-test plumbing.

The patch is intentionally small and reversible by using a fresh EPyMARL
checkout.  It copies the framework-agnostic adapter into EPyMARL, adds optional
ACCT config fields, and inserts ACCT advantage transport after PPO/actor-critic
advantages are computed.  The patch supports a policy-confidence proxy smoke
mode and a tiny learned action-value head smoke mode.  Neither mode is a tuned
benchmark implementation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


METHOD_BEGIN = "# --- ACCT smoke methods patch begin ---"
METHOD_END = "# --- ACCT smoke methods patch end ---"
INIT_BEGIN = "# --- ACCT smoke init patch begin ---"
INIT_END = "# --- ACCT smoke init patch end ---"
ADV_BEGIN = "# --- ACCT smoke advantage patch begin ---"
ADV_END = "# --- ACCT smoke advantage patch end ---"
Q_TRAIN_BEGIN = "# --- ACCT smoke q-head train patch begin ---"
Q_TRAIN_END = "# --- ACCT smoke q-head train patch end ---"
Q_LOG_BEGIN = "# --- ACCT smoke q-head log patch begin ---"
Q_LOG_END = "# --- ACCT smoke q-head log patch end ---"
CONFIG_BEGIN = "# --- ACCT smoke config patch begin ---"
CONFIG_END = "# --- ACCT smoke config patch end ---"
IMPORT_LINE = (
    "from components.acct_epymarl_adapter import acct_epymarl_advantages, "
    "counterfactual_influence_from_replacements\n"
)
CONFIG_BLOCK = """
# --- ACCT smoke config patch begin ---
# ACCT smoke-test options. These defaults keep ordinary EPyMARL runs unchanged.
use_acct: False
acct_influence_source: "policy_confidence_proxy"
acct_lambda: 0.90
acct_residual_weight: 0.5
acct_transport_mode: "absolute"
acct_standardize: "masked"
acct_q_lr: 0.0003
# --- ACCT smoke config patch end ---
"""


def insert_once(text: str, marker: str, insertion: str) -> str:
    if insertion.strip() in text:
        return text
    if marker not in text:
        raise ValueError(f"marker not found: {marker!r}")
    return text.replace(marker, marker + insertion, 1)


def insert_before_once(text: str, marker: str, insertion: str) -> str:
    if insertion.strip() in text:
        return text
    if marker not in text:
        raise ValueError(f"marker not found: {marker!r}")
    return text.replace(marker, insertion + marker, 1)


def proxy_method_text() -> str:
    return f'''
    {METHOD_BEGIN}
    def _acct_proxy_influence(self, pi, actions, mask):
        """Policy-confidence proxy used only for ACCT framework smoke tests."""

        taken_prob = th.gather(pi, dim=3, index=actions).squeeze(3)
        uniform_prob = 1.0 / float(self.n_actions)
        return (taken_prob - uniform_prob) * mask

    def _acct_actions_onehot(self, actions):
        onehot = th.zeros(
            actions.shape[0],
            actions.shape[1],
            self.n_agents,
            self.n_actions,
            device=actions.device,
            dtype=th.float32,
        )
        return onehot.scatter_(3, actions.long(), 1.0)

    def _acct_q_head_values(self, batch, actions):
        if self.acct_q_head is None:
            raise RuntimeError("acct_q_head is not initialised")
        batch_size, time = actions.shape[0], actions.shape[1]
        state = batch["state"][:, :time].float()
        state = state.reshape(batch_size, time, -1)
        action_onehot = self._acct_actions_onehot(actions)
        joint_flat = action_onehot.reshape(batch_size, time, -1)
        agent_eye = th.eye(self.n_agents, device=actions.device).view(1, 1, self.n_agents, self.n_agents)

        state_by_agent = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        joint_by_agent = joint_flat.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        agent_by_agent = agent_eye.expand(batch_size, time, -1, -1)
        q_taken_inputs = th.cat([state_by_agent, joint_by_agent, agent_by_agent], dim=-1)
        q_taken = self.acct_q_head(q_taken_inputs).squeeze(-1)

        q_replacements = th.zeros(
            batch_size,
            time,
            self.n_agents,
            self.n_actions,
            device=actions.device,
            dtype=q_taken.dtype,
        )
        for agent_i in range(self.n_agents):
            agent_id = agent_eye[:, :, agent_i, :].expand(batch_size, time, -1)
            for action_i in range(self.n_actions):
                replaced = action_onehot.clone()
                replaced[:, :, agent_i, :] = 0.0
                replaced[:, :, agent_i, action_i] = 1.0
                replaced_flat = replaced.reshape(batch_size, time, -1)
                inputs = th.cat([state, replaced_flat, agent_id], dim=-1)
                q_replacements[:, :, agent_i, action_i] = self.acct_q_head(inputs).squeeze(-1)
        return q_taken, q_replacements

    def _train_acct_q_head(self, batch, target_returns, mask):
        if (
            self.acct_q_head is None
            or getattr(self.args, "acct_influence_source", "policy_confidence_proxy") != "learned_q_head"
        ):
            return {{}}
        actions = batch["actions"][:, : target_returns.shape[1]]
        q_taken, _ = self._acct_q_head_values(batch, actions)
        loss = (((q_taken - target_returns.detach()) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
        self.acct_q_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.acct_q_head.parameters(), self.args.grad_norm_clip)
        self.acct_q_optimiser.step()
        return {{"acct_q_loss": loss.item(), "acct_q_grad_norm": grad_norm.item()}}

    def _acct_influence(self, batch, pi, actions, mask):
        source = getattr(self.args, "acct_influence_source", "policy_confidence_proxy")
        if source == "policy_confidence_proxy":
            return self._acct_proxy_influence(pi, actions, mask)
        if source == "learned_q_head":
            with th.no_grad():
                q_taken, q_replacements = self._acct_q_head_values(batch, actions)
                influence = counterfactual_influence_from_replacements(q_taken, q_replacements, pi)
            return influence.to(pi.dtype) * mask
        raise ValueError(f"unknown acct_influence_source: {{source}}")
    {METHOD_END}

'''


def init_block_text() -> str:
    return f'''        {INIT_BEGIN}
        self.acct_q_head = None
        self.acct_q_optimiser = None
        if (
            getattr(args, "use_acct", False)
            and getattr(args, "acct_influence_source", "policy_confidence_proxy") == "learned_q_head"
        ):
            state_dim = args.state_shape
            if not isinstance(state_dim, int):
                state_dim = int(th.tensor(state_dim).prod().item())
            acct_q_input_dim = state_dim + self.n_agents * self.n_actions + self.n_agents
            self.acct_q_head = th.nn.Sequential(
                th.nn.Linear(acct_q_input_dim, args.hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(args.hidden_dim, 1),
            )
            self.acct_q_optimiser = Adam(
                params=self.acct_q_head.parameters(),
                lr=getattr(args, "acct_q_lr", args.lr),
            )
        {INIT_END}
'''


def acct_block_text(indent: str = "            ") -> str:
    return f'''{indent}{ADV_BEGIN}
{indent}if getattr(self.args, "use_acct", False):
{indent}    acct_influence = self._acct_influence(batch, pi, actions, mask)
{indent}    advantages = acct_epymarl_advantages(
{indent}        influence=acct_influence,
{indent}        td_residuals=advantages,
{indent}        mask=mask,
{indent}        gamma=self.args.gamma,
{indent}        lam=getattr(self.args, "acct_lambda", 0.90),
{indent}        residual_weight=getattr(self.args, "acct_residual_weight", 0.5),
{indent}        transport_mode=getattr(self.args, "acct_transport_mode", "absolute"),
{indent}        standardize=getattr(self.args, "acct_standardize", "masked"),
{indent}    ).detach()
{indent}{ADV_END}
'''


def q_train_block_text() -> str:
    return f'''        {Q_TRAIN_BEGIN}
        acct_q_stats = self._train_acct_q_head(batch, target_returns.detach(), mask)
        {Q_TRAIN_END}
'''


def q_stats_block_text() -> str:
    return f'''        {Q_TRAIN_BEGIN}
        for key, value in acct_q_stats.items():
            running_log.setdefault(key, []).append(value)
        {Q_TRAIN_END}
'''


def q_log_block_text() -> str:
    return f'''            {Q_LOG_BEGIN}
            for key in ["acct_q_loss", "acct_q_grad_norm"]:
                if key in critic_train_stats:
                    self.logger.log_stat(
                        key, sum(critic_train_stats[key]) / len(critic_train_stats[key]), t_env
                    )
            {Q_LOG_END}

'''


def patch_learner_text(text: str) -> str:
    text = insert_once(
        text,
        "from modules.critics import REGISTRY as critic_resigtry\n",
        IMPORT_LINE,
    )
    text = insert_once(
        text,
        "        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)\n",
        init_block_text(),
    )
    text = insert_before_once(
        text,
        "    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):\n",
        proxy_method_text(),
    )
    if "            advantages = advantages.detach()\n" in text:
        text = insert_once(text, "            advantages = advantages.detach()\n", acct_block_text(indent="            "))
    elif "        advantages = advantages.detach()\n" in text:
        text = insert_once(text, "        advantages = advantages.detach()\n", acct_block_text(indent="        "))
    else:
        raise ValueError("advantages detach marker not found")
    text = insert_once(text, "        self.critic_optimiser.step()\n", q_train_block_text())
    text = insert_once(
        text,
        '        running_log["target_mean"].append(\n            (target_returns * mask).sum().item() / mask_elems\n        )\n',
        q_stats_block_text(),
    )
    text = insert_before_once(
        text,
        '            self.logger.log_stat(\n                "advantage_mean",\n',
        q_log_block_text(),
    )
    return text


def patch_config_text(text: str) -> str:
    if CONFIG_BEGIN in text:
        return text
    if not text.endswith("\n"):
        text += "\n"
    return text + CONFIG_BLOCK


def apply_patch_to_epymarl(epymarl_root: Path, adapter_source: Path) -> list[Path]:
    epymarl_root = epymarl_root.resolve()
    adapter_source = adapter_source.resolve()
    if not (epymarl_root / "src" / "main.py").exists():
        raise FileNotFoundError(f"not an EPyMARL root: {epymarl_root}")
    if not adapter_source.exists():
        raise FileNotFoundError(adapter_source)

    adapter_target = epymarl_root / "src" / "components" / "acct_epymarl_adapter.py"
    shutil.copyfile(adapter_source, adapter_target)

    patched = [adapter_target]
    for rel in ("src/learners/ppo_learner.py", "src/learners/actor_critic_learner.py"):
        path = epymarl_root / rel
        original = path.read_text()
        updated = patch_learner_text(original)
        if updated != original:
            path.write_text(updated)
        patched.append(path)
    config_path = epymarl_root / "src" / "config" / "algs" / "mappo.yaml"
    original_config = config_path.read_text()
    updated_config = patch_config_text(original_config)
    if updated_config != original_config:
        config_path.write_text(updated_config)
    patched.append(config_path)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epymarl-root", type=Path, required=True)
    parser.add_argument(
        "--adapter-source",
        type=Path,
        default=Path(__file__).with_name("epymarl_acct_adapter.py"),
    )
    args = parser.parse_args()
    patched = apply_patch_to_epymarl(args.epymarl_root, args.adapter_source)
    for path in patched:
        print(path)


if __name__ == "__main__":
    main()
