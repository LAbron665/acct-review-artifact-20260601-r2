import unittest

from frameworks.patch_epymarl_acct import patch_config_text, patch_learner_text


BASE_LEARNER = """import torch as th
from modules.critics import REGISTRY as critic_resigtry


class PPOLearner:
    def __init__(self):
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        pi = None
        actions = None
        mask = None
        advantages = advantages.detach()
        if t_env:
            self.logger.log_stat(
                "advantage_mean",
                0.0,
                t_env,
            )

    def train_critic_sequential(self):
        self.critic_optimiser.step()
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
"""


class EPyMARLPatchTests(unittest.TestCase):
    def test_patch_inserts_import_method_and_acct_block(self):
        patched = patch_learner_text(BASE_LEARNER)
        self.assertIn("acct_epymarl_advantages", patched)
        self.assertIn("def _acct_proxy_influence", patched)
        self.assertIn("def _acct_q_head_values", patched)
        self.assertIn('getattr(self.args, "use_acct", False)', patched)
        self.assertIn("acct_influence_source", patched)

    def test_patch_is_idempotent(self):
        patched = patch_learner_text(BASE_LEARNER)
        repatched = patch_learner_text(patched)
        self.assertEqual(patched, repatched)

    def test_config_patch_adds_acct_defaults_once(self):
        patched = patch_config_text("name: mappo\n")
        self.assertIn("use_acct: False", patched)
        self.assertEqual(patched, patch_config_text(patched))


if __name__ == "__main__":
    unittest.main()
