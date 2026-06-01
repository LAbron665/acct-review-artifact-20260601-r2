import unittest

import numpy as np

from acct.credit import (
    acct_advantages,
    counterfactual_influence,
    directional_temporal_credit_transport,
    return_equivalent_transport,
    sampled_agent_time_shapley,
    temporal_credit_transport,
    top_fraction_salience,
)
from acct.critics import fit_quadratic_reward_model
from acct.envs import DelayedLeverEnv
from acct.learners import TabularPolicy, _credits_for_method, run_training


class CreditTests(unittest.TestCase):
    def test_temporal_transport_conserves_td_mass(self):
        td = np.array([0.25, -0.1, 1.5])
        salience = np.array([[1.0, 0.5], [0.0, 2.0], [3.0, 1.0]])
        credits = temporal_credit_transport(td, salience, gamma=0.9, lam=0.8)
        self.assertAlmostEqual(float(credits.sum()), float(td.sum()), places=8)

    def test_temporal_transport_is_invariant_to_positive_salience_scale(self):
        td = np.array([0.25, -0.1, 1.5])
        salience = np.array([[1.0, 0.5], [0.0, 2.0], [3.0, 1.0]])
        original = temporal_credit_transport(td, salience, gamma=0.9, lam=0.8)
        scaled = temporal_credit_transport(td, 17.0 * salience, gamma=0.9, lam=0.8)
        self.assertTrue(np.allclose(original, scaled))

    def test_directional_transport_conserves_td_mass(self):
        td = np.array([0.25, -0.1, 1.5])
        influence = np.array([[1.0, -0.5], [0.0, -2.0], [3.0, 1.0]])
        credits = directional_temporal_credit_transport(td, influence, gamma=0.9, lam=0.8)
        self.assertAlmostEqual(float(credits.sum()), float(td.sum()), places=8)

    def test_directional_transport_respects_residual_sign_when_available(self):
        td = np.array([1.0])
        influence = np.array([[2.0, -3.0]])
        directional = directional_temporal_credit_transport(td, influence)
        absolute = temporal_credit_transport(td, np.abs(influence))
        self.assertGreater(directional[0, 0], 0.99)
        self.assertAlmostEqual(float(directional[0, 1]), 0.0, places=8)
        self.assertGreater(float(absolute[0, 1]), 0.0)

    def test_temporal_transport_uniform_fallback_conserves_td_mass(self):
        td = np.array([0.0, 1.0])
        salience = np.zeros((2, 2))
        credits = temporal_credit_transport(td, salience, gamma=0.9, lam=0.8)
        self.assertAlmostEqual(float(credits.sum()), 1.0, places=8)
        self.assertTrue(np.allclose(credits, 0.25))

    def test_return_equivalent_transport_sums_to_team_return(self):
        salience = np.array([[1.0, 0.0], [2.0, 1.0]])
        credits = return_equivalent_transport(1.75, salience, gamma=0.9, lam=0.8)
        self.assertEqual(credits.shape, salience.shape)
        self.assertAlmostEqual(float(credits.sum()), 1.75, places=8)

    def test_directional_transport_falls_back_to_absolute_salience(self):
        td = np.array([1.0])
        influence = np.array([[-2.0, -6.0]])
        directional = directional_temporal_credit_transport(td, influence)
        absolute = temporal_credit_transport(td, np.abs(influence))
        self.assertTrue(np.allclose(directional, absolute))
        self.assertAlmostEqual(float(directional.sum()), 1.0, places=8)

    def test_top_fraction_salience_keeps_largest_entries(self):
        salience = np.array([[0.1, -4.0], [2.0, 0.3]])
        pruned = top_fraction_salience(salience, 0.5)
        self.assertTrue(np.allclose(pruned, np.array([[0.0, -4.0], [2.0, 0.0]])))

    def test_top_fraction_salience_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            top_fraction_salience(np.ones((2, 2)), 0.0)

    def test_transport_only_uses_past_agent_time_pairs(self):
        td = np.array([0.0, 0.0, 1.0, 0.0])
        salience = np.ones((4, 2))
        credits = temporal_credit_transport(td, salience)
        self.assertTrue(np.allclose(credits[3], 0.0))
        self.assertAlmostEqual(float(credits[:3].sum()), 1.0, places=8)

    def test_zero_salience_pairs_receive_no_transport_when_mass_exists(self):
        td = np.array([0.0, 1.0, 2.0])
        salience = np.array([[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]])
        credits = temporal_credit_transport(td, salience)
        self.assertTrue(np.allclose(credits[:, 1], 0.0))
        self.assertAlmostEqual(float(credits[:, 0].sum()), float(td.sum()), places=8)

    def test_counterfactual_action_replacement(self):
        actions = np.array([[1, 0]])
        probs = np.full((1, 2, 2), 0.5)

        def reward_fn(a):
            return float(a[0, 0] + 2 * a[0, 1])

        influence = counterfactual_influence(reward_fn, actions, probs)
        self.assertEqual(influence.shape, (1, 2))
        self.assertAlmostEqual(influence[0, 0], 0.5)
        self.assertAlmostEqual(influence[0, 1], -1.0)

    def test_acct_beta_zero_recovers_instantaneous_influence(self):
        influence = np.array([[0.25, -0.5], [1.0, 0.0]])
        td = np.array([0.3, 1.2])
        credits = acct_advantages(influence, td, residual_weight=0.0)
        self.assertTrue(np.allclose(credits, influence))

    def test_acct_combined_credit_is_not_globally_scale_invariant(self):
        influence = np.array([[0.25, -0.5], [1.0, 0.0]])
        td = np.array([0.3, 1.2])
        original = acct_advantages(influence, td, residual_weight=0.75)
        scaled = acct_advantages(3.0 * influence, td, residual_weight=0.75)
        self.assertFalse(np.allclose(original, scaled))

    def test_counterfactual_influence_recovers_additive_advantages(self):
        actions = np.array([[1, 0], [0, 1]])
        probs = np.full((2, 2, 2), 0.5)
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])

        def reward_fn(a):
            return float((weights * a).sum())

        influence = counterfactual_influence(reward_fn, actions, probs)
        expected = weights * (actions - 0.5)
        self.assertTrue(np.allclose(influence, expected))

    def test_sampled_shapley_conserves_reference_gap_for_additive_reward(self):
        actions = np.array([[1, 1]])
        probs = np.array([[[1.0, 0.0], [1.0, 0.0]]])

        def reward_fn(a):
            return float(a[0, 0] + 2 * a[0, 1])

        credits = sampled_agent_time_shapley(reward_fn, actions, probs, samples=8, rng=np.random.default_rng(5))
        self.assertEqual(credits.shape, actions.shape)
        self.assertAlmostEqual(float(credits.sum()), 3.0)

    def test_actor_credit_shape_compatibility(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=7)
        actions = policy.sample()
        reward = env.reward(actions)
        credits = _credits_for_method(env, policy, actions, reward, baseline=0.0, method="acct")
        self.assertEqual(credits.shape, actions.shape)
        old_shape = policy.logits.shape
        policy.update(actions, credits, lr=0.01)
        self.assertEqual(policy.logits.shape, old_shape)

    def test_uniform_transport_method_shape(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=13)
        actions = policy.sample()
        reward = env.reward(actions)
        credits = _credits_for_method(env, policy, actions, reward, baseline=0.0, method="uniform_transport")
        self.assertEqual(credits.shape, actions.shape)
        self.assertAlmostEqual(float(credits.sum()), reward)

    def test_cf_transport_only_conserves_advantage(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=19)
        actions = policy.sample()
        reward = env.reward(actions)
        baseline = 0.25
        credits = _credits_for_method(env, policy, actions, reward, baseline=baseline, method="cf_transport")
        self.assertEqual(credits.shape, actions.shape)
        self.assertAlmostEqual(float(credits.sum()), reward - baseline, places=8)

    def test_return_equivalent_cf_transport_conserves_return(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=23)
        actions = policy.sample()
        reward = env.reward(actions)
        credits = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline=0.5,
            method="return_eq_cf_transport",
        )
        self.assertEqual(credits.shape, actions.shape)
        self.assertAlmostEqual(float(credits.sum()), reward, places=8)

    def test_agent_time_cf_keeps_early_influence(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=3)
        actions = np.zeros((env.horizon, env.n_agents), dtype=np.int64)
        actions[0, 0] = 1
        reward = env.reward(actions)
        final_only = _credits_for_method(env, policy, actions, reward, baseline=0.0, method="final_step_cf")
        agent_time = _credits_for_method(env, policy, actions, reward, baseline=0.0, method="agent_time_cf")
        self.assertAlmostEqual(float(final_only[0, 0]), 0.0)
        self.assertGreater(abs(float(agent_time[0, 0])), 0.0)

    def test_learned_critic_method_shape(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=11)
        critic = fit_quadratic_reward_model(env, seed=11, samples=128)
        actions = policy.sample()
        reward = env.reward(actions)
        credits = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline=0.0,
            method="learned_acct",
            critic_reward_fn=critic.predict,
        )
        self.assertEqual(credits.shape, actions.shape)
        self.assertTrue(np.all(np.isfinite(credits)))

    def test_learned_pruned_acct_method_shape(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=31)
        critic = fit_quadratic_reward_model(env, seed=31, samples=128)
        actions = policy.sample()
        reward = env.reward(actions)
        credits = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline=0.0,
            method="learned_pruned_acct",
            critic_reward_fn=critic.predict,
            salience_fraction=0.2,
        )
        self.assertEqual(credits.shape, actions.shape)
        self.assertTrue(np.all(np.isfinite(credits)))

    def test_learned_pruned_acct_training_accepts_fraction(self):
        env = DelayedLeverEnv()
        rows = run_training(
            env,
            method="learned_pruned_acct",
            seed=37,
            episodes=4,
            eval_every=2,
            critic_samples=64,
            salience_fraction=0.3,
        )
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row["method"] == "learned_pruned_acct" for row in rows))

    def test_acct_transport_parameters_are_configurable(self):
        env = DelayedLeverEnv()
        policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=17)
        actions = policy.sample()
        reward = env.reward(actions)
        short_transport = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline=0.0,
            method="acct",
            transport_lam=0.0,
            residual_weight=0.25,
        )
        long_transport = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline=0.0,
            method="acct",
            transport_lam=0.99,
            residual_weight=1.25,
        )
        self.assertEqual(short_transport.shape, actions.shape)
        self.assertEqual(long_transport.shape, actions.shape)
        self.assertFalse(np.allclose(short_transport, long_transport))

    def test_acct_directional_transport_mode_is_configurable(self):
        influence = np.array([[2.0, -3.0]])
        td = np.array([1.0])
        absolute = acct_advantages(influence, td, residual_weight=1.0, transport_mode="absolute")
        directional = acct_advantages(influence, td, residual_weight=1.0, transport_mode="directional")
        self.assertFalse(np.allclose(absolute, directional))
        self.assertGreater(directional[0, 0] - absolute[0, 0], 0.0)
        self.assertLess(directional[0, 1] - absolute[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
