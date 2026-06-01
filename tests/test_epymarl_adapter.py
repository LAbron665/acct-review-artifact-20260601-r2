import unittest

import torch

from frameworks.epymarl_acct_adapter import (
    acct_epymarl_advantages,
    counterfactual_influence_from_replacements,
    expand_agent_dim,
)


class EPyMARLAdapterTests(unittest.TestCase):
    def test_expand_common_residual_to_agents(self):
        values = torch.ones(2, 3, 1)
        expanded = expand_agent_dim(values, n_agents=4, name="values")
        self.assertEqual(expanded.shape, (2, 3, 4))
        self.assertTrue(torch.allclose(expanded[..., 0], values.squeeze(-1)))

    def test_counterfactual_influence_from_replacements(self):
        q_taken = torch.tensor([[[2.0]]])
        q_replacements = torch.tensor([[[[1.0, 3.0], [4.0, 0.0]]]])
        action_probs = torch.full_like(q_replacements, 0.5)
        influence = counterfactual_influence_from_replacements(q_taken, q_replacements, action_probs)
        self.assertEqual(influence.shape, (1, 1, 2))
        self.assertTrue(torch.allclose(influence, torch.tensor([[[0.0, 0.0]]])))

    def test_acct_epymarl_advantages_shape_and_mask(self):
        influence = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0], [0.5, 0.0]],
                [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
            ]
        )
        residuals = torch.zeros(2, 3, 1)
        residuals[:, -1, 0] = 1.0
        mask = torch.ones(2, 3, 1)
        mask[1, -1, 0] = 0.0
        credits = acct_epymarl_advantages(
            influence,
            residuals,
            mask,
            gamma=0.9,
            lam=0.8,
            residual_weight=0.5,
            standardize="none",
        )
        self.assertEqual(credits.shape, influence.shape)
        self.assertTrue(torch.allclose(credits[1, -1], torch.zeros(2)))

    def test_directional_mode_differs_from_absolute_mode(self):
        influence = torch.tensor([[[2.0, -3.0]]])
        residuals = torch.tensor([[[1.0]]])
        mask = torch.ones(1, 1, 1)
        absolute = acct_epymarl_advantages(
            influence,
            residuals,
            mask,
            residual_weight=1.0,
            transport_mode="absolute",
            standardize="none",
        )
        directional = acct_epymarl_advantages(
            influence,
            residuals,
            mask,
            residual_weight=1.0,
            transport_mode="directional",
            standardize="none",
        )
        self.assertGreater(float(directional[0, 0, 0] - absolute[0, 0, 0]), 0.0)
        self.assertLess(float(directional[0, 0, 1] - absolute[0, 0, 1]), 0.0)

    def test_masked_standardization_zero_mean_on_valid_entries(self):
        influence = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        residuals = torch.tensor([[[0.0], [1.0]]])
        mask = torch.ones(1, 2, 1)
        credits = acct_epymarl_advantages(influence, residuals, mask, standardize="masked")
        self.assertAlmostEqual(float(credits.mean()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
