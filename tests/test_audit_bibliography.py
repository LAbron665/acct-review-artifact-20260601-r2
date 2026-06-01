import unittest

from experiments.audit_bibliography import citation_keys, has_locator, is_arxiv_entry, parse_bib_entries


class BibliographyAuditTests(unittest.TestCase):
    def test_citation_keys_support_natbib_variants(self):
        text = r"""
        \citep[see][Sec.~2]{rashid2018qmix, son2019qtran}
        \citet{foerster2018coma}
        \citealp{wang2021qplex}
        \citeauthor*{ellis2023smacv2}
        \citeyearpar{samvelyan2019smac}
        """
        self.assertEqual(
            citation_keys(text),
            {
                "ellis2023smacv2",
                "foerster2018coma",
                "rashid2018qmix",
                "samvelyan2019smac",
                "son2019qtran",
                "wang2021qplex",
            },
        )

    def test_parse_bib_entries_extracts_basic_fields(self):
        bib = """
        @article{xiao2022arel,
          title={Agent-Temporal Attention for Reward Redistribution},
          journal={arXiv preprint arXiv:2201.04612},
          year={2022},
          url={https://arxiv.org/abs/2201.04612}
        }

        @inproceedings{ellis2023smacv2,
          title={{SMACv2}: An Improved Benchmark},
          booktitle={Advances in Neural Information Processing Systems},
          year={2023}
        }
        """
        entries = parse_bib_entries(bib)

        self.assertEqual(entries["xiao2022arel"]["entry_type"], "article")
        self.assertEqual(entries["ellis2023smacv2"]["entry_type"], "inproceedings")
        self.assertTrue(is_arxiv_entry(entries["xiao2022arel"]))
        self.assertTrue(has_locator(entries["xiao2022arel"]))
        self.assertFalse(is_arxiv_entry(entries["ellis2023smacv2"]))


if __name__ == "__main__":
    unittest.main()
