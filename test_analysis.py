import math
import unittest

from analyse_scoring import analyse


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.attempts, cls.diagnostics = analyse()

    def test_weights_and_counts_are_consistent(self) -> None:
        self.assertTrue(math.isclose(sum(self.diagnostics["weights"].values()), 1.0))
        self.assertEqual(len(self.rows), self.diagnostics["team_count"])
        self.assertEqual(len(self.attempts), self.diagnostics["attempt_count"])
        self.assertEqual(
            self.diagnostics["valid_count"] + self.diagnostics["invalid_count"],
            self.diagnostics["attempt_count"],
        )

    def test_outcomes_are_classified_without_half_successes(self) -> None:
        for attempt in self.attempts:
            if attempt["status"] == "Valid":
                self.assertIn(attempt["outcome"], (0.0, 1.0))
            else:
                self.assertEqual(attempt["outcome"], 0.5)

    def test_rankings_and_intervals_are_well_formed(self) -> None:
        ranks = sorted(int(row["adjusted_rank"]) for row in self.rows)
        self.assertEqual(ranks, list(range(1, len(self.rows) + 1)))
        for row in self.rows:
            self.assertLessEqual(row["adjusted_total_low"], row["adjusted_total"])
            self.assertLessEqual(row["adjusted_total"], row["adjusted_total_high"])
            self.assertGreaterEqual(row["adjusted_total_low"], 0)
            self.assertLessEqual(row["adjusted_total_high"], 100)

    def test_winner_review_rule_matches_pairwise_interval(self) -> None:
        leaders = [row for row in self.rows if row["adjusted_rank"] == 1]
        self.assertEqual(len(leaders), 1)
        self.assertTrue(leaders[0]["winner_review"])
        for row in self.rows:
            expected = row["adjusted_rank"] == 1 or row["difference_from_leader_high"] >= 0
            self.assertEqual(row["winner_review"], expected)

    def test_peer_adjustment_is_bounded_and_mean_preserving(self) -> None:
        raw_mean = sum(float(row["peer"]) for row in self.rows) / len(self.rows)
        adjusted_mean = sum(float(row["adjusted_peer"]) for row in self.rows) / len(self.rows)
        self.assertAlmostEqual(raw_mean, adjusted_mean, places=12)
        for row in self.rows:
            self.assertLessEqual(0, row["peer_low"])
            self.assertLessEqual(row["peer_low"], row["adjusted_peer"])
            self.assertLessEqual(row["adjusted_peer"], row["peer_high"])
            self.assertLessEqual(row["peer_high"], 1)
            self.assertGreater(row["peer_ratings_given"], 0)
            self.assertGreater(row["peer_ratings_received"], 0)

    def test_peer_model_beats_cross_validated_mean_baseline(self) -> None:
        self.assertLess(self.diagnostics["peer_cv_rmse"], self.diagnostics["peer_baseline_rmse"])


if __name__ == "__main__":
    unittest.main()