# Experiment resources related to the MuLMS-AZ corpus (CODI 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
This module verifies correctness of P, R, and F1 calculation functions.
"""

import unittest

from source.arg_zoning.evaluation.evaluation import calculate_average_scores
from source.constants.constants import az_filtered_id2label


class EvaluationTester(unittest.TestCase):
    def test_average_calculation(self):

        # Just for testing purposes, the values are not correctly correlated
        scores: list[dict[str, float]] = [
            {
                "avg_loss": 0.15,
                "micro_h_f1": 0.77,
                "micro_h_r": 0.75,
                "micro_h_p": 0.71,
                "macro_h_f1": 0.71,
                "macro_h_p": 0.77,
                "macro_h_r": 0.67,
                "p_labelwise": [0.1, 0.7, 0.9],
                "r_labelwise": [0.2, 0.4, 0.54],
                "f1_labelwise": [0.13, 0.50, 0.61],
            },
            {
                "avg_loss": 0.19,
                "micro_h_f1": 0.67,
                "micro_h_r": 0.70,
                "micro_h_p": 0.63,
                "macro_h_f1": 0.72,
                "macro_h_p": 0.89,
                "macro_h_r": 0.57,
                "p_labelwise": [0.14, 0.12, 0.55],
                "r_labelwise": [0.22, 0.49, 0.90],
                "f1_labelwise": [0.17, 0.50, 0.67],
            },
            {
                "avg_loss": 0.33,
                "micro_h_f1": 0.87,
                "micro_h_r": 0.79,
                "micro_h_p": 0.65,
                "macro_h_f1": 0.71,
                "macro_h_p": 1.0,
                "macro_h_r": 0.89,
                "p_labelwise": [0.17, 0.99, 0.67],
                "r_labelwise": [0.27, 0.64, 0.10],
                "f1_labelwise": [0.09, 0.50, 0.55],
            },
        ]

        calculated_avg_scores: dict[str, float] = calculate_average_scores(scores)
        true_avg_scores: dict[str, float] = {
            "avg_loss": 0.2233333333333333,
            "micro_h_f1": 0.77,
            "micro_h_r": 0.7466666666666667,
            "micro_h_p": 0.6633333333333332,
            "macro_h_f1": 0.7133333333333333,
            "macro_h_p": 0.8866666666666667,
            "macro_h_r": 0.71,
        }
        for k in true_avg_scores.keys():
            self.assertAlmostEqual(calculated_avg_scores[k], true_avg_scores[k], places=6)

        true_labelwise_scores: dict[str, list[float]] = {
            "p_labelwise": [0.1366666666666667, 0.6033333333333334, 0.7066666666666667],
            "r_labelwise": [0.23, 0.51, 0.5133333333333333],
            "f1_labelwise": [0.13, 0.5, 0.61],
        }

        for k in ["p_labelwise", "r_labelwise", "f1_labelwise"]:
            for i in range(len(true_labelwise_scores["p_labelwise"])):
                self.assertAlmostEqual(
                    calculated_avg_scores[k][az_filtered_id2label[i]],
                    true_labelwise_scores[k][i],
                    places=6,
                )
