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
Calculates the metric scores for the AZ transfer learning experiment.
"""
import os
from argparse import ArgumentParser

import numpy as np

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--input_path", type=str, help="Path to directoy which all CV scores are stored in."
)
parser.add_argument(
    "--set",
    type=str,
    choices=["dev", "test"],
    help="Determines which split to evaluate.",
    default="test",
)

args = parser.parse_args()


def main():
    """
    Entry point.
    """
    result_scores: dict = np.load(
        os.path.join(args.input_path, f"scores_{args.set}.npz"), allow_pickle=True
    )["arr_0"].item()
    scores_label_other: dict = result_scores.pop("OTHER")
    scores_target_label: dict = result_scores[list(result_scores.keys())[0]]

    print(f"Input Path: {args.input_path}")
    print(f"Scores for label {list(result_scores.keys())[0]}:")
    print(
        f"P: {scores_target_label['P']}, R: {scores_target_label['R']}, F1: {scores_target_label['F1']}"
    )
    print("----------------------------")
    print("Scores for label OTHER:")
    print(
        f"P: {scores_label_other['P']}, R: {scores_label_other['R']}, F1: {scores_label_other['F1']}"
    )


if __name__ == "__main__":
    main()
