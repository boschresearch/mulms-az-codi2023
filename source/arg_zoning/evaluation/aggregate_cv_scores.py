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
Calculates all metric scores across n (default: 5) folds.
"""
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from source.arg_zoning.evaluation.evaluation import calculate_average_scores

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--input_path", type=str, help="Path to directoy which all CV scores are stored in."
)
parser.add_argument("--num_folds", type=int, help="Only change if really necessary.", default=5)
parser.add_argument(
    "--set",
    type=str,
    choices=["dev", "test"],
    help="Determines which split to evaluate.",
    default="test",
)
parser.add_argument(
    "--export_as_latex_table",
    action="store_true",
    help="Whether to export scores as Latex table.",
    default=False,
)

args = parser.parse_args()


def main():
    """
    Entry point.
    """
    result_scores: list = []
    for fold in range(1, args.num_folds + 1):
        result_scores.append(
            np.load(
                os.path.join(args.input_path, f"cv_{fold}", f"scores_{args.set}.npz"),
                allow_pickle=True,
            )["arr_0"].item()
        )
    average_scores: dict = calculate_average_scores(result_scores)
    print("Final avg results are: ")
    print(f"  Average micro hierarchical precision: {round(average_scores['micro_h_p'] * 100, 1)}")
    print(f"  Average micro hierarchical recall: {round(average_scores['micro_h_r'] * 100, 1)}")
    print(f"  Average micro hierarchical f1: {round(average_scores['micro_h_f1'] * 100, 1)}")
    print(
        f"  Average micro hierarchical f1 standard deviation: {round(average_scores['micro_h_f1_std'] * 100, 1)}"
    )
    print(f"  Average macro hierarchical f1: {round(average_scores['macro_h_f1'] * 100, 1)}")
    print(
        f"  Average macro hierarchical f1 standard deviation: {round(average_scores['macro_h_f1_std'] * 100, 1)}"
    )
    print("\n\n Average labelwise scores:")
    print("\nLABEL\tPREC.\tRECALL\tF1")
    print("----------------------------------------")
    for label in average_scores["p_labelwise"]:
        print(
            f"{label[:7]}\t{round(average_scores['p_labelwise'][label] * 100, 1)}\t{round(average_scores['r_labelwise'][label] * 100, 1)}\t{round(average_scores['f1_labelwise'][label] * 100, 1)}"
        )

    if args.export_as_latex_table:
        df: pd.DataFrame = pd.DataFrame(
            {
                "H. Prec.": average_scores["p_labelwise"].values(),
                "H. Rec.": average_scores["r_labelwise"].values(),
                "H. F1": average_scores["f1_labelwise"].values(),
            },
            index=average_scores["f1_labelwise"].keys(),
        )
        df = df.append(
            pd.DataFrame(
                {
                    "H. Prec.": average_scores["micro_h_p"],
                    "H. Rec.": average_scores["micro_h_r"],
                    "H. F1": average_scores["micro_h_f1"],
                },
                index=["Micro Overall"],
            )
        )
        df = df.append(
            pd.DataFrame(
                {
                    "H. Prec.": average_scores["macro_h_p"],
                    "H. Rec.": average_scores["macro_h_r"],
                    "H. F1": average_scores["macro_h_f1"],
                },
                index=["Macro Overall"],
            )
        )
        df["H. Prec."] = df["H. Prec."].apply(lambda x: round(x * 100, 1))
        df["H. Rec."] = df["H. Rec."].apply(lambda x: round(x * 100, 1))
        df["H. F1"] = df["H. F1"].apply(lambda x: round(x * 100, 1))
        df.to_latex(os.path.join(args.input_path, f"scores_{args.set}.tex"))


if __name__ == "__main__":
    main()
