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
This module contains constant variables used in the whole project.
"""

import os
from pathlib import Path

from source.constants.bilou_tags import *  # noqa: F401,F403

# General constants #

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DATASET_PATH: Path = PROJECT_ROOT.joinpath("./data")
MULMS_PATH: Path = DATASET_PATH.joinpath("./mulms_corpus")
CODE_PATH: Path = PROJECT_ROOT.joinpath("./source")
MULMS_DATASET_READER_PATH: Path = CODE_PATH.joinpath("./data_handling/mulms_dataset.py")

AZ_DATASET_PATHS: dict = {
    "PUBMED": DATASET_PATH.joinpath(
        "./additional_arg_zoning_datasets/materials_science_pubmed.pickle"
    ),
    "ART": DATASET_PATH.joinpath("./additional_arg_zoning_datasets/ART.csv"),
    "AZ-CL": DATASET_PATH.joinpath("./additional_arg_zoning_datasets/AZ-CL.csv"),
    "DRI": DATASET_PATH.joinpath("./additional_arg_zoning_datasets/DRI.csv"),
}

MULMS_FILES: list = [
    str(MULMS_PATH.joinpath(f"./xmi/{f}"))
    for f in os.listdir(MULMS_PATH.joinpath("./xmi"))
    if "TypeSystem" not in f
]

CPU: str = "cpu"
CUDA: str = "cuda"


# Argumentative Zoning related labels #
az_content_labels: list = [
    "Experiment",
    "Results",
    "Exp_Preparation",
    "Exp_Characterization",
    "Background_PriorWork",
    "Explanation",
    "Conclusion",
    "Motivation",
    "Background",
]
az_structure_labels: list = ["Metadata", "Caption", "Heading", "Abstract"]

az_label2id: dict = {
    "PADDING": 0,
    "Experiment": 1,
    "Preparation_Procedure": 2,
    "Results": 3,
    "Explanation_Hypothesis": 4,
    "Exp_Preparation": 5,
    "Exp_Characterization": 6,
    "Background_PriorWork": 7,
    "Background": 8,
    "Explanation_Assumption": 9,
    "Explanation": 10,
    "Conclusion": 11,
    "Motivation": 12,
    "Abstract": 13,
    "Metadata": 14,
    "Acknowledgment": 15,
    "Caption": 16,
    "Heading": 17,
    "Figure/Table": 18,
    "References": 19,
    "Captionc": 20,
}
az_id2label: dict = dict([(v, k) for k, v in az_label2id.items()])

# These are used for AZ training!
az_filtered_label2id: dict = {l: i for i, l in enumerate(az_content_labels + az_structure_labels)}
az_filtered_id2label: dict = dict([(v, k) for k, v in az_filtered_label2id.items()])

# Pubmed related labels
pubmed_to_az_mapping: dict = {
    "BACKGROUND": "Background",
    "OBJECTIVE": "Motivation",
    "RESULTS": "Results",
    "CONCLUSIONS": "Conclusion",
}

pubmed_labels: list = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
pubmed_label2id: dict = {l: i for i, l in enumerate(pubmed_labels)}
pubmed_id2label: dict = dict([(v, k) for k, v in pubmed_label2id.items()])

# Custom AZ Dataset related labels

art_labels: list = ["Met", "Exp", "Goa", "Mod", "Mot", "Obj", "Con", "Res", "Hyp", "Obs", "Bac"]
art_label2id: dict = {l: i for i, l in enumerate(art_labels)}
art_id2label: dict = dict([(v, k) for k, v in art_label2id.items()])

az_cl_labels: list = ["OTH", "CTR", "BAS", "OWN", "AIM", "BKG", "TXT"]
az_cl_label2id: dict = {l: i for i, l in enumerate(az_cl_labels)}
az_cl_id2label: dict = dict([(v, k) for k, v in az_cl_label2id.items()])

dri_labels: list = [
    "Sentence",
    "DRI_Background",
    "DRI_Outcome_Contribution",
    "DRI_Outcome",
    "DRI_Challenge",
    "DRI_Unspecified",
    "DRI_FutureWork",
    "DRI_Approach",
    "DRI_Challenge_Hypothesis",
]
dri_label2id: dict = {l: i for i, l in enumerate(dri_labels)}
dri_id2label: dict = dict([(v, k) for k, v in dri_label2id.items()])

ADDITIONAL_AZ_LABELS: dict = {
    "PUBMED": {
        "label2id": pubmed_label2id,
        "id2label": pubmed_id2label,
        "num_labels": len(pubmed_labels),
    },
    "ART": {"label2id": art_label2id, "id2label": art_id2label, "num_labels": len(art_labels)},
    "AZ-CL": {
        "label2id": az_cl_label2id,
        "id2label": az_cl_id2label,
        "num_labels": len(az_cl_labels),
    },
    "DRI": {"label2id": dri_label2id, "id2label": dri_id2label, "num_labels": len(dri_labels)},
}

# Named Entity related labels #
mulms_ne_labels: list = [
    "MAT",
    "NUM",
    "VALUE",
    "UNIT",
    "PROPERTY",
    "CITE",
    "TECHNIQUE",
    "RANGE",
    "INSTRUMENT",
    "SAMPLE",
    "FORM",
    "DEV",
    "MEASUREMENT",
]
mulms_ne_combined_dependency_labels: list = ["VALUE+NUM", "VALUE+RANGE", "MAT+FORM"]
mulms_ne_dependency_labels: list = mulms_ne_labels + mulms_ne_combined_dependency_labels
mulms_ne_label2id: dict = {l: i for i, l in enumerate(mulms_ne_labels)}
mulms_id2ne_label: dict = dict([(v, k) for k, v in mulms_ne_label2id.items()])

# Measurement labels #
meas_labels: list = ["MEASUREMENT", "QUAL_MEASUREMENT", "O"]
meas_label2id: dict = {l: i for i, l in enumerate(meas_labels)}
id2meas_label: dict = dict([(v, k) for k, v in meas_label2id.items()])

# Relation labels #
rel_labels: list = [
    "hasForm",
    "measuresProperty",
    "usedAs",
    "conditionProperty",
    "conditionSampleFeatures",
    "usesTechnique",
    "conditionEnvironment",
    "propertyValue",
    "usedIn",
    "conditionInstrument",
    "dopedBy",
    "takenFrom",
    "usedTogether",
]
rel_label_set: set = set(rel_labels)
