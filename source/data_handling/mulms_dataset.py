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
This module contains the HuggingFace dataset reader for the "Multi-Layer Materials Science Corpus (MuLMS)"
"""

from dataclasses import dataclass
from os.path import exists, join

import datasets
import pandas as pd
from datasets.utils.download_manager import DownloadManager
from puima.collection_utils import DocumentCollection

from source.constants.constants import (
    az_content_labels,
    az_structure_labels,
    meas_labels,
    mulms_ne_labels,
    rel_label_set,
)
from source.constants.uimatypes import *  # noqa: F403
from source.data_handling.util import (
    get_token_index_for_annot_if_subtoken,
    get_token_indices_for_annot,
)

ne_labels_set = set(mulms_ne_labels)
all_az_labels = set(az_content_labels) | set(az_structure_labels)
meas_labels_set = set(meas_labels)

_CITATION = """\
@InProceedings{schrader-etal-2023-mulms,
title = {MuLMS-AZ: An Argumentative Zoning Dataset for the Materials Science Domain},
author={Timo Pierre Schrader, Teresa Bürkle, Sophie Henning, Sherry Tan, Matteo Finco, Stefan Grünewald, Maira Indrikova, Felix Hildebrand, Annemarie Friedrich
},
year={2023}
},
@InProceedings{schrader-etal-2023-mulms,
title = {MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain},
author={Timo Pierre Schrader, Matteo Finco, Stefan Grünewald, Felix Hildebrand, Annemarie Friedrich
},
year={2023}
}
"""

_DESCRIPTION = """\
This dataset represents the Multi-Layer Material Science (MuLMS) corpus.
It consists of 50 thoroughly annotated documents from the materials science domain and
provides annotations for named entities, argumentative zoning (AZ), relation extraction,
measurement classification and citation context retrieval. Please refer to our papers for
more details about the MuLMS corpus.
"""

_HOMEPAGE = "https://github.com/boschresearch/mulms-az-codi2023"

_LICENSE = "AGPL-3"

_URLS = {"MuLMS_Corpus": "../../data/mulms_corpus/xmi"}  # "Path to MuLMS-AZ files"


@dataclass
class MuLMSDatasetBuilderConfig(datasets.BuilderConfig):
    """
    Config class for the dataset class.
    """

    replace_heading_AZ_labels: bool = True
    remove_figure_and_table_labels: bool = True


class MuLMSDataset(datasets.GeneratorBasedBuilder):
    """This dataset represents the Multi-Layer Material Science Corpus with 50 documents across multiple domains."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = MuLMSDatasetBuilderConfig

    BUILDER_CONFIGS = [
        MuLMSDatasetBuilderConfig(
            name="MuLMS_Corpus",
            version=VERSION,
            description="This part of the dataset covers all annotations.",
        ),
        datasets.BuilderConfig(
            name="NER_Dependencies",
            version=VERSION,
            description="This part of the dataset represents Named Entities as dependencies (returned in CONLL format).",
        ),
    ]

    DEFAULT_CONFIG_NAME = "MuLMS_Corpus"

    AZ_HEADING_REPLACEMENT_LABELS = [
        "Supporting Information",
        "Author Contribution",
        "Confict of Interest",
        "Acknowledgment",
    ]

    def _info(self) -> datasets.DatasetInfo:
        """
        Provides information about this dataset.

        Returns:
            datasets.DatasetInfo
        """
        if self.config.name == "default":
            self.config.name = self.DEFAULT_CONFIG_NAME
        if self.config.name == "MuLMS_Corpus":
            features: datasets.Features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "beginOffset": datasets.Value("int32"),
                    "endOffset": datasets.Value("int32"),
                    "AZ_labels": datasets.Value("string"),
                    "Measurement_label": datasets.Value("string"),
                    "NER_labels": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "id": datasets.Value("int32"),
                            "value": datasets.Value("string"),
                            "begin": datasets.Value("string"),
                            "end": datasets.Value("string"),
                            "tokenIndices": datasets.Sequence(datasets.Value("int32")),
                        }
                    ),
                    "NER_labels_BILOU": datasets.Sequence(datasets.Value("string")),
                    "relations": datasets.Sequence(
                        {
                            "ne_id_gov": datasets.Value("int32"),
                            "ne_id_dep": datasets.Value("int32"),
                            "label": datasets.Value("string"),
                        }
                    ),
                    "docFileName": datasets.Value("string"),
                    "data_split": datasets.Value("string"),
                    "category": datasets.Value("string"),
                }
            )
        elif self.config.name == "NER_Dependencies":
            features: datasets.Features = datasets.Features(
                {
                    "ID": datasets.Value("int32"),
                    "sentence": datasets.Value("string"),
                    "token_id": datasets.Value("int32"),
                    "token_text": datasets.Value("string"),
                    "NE_Dependencies": datasets.Value("string"),
                    "data_split": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list:
        """
        Downloads files from URL or reads them from the file system and provides _generate_examples
        with necessary information.

        Args:
            dl_manager (DownloadManager): Handles data retrieval

        Returns:
            list: Information about files and splits
        """
        if self.config.data_files is not None:
            data_files: list = dl_manager.download_and_extract(self.config.data_files)["train"]
        else:
            urls = _URLS[self.config.name]
            data_files: list = dl_manager.download_and_extract(urls)
        assert exists(
            join(self.config.data_dir, "MuLMS_Corpus_Metadata.csv")
        ), "MuLMS_Corpus_Metadata.csv is missing."

        if "/" in data_files[0]:
            data_files = [f.split("/")[-1] for f in data_files]
        if "\\" in data_files[0]:
            data_files = [f.split("\\")[-1] for f in data_files]

        metadata_df: pd.DataFrame = pd.read_csv(
            join(self.config.data_dir, "MuLMS_Corpus_Metadata.csv")
        )
        train_files: list = sorted(
            [
                f
                for f in data_files
                if any(
                    name in f
                    for name in list(metadata_df[metadata_df["set"].str.contains("train")]["name"])
                )
            ]
        )
        dev_files: list = sorted(
            [
                f
                for f in data_files
                if any(
                    name in f for name in list(metadata_df[metadata_df["set"] == "dev"]["name"])
                )
            ]
        )
        test_files: list = sorted(
            [
                f
                for f in data_files
                if any(
                    name in f for name in list(metadata_df[metadata_df["set"] == "test"]["name"])
                )
            ]
        )

        if self.config.name == "MuLMS_Corpus":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "xmi"),
                        "files": train_files,
                        "data_split": metadata_df[metadata_df["set"].str.contains("train")][
                            ["name", "set", "category"]
                        ],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "xmi"),
                        "files": dev_files,
                        "data_split": metadata_df[metadata_df["set"] == "dev"][
                            ["name", "set", "category"]
                        ],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "xmi"),
                        "files": test_files,
                        "data_split": metadata_df[metadata_df["set"] == "test"][
                            ["name", "set", "category"]
                        ],
                    },
                ),
            ]

        elif self.config.name == "NER_Dependencies":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "ne_dependencies_conll"),
                        "files": [
                            "ne_deps_train1.conllu",
                            "ne_deps_train2.conllu",
                            "ne_deps_train3.conllu",
                            "ne_deps_train4.conllu",
                            "ne_deps_train5.conllu",
                        ],
                        "data_split": None,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "ne_dependencies_conll"),
                        "files": ["ne_deps_dev.conllu"],
                        "data_split": None,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "dir": join(self.config.data_dir, "ne_dependencies_conll"),
                        "files": ["ne_deps_test.conllu"],
                        "data_split": None,
                    },
                ),
            ]

    def _generate_examples(self, dir: str, files: list, data_split: pd.DataFrame):
        """
        Yields the data during runtime.

        Args:
            dir (str): Path to downloaded or local files.
            files (list): List of filenames corresponding to the current split; must be contained within "dir"
            data_split (pd.DataFrame): Category and train/dev/test split info for each document

        Yields:
            tuple: Yields document ID and dictionary with current data sample
        """

        if self.config.name == "MuLMS_Corpus":

            doc_coll: DocumentCollection = DocumentCollection(xmi_dir=dir, file_list=files)

            split_info: str = None
            category_info: str = None

            for doc_name in doc_coll.docs:
                doc = doc_coll.docs[doc_name]
                for sent_id, sent_annot in enumerate(
                    doc.select_annotations(SENT_TYPE)  # noqa: F405
                ):
                    sent_text = doc.get_covered_text(sent_annot)

                    # Argumentative Zoning labels
                    az_labels = set()
                    for matsci_sent in doc.select_covered(
                        MATSCI_SENT_TYPE, sent_annot  # noqa: F405
                    ):
                        content_info = matsci_sent.get_feature_value("ContentInformation")
                        struct_info = matsci_sent.get_feature_value("StructureInformation")
                        az_labels.add(content_info)
                        az_labels.add(struct_info)
                    if "None" in az_labels:
                        az_labels.remove("None")
                    if None in az_labels:
                        az_labels.remove(None)
                    # Extract Measurement related label
                    sent_meas_label = list(set.intersection(az_labels, meas_labels))
                    if len(sent_meas_label) == 2:
                        sent_meas_label = "MEASUREMENT"
                    elif len(sent_meas_label) == 1:
                        sent_meas_label = sent_meas_label[0]
                    else:
                        sent_meas_label = "O"
                    # Remove AZ labels that are not in structure / content tags defined
                    az_labels = list(
                        set.intersection(all_az_labels, az_labels)
                    )  # keep only valid labels

                    if len(az_labels) == 0:
                        continue

                    # Tokens
                    sent_tokens = list(doc.select_covered(TOKEN_TYPE, sent_annot))  # noqa: F405
                    sent_token_list = [0] * len(sent_tokens)

                    token2idx = {}
                    for i, token in enumerate(sent_tokens):
                        token2idx[token.begin] = i
                        sent_token_list[i] = doc.get_covered_text(token)  # token text

                    # Named Entity annotations
                    sent_offset: int = sent_annot.begin
                    ner_labels: list = []
                    ner_labels_duplicate_lookup: dict = (
                        dict()
                    )  # Used to detect duplicate Named Entity annotations
                    ne_annot2id: dict = {}
                    for ent_annot in doc.select_covered(ENTITY_TYPE, sent_annot):  # noqa: F405
                        if ent_annot.get_feature_value("implicitEntity") is None:
                            label = ent_annot.get_feature_value("value")
                            if label in ne_labels_set:  # filter by applicable labels
                                # retrieve token indices
                                ent_indices = get_token_indices_for_annot(
                                    ent_annot, sent_tokens, doc
                                )
                                if (
                                    None in ent_indices
                                ):  # happens if entity annotation is subtoken : choose covering token
                                    try:
                                        ent_indices = get_token_index_for_annot_if_subtoken(
                                            ent_annot, sent_tokens, doc
                                        )
                                    except StopIteration:
                                        pass
                                    except ValueError:
                                        pass
                                if None in ent_indices:
                                    continue

                                try:
                                    ne_annot2id[ent_annot] = ent_annot.id
                                    ne_dict: dict = {
                                        "text": doc.get_covered_text(ent_annot),
                                        "id": ent_annot.id,
                                        "value": label,
                                        "begin": ent_annot.begin - sent_offset,
                                        "end": ent_annot.end - sent_offset,
                                        "tokenIndices": ent_indices,
                                    }  # index of first + last token of the NE
                                    if (
                                        not tuple(
                                            [ne_dict["value"], ne_dict["begin"], ne_dict["end"]]
                                        )
                                        in ner_labels_duplicate_lookup.keys()
                                    ):
                                        ner_labels.append(ne_dict)
                                        ner_labels_duplicate_lookup[
                                            (
                                                tuple(
                                                    [
                                                        ne_dict["value"],
                                                        ne_dict["begin"],
                                                        ne_dict["end"],
                                                    ]
                                                )
                                            )
                                        ] = ent_annot.id
                                    else:
                                        ne_annot2id[ent_annot] = ner_labels_duplicate_lookup[
                                            (
                                                tuple(
                                                    [
                                                        ne_dict["value"],
                                                        ne_dict["begin"],
                                                        ne_dict["end"],
                                                    ]
                                                )
                                            )
                                        ]
                                except KeyError:
                                    pass

                    # Creating Nested Named Entity BIO Labels
                    B: str = "B-{0}"
                    I: str = "I-{0}"
                    L: str = "L-{0}"
                    O: str = "O"
                    U: str = "U-{0}"
                    ner_labels.sort(
                        key=lambda x: (x["tokenIndices"][0], -x["tokenIndices"][1])
                    )  # Work from left to right and prioritize longer strings
                    nested_bilou_labels: list = [O] * len(sent_tokens)
                    if len(ner_labels) > 0:
                        for i in range(len(ner_labels)):
                            begin_idx: int = ner_labels[i]["tokenIndices"][0]
                            end_idx: int = ner_labels[i]["tokenIndices"][1]

                            # Check whether there are already two NE annotation layers within this span
                            skip_current_entity: bool = False
                            for j in range(begin_idx, end_idx + 1):
                                if (
                                    nested_bilou_labels[j].count("+") == 2
                                ):  # Already 3 annotations connected via "+"
                                    skip_current_entity = True
                                    break

                            if skip_current_entity:
                                continue

                            tag: str = ner_labels[i]["value"]

                            # Case of Unit Length Tag
                            if begin_idx == end_idx:
                                if nested_bilou_labels[begin_idx] == O:
                                    nested_bilou_labels[begin_idx] = U.format(tag)
                                else:
                                    nested_bilou_labels[begin_idx] += "+" + U.format(tag)
                                continue

                            # Tags that span over more than one token
                            if nested_bilou_labels[begin_idx] == O:
                                nested_bilou_labels[begin_idx] = B.format(tag)
                            else:
                                nested_bilou_labels[begin_idx] += "+" + B.format(tag)

                            for j in range(begin_idx + 1, end_idx + 1):  # Append all inside tags
                                if j < end_idx:
                                    if nested_bilou_labels[j] == O:
                                        nested_bilou_labels[j] = I.format(tag)
                                    else:
                                        nested_bilou_labels[j] += "+" + I.format(tag)
                                else:
                                    if nested_bilou_labels[j] == O:
                                        nested_bilou_labels[j] = L.format(tag)
                                    else:
                                        nested_bilou_labels[j] += "+" + L.format(tag)

                    # positive relation instances
                    rel_labels: list = []
                    for rel_annot in doc.select_covered(RELATION_TYPE, sent_annot):  # noqa: F405
                        label: str = rel_annot.get_feature_value("RelationType")

                        gov_annot = rel_annot.get_feature_value("Governor", True)
                        dep_annot = rel_annot.get_feature_value("Dependent", True)

                        gov_label = gov_annot.get_feature_value("value")

                        if (
                            label in rel_label_set
                        ):  # only consider annotation if in selected set of relations

                            # --- Adding transitive links --- #
                            # conditionProperty + propertyValue --> conditionPropertyValue
                            # measuresProperty  + propertyValue --> measuresPropertyValue
                            # Note that this will also happen when the intermediate entity is implicit!
                            if gov_label == "MEASUREMENT" and label in {
                                "conditionProperty",
                                "measuresProperty",
                            }:
                                for rel_annot2 in doc.select_covered(
                                    RELATION_TYPE, sent_annot  # noqa: F405
                                ):
                                    label2: str = rel_annot2.get_feature_value("RelationType")
                                    gov_annot2 = rel_annot2.get_feature_value("Governor", True)
                                    dep_annot2 = rel_annot2.get_feature_value("Dependent", True)

                                    if label2 == "propertyValue" and gov_annot2 == dep_annot:
                                        if label == "conditionProperty":
                                            transitiveLabel = "conditionPropertyValue"
                                        elif label == "measuresProperty":
                                            transitiveLabel = "measuresPropertyValue"
                                        else:
                                            assert False

                                        try:
                                            rel_labels.append(
                                                {
                                                    "ne_id_gov": ne_annot2id[gov_annot],
                                                    "ne_id_dep": ne_annot2id[dep_annot2],
                                                    "label": transitiveLabel,
                                                }
                                            )
                                        except KeyError:
                                            continue
                            # --- End of adding transitive links --- #

                            if (
                                gov_annot.get_feature_value("value") not in ne_labels_set
                                or dep_annot.get_feature_value("value") not in ne_labels_set
                            ):
                                # only considering relations between "valid" NE types for now
                                continue
                            if gov_annot is dep_annot:
                                continue
                            if (
                                gov_annot.begin == dep_annot.begin
                                and gov_annot.end == dep_annot.end
                            ):
                                # same span, continue
                                continue

                            if gov_annot not in ne_annot2id:
                                # check if it's in a different sentence
                                if doc.get_covered_text(dep_annot) == "nanoparticle-type":
                                    continue
                                sent_list2 = list(
                                    doc.select_covering(SENT_TYPE, gov_annot)  # noqa: F405
                                )
                                try:
                                    sent_list2.remove(sent_annot)
                                except ValueError:
                                    pass
                                if len(sent_list2) == 0:
                                    continue
                                sent_annot2 = sent_list2[0]
                                if sent_annot2 is not sent_annot:
                                    # gov in different sentence, skipping cross-sentence links
                                    continue
                            if dep_annot not in ne_annot2id:
                                sent_list2 = list(
                                    doc.select_covering(SENT_TYPE, dep_annot)  # noqa: F405
                                )
                                try:
                                    sent_list2.remove(sent_annot)
                                except ValueError:
                                    pass
                                if len(sent_list2) == 0:
                                    continue
                                sent_annot2 = sent_list2[0]
                                if sent_annot2 != sent_annot:
                                    # dep in different sentence, skipping cross-sentence links
                                    continue

                            if gov_annot not in ne_annot2id:
                                if gov_annot.get_feature_value("valueType") == "implicit":
                                    # skip this case, implicit PROPERTY
                                    continue
                                assert False
                            if dep_annot not in ne_annot2id:
                                assert False
                            rel_labels.append(
                                {
                                    "ne_id_gov": ne_annot2id[gov_annot],
                                    "ne_id_dep": ne_annot2id[dep_annot],
                                    "label": label,
                                }
                            )

                    if split_info is None:
                        split_info = data_split[data_split["name"] == doc_name]["set"].values[0]
                        category_info = data_split[data_split["name"] == doc_name][
                            "category"
                        ].values[0]

                    # Iterator yields data sample sentence-wise
                    yield doc_name + "/" + str(sent_id), {
                        "doc_id": doc_name,
                        "sentence": sent_text,
                        "tokens": sent_token_list,
                        "beginOffset": sent_annot.begin,
                        "endOffset": sent_annot.end,
                        "AZ_labels": az_labels,
                        "Measurement_label": sent_meas_label,
                        "NER_labels": ner_labels,
                        "NER_labels_BILOU": nested_bilou_labels,
                        "relations": rel_labels,  # [(ne_index_in_list, ne_index_in_list, relation_name)]
                        "docFileName": doc_name,
                        "data_split": split_info,
                        "category": category_info,
                    }

                split_info = None
                category_info = None
        elif self.config.name == "NER_Dependencies":
            id: int = 0
            sent_id: int = 0
            sent_text: str = None
            for i, f in enumerate(files):
                split_info: str = (
                    f"train{i+1}" if "train" in f else ("dev" if "dev" in f else "test")
                )
                with open(join(dir, f), mode="r", encoding="utf-8") as cf:
                    conll_lines: list[str] = cf.read().splitlines()
                for line in conll_lines:
                    if line.startswith("#"):
                        sent_text = line.split("# text = ")[-1]
                        sent_id += 1
                        continue
                    elif line == "":
                        continue
                    t_id, t_text, deps = line.split("\t")
                    yield id, {
                        "ID": sent_id,
                        "sentence": sent_text,
                        "token_id": t_id,
                        "token_text": t_text,
                        "NE_Dependencies": deps,
                        "data_split": split_info,
                    }
                    id += 1
