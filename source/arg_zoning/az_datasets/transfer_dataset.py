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
This module contains the Pytorch dataset for the AZ label transfer experiment.
"""
from ast import literal_eval

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from source.arg_zoning.az_datasets.az_dataset import AZ_Dataset
from source.arg_zoning.az_datasets.pubmed_dataset import (
    PubmedDataset,
    load_pubmed_dataset,
)
from source.arg_zoning.az_datasets.pubmed_dataset_pytorch import PubmedAZDataset
from source.constants.constants import (
    ADDITIONAL_AZ_LABELS,
    AZ_DATASET_PATHS,
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    art_id2label,
    az_cl_id2label,
    dri_id2label,
    pubmed_id2label,
)


class AZ_TransferDataset(Dataset):
    """
    Pytorch dataset for the AZ label transfer experiment.
    """

    def __init__(
        self,
        tokenizer_path: str,
        mulms_az_target_label: str,
        mapping_table: dict,
        subsample_rate: float = 1.0,
        train_on_mulms: bool = False,
    ) -> None:
        """
        Initializes the AZ transfer dataset.

        Args:
            tokenizer_path (str): Path to the BERT-based model for the tokenizer.
            mulms_az_target_label (str): Target AZ label from the MuLMS-AZ corpus.
            mapping_table (dict): Mapping table that maps the different labels of the different corpora onto each other.
            subsample_rate (float, optional): Subsample rate for training. Defaults to 1.0.
            train_on_mulms (bool, optional): Whether to use MuLMS-AZ for training. Defaults to False.
        """
        super().__init__()
        self._encoded_batch: dict = {}
        self._binary_labels: list = []
        if train_on_mulms:
            train_dataset: Dataset = load_dataset(
                MULMS_DATASET_READER_PATH.__str__(),
                data_dir=MULMS_PATH.__str__(),
                data_files=MULMS_FILES,
                name="MuLMS_Corpus",
                split="train",
            )
            tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_path)
            tokens: list[list[str]] = train_dataset["tokens"]
            self._encoded_batch = tokenizer.batch_encode_plus(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            az_labels: list[str] = [literal_eval(label) for label in train_dataset["AZ_labels"]]
            target_labels: set = set(mapping_table[mulms_az_target_label]["targets"])
            for labels in az_labels:
                if len(target_labels.intersection(set(labels))) > 0:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)
        else:
            pubmed_dataset: PubmedDataset = load_pubmed_dataset()
            pubmed_dataset_train: PubmedAZDataset = PubmedAZDataset(
                tokenizer_path,
                split="train",
                pubmed_dataset=pubmed_dataset,
                pad_to_max_length=True,
            )
            pubmed_dataset_dev: PubmedAZDataset = PubmedAZDataset(
                tokenizer_path,
                split="validation",
                pubmed_dataset=pubmed_dataset,
                pad_to_max_length=True,
            )
            pubmed_dataset_test: PubmedAZDataset = PubmedAZDataset(
                tokenizer_path, split="test", pubmed_dataset=pubmed_dataset, pad_to_max_length=True
            )
            art_dataset: AZ_Dataset = AZ_Dataset(
                "ART",
                AZ_DATASET_PATHS["ART"],
                tokenizer_path,
                ADDITIONAL_AZ_LABELS["ART"]["label2id"],
                ADDITIONAL_AZ_LABELS["ART"]["id2label"],
                subsample_rate=subsample_rate,
                pad_to_max_length=True,
            )
            az_cl_dataset: AZ_Dataset = AZ_Dataset(
                "AZ-CL",
                AZ_DATASET_PATHS["AZ-CL"],
                tokenizer_path,
                ADDITIONAL_AZ_LABELS["AZ-CL"]["label2id"],
                ADDITIONAL_AZ_LABELS["AZ-CL"]["id2label"],
                subsample_rate=subsample_rate,
                pad_to_max_length=True,
            )
            dri_dataset: AZ_Dataset = AZ_Dataset(
                "DRI",
                AZ_DATASET_PATHS["DRI"],
                tokenizer_path,
                ADDITIONAL_AZ_LABELS["DRI"]["label2id"],
                ADDITIONAL_AZ_LABELS["DRI"]["id2label"],
                subsample_rate=subsample_rate,
                pad_to_max_length=True,
            )
            self._encoded_batch: dict = {
                "input_ids": art_dataset._encoded_batch["input_ids"],
                "attention_mask": art_dataset._encoded_batch["attention_mask"],
                "token_type_ids": art_dataset._encoded_batch["token_type_ids"],
            }
            for id in art_dataset._labels:
                if art_id2label[id] in mapping_table[mulms_az_target_label]["ART"]:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)

            if "AZ-CL" in mapping_table:  # Not always the case!
                self._encoded_batch["input_ids"] = torch.cat(
                    (self._encoded_batch["input_ids"], az_cl_dataset._encoded_batch["input_ids"])
                )
                self._encoded_batch["attention_mask"] = torch.cat(
                    (
                        self._encoded_batch["attention_mask"],
                        az_cl_dataset._encoded_batch["attention_mask"],
                    )
                )
                self._encoded_batch["token_type_ids"] = torch.cat(
                    (
                        self._encoded_batch["token_type_ids"],
                        az_cl_dataset._encoded_batch["token_type_ids"],
                    )
                )
                for id in az_cl_dataset._labels:
                    if az_cl_id2label[id] in mapping_table[mulms_az_target_label]["AZ-CL"]:
                        self._binary_labels.append(1)
                    else:
                        self._binary_labels.append(0)

            self._encoded_batch["input_ids"] = torch.cat(
                (self._encoded_batch["input_ids"], dri_dataset._encoded_batch["input_ids"])
            )
            self._encoded_batch["attention_mask"] = torch.cat(
                (
                    self._encoded_batch["attention_mask"],
                    dri_dataset._encoded_batch["attention_mask"],
                )
            )
            self._encoded_batch["token_type_ids"] = torch.cat(
                (
                    self._encoded_batch["token_type_ids"],
                    dri_dataset._encoded_batch["token_type_ids"],
                )
            )
            for id in dri_dataset._labels:
                if dri_id2label[id] in mapping_table[mulms_az_target_label]["DRI"]:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)

            self._encoded_batch["input_ids"] = torch.cat(
                (
                    self._encoded_batch["input_ids"],
                    pubmed_dataset_train._data["tensors"].data["input_ids"],
                    pubmed_dataset_dev._data["tensors"].data["input_ids"],
                    pubmed_dataset_test._data["tensors"].data["input_ids"],
                )
            )
            self._encoded_batch["attention_mask"] = torch.cat(
                (
                    self._encoded_batch["attention_mask"],
                    pubmed_dataset_train._data["tensors"].data["attention_mask"],
                    pubmed_dataset_dev._data["tensors"].data["attention_mask"],
                    pubmed_dataset_test._data["tensors"].data["attention_mask"],
                )
            )
            self._encoded_batch["token_type_ids"] = torch.cat(
                (
                    self._encoded_batch["token_type_ids"],
                    pubmed_dataset_train._data["tensors"].data["token_type_ids"],
                    pubmed_dataset_dev._data["tensors"].data["token_type_ids"],
                    pubmed_dataset_test._data["tensors"].data["token_type_ids"],
                )
            )
            for id in pubmed_dataset_train._data["az_label"]:
                if pubmed_id2label[id] in mapping_table[mulms_az_target_label]["PUBMED"]:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)

            for id in pubmed_dataset_dev._data["az_label"]:
                if pubmed_id2label[id] in mapping_table[mulms_az_target_label]["PUBMED"]:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)

            for id in pubmed_dataset_test._data["az_label"]:
                if pubmed_id2label[id] in mapping_table[mulms_az_target_label]["PUBMED"]:
                    self._binary_labels.append(1)
                else:
                    self._binary_labels.append(0)

        self._tensor_encoded_labels: torch.Tensor = torch.tensor(self._binary_labels).float()
        self._ids: list[int] = list(range(1, len(self._binary_labels) + 1))

    def __len__(self) -> int:
        """
        Returns number of AZ samples in the dataset.

        Returns:
            int: Number of AZ samples.
        """
        return len(self._binary_labels)

    def __getitem__(self, index: int):
        """
        Returns a sample of the dataset.

        Args:
            index (int): (Random) Index to get a sample.

        Returns:
            dict: Sample containing ID, BERT tensors and label.
        """
        return {
            "id": self._ids[index],
            "tensor": {
                "input_ids": self._encoded_batch["input_ids"][index],
                "attention_mask": self._encoded_batch["attention_mask"][index],
                "token_type_ids": self._encoded_batch["token_type_ids"][index],
            },
            "label": self._tensor_encoded_labels[index],
        }
