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
This module contains the Pytorch dataset class for the MuLMS-AZ corpus.
"""
import random
import re
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset as HF_Dataset
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from source.arg_zoning.az_datasets.pubmed_dataset_pytorch import PubmedAZDataset
from source.constants.constants import (
    az_filtered_label2id,
    pubmed_label2id,
    pubmed_to_az_mapping,
)


class MuLMS_AZDataset(Dataset):
    """
    Pytorch dataset class for the MuLMS-AZ corpus.
    """

    def __init__(
        self,
        dataset: HF_Dataset,
        split: str,
        tokenizer_model_name: str,
        hyperparameters: dict,
        hierarchy: dict,
        labels_to_remove: list,
        tune_id: int = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            dataset (HF_Dataset): The HuggingFace dataset that contains the MuLMS-AZ corpus.
            split (str): Desired split. Must be one of [train, tune, validation, test].
            tokenizer_model_name (str): Name of the BERT-based model for the tokenizer.
            hyperparameters (dict): Training hyperparameters.
            hierarchy (dict): AZ label hierarchy for super-/sublabels.
            labels_to_remove (list): AZ labels that should not appear in the dataset.
            tune_id (int, optional): Used to select the correct fold if split = tune. Defaults to None.
        """
        assert (
            dataset is not None and tokenizer_model_name is not None
        ), "Dataset and tokenizer must not be None."
        assert split in [
            "train",
            "tune",
            "validation",
            "test",
        ], "Invalid split provided. Must be either train, tune, validation or test."
        if split == "tune":
            assert tune_id is not None and tune_id in [
                1,
                2,
                3,
                4,
                5,
            ], "Invalid tune set. Must be either 1, 2, 3, 4 or 5."
        super().__init__()
        self.name: str = "MuLMS"
        self._split: str = split
        self._tune_id: int = tune_id
        self._df: pd.DataFrame = pd.DataFrame(
            {
                "doc_id": dataset["doc_id"],
                "sentence": dataset["sentence"],
                "tokens": dataset["tokens"],
                "beginOffset": dataset["beginOffset"],
                "endOffset": dataset["endOffset"],
                "AZ_labels": dataset["AZ_labels"],
                "NER_labels": dataset["NER_labels"],
                "docFileName": dataset["docFileName"],
                "data_split": dataset["data_split"],
                "category_info": dataset["category"],
            }
        )

        self._data: dict = {}
        all_samples: list = []
        if split == "tune":
            self._tune_id = tune_id
            all_samples, _ = self._load_ms_samples(
                split=f"train{self._tune_id}",
                dataframe=self._df,
                hierarchy=hierarchy,
                labels_to_remove=labels_to_remove,
            )
        elif split == "train":
            for fold in range(1, 6):
                if fold != self._tune_id:
                    samples, _ = self._load_ms_samples(
                        split=f"train{fold}",
                        dataframe=self._df,
                        hierarchy=hierarchy,
                        labels_to_remove=labels_to_remove,
                    )
                    all_samples += samples
        elif split == "validation":
            all_samples, _ = self._load_ms_samples(
                split="dev",
                dataframe=self._df,
                hierarchy=hierarchy,
                labels_to_remove=labels_to_remove,
            )
        else:
            all_samples, _ = self._load_ms_samples(
                split=self._split,
                dataframe=self._df,
                hierarchy=hierarchy,
                labels_to_remove=labels_to_remove,
            )
        self._prepare_tensors(
            all_samples, az_filtered_label2id, tokenizer_model_name, hyperparameters
        )

    def __len__(self) -> int:
        """
        Returns number of AZ samples in the dataset.

        Returns:
            int: Number of AZ samples.
        """
        return len(self._data["ID"])

    def __getitem__(self, index: int) -> dict:
        """
        Returns a sample of the dataset.

        Args:
            index (int): (Random) Index to get a sample.

        Returns:
            dict: Sample containing ID, BERT tensors and label and flag whether it has been augmented from PUBMED.
        """
        return {
            "id": self._data["ID"][index],
            "tensor": {
                "input_ids": self._data["tensors"]["input_ids"][index],
                "attention_mask": self._data["tensors"]["attention_mask"][index],
                "token_type_ids": self._data["tensors"]["token_type_ids"][index],
            },
            "label": self._data["tensorEncodedLabels"][index],
            "isAugmented": self._data["isAugmented"][index],
        }

    def _load_ms_samples(
        self, split: str, dataframe: pd.DataFrame, hierarchy=None, labels_to_remove=None
    ) -> Tuple[list, set]:
        """
        Reads samples from dataframe row corresponding to the current split

        Args:
            split (str): Corresponding split
            dataframe (pd.DataFrame): Dataframe containing all data
            hierarchy (dict, optional): Label hierarchy
            labels_to_remove (list, optional): Labels to be removed. Defaults to None.

        Returns:
            Tuple[list, set]: Samples and AZ labels
        """
        samples: list = []
        labels: list = []
        for row in dataframe.iterrows():
            if row[1]["data_split"] == split:
                sentence = row[1]["sentence"]
                label_list = []
                for label in row[1]["AZ_labels"].split(","):
                    label_cleared = re.sub("['\]\[]", "", label).strip()  # noqa: W605
                    if label_cleared not in label_list:
                        label_list.append(label_cleared)
                for superlabel in hierarchy:
                    if label_cleared in hierarchy[superlabel] and superlabel not in label_list:
                        label_list.append(superlabel)
                for label in label_list:
                    if label in labels_to_remove:
                        label_list.remove(label)
                    else:
                        labels.append(label)
                if len(label_list) == 0:
                    print(f"No labels for sentence '{sentence}'")
                if None in label_list or "None" in label_list:
                    print(f"Found None label {label_list} for sentence {sentence}")
                samples.append((sentence, label_list))
        return samples, set(labels)

    def _prepare_tensors(
        self, samples: list, label2id: dict, tokenizer_model_name: str, hyperparameters: dict
    ) -> None:
        """
        Prepares all BERT tensors.

        Args:
            samples (list): AZ samples.
            label2id (dict): Lookup to map label to ID.
            tokenizer_model_name (str): Name of the BERT-based model for the tokenizer.
            hyperparameters (dict): Training hyperparameters.
        """
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_model_name)
        label_id_array: list = []
        samples = np.array(samples)
        all_sents: list = list(samples[:, 0])
        all_az_labels: list = list(samples[:, 1])
        encoded_sents: dict = tokenizer.batch_encode_plus(
            all_sents,
            add_special_tokens=True,
            max_length=hyperparameters["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for labels in all_az_labels:
            # convert labels to IDs:
            label_ids: list = []
            for label in labels:
                if label in label2id:
                    id: int = label2id[label]
                label_ids.append(id)
            label_ids: torch.Tensor = torch.tensor(label_ids, dtype=torch.long)
            label_onehot: torch.Tensor = F.one_hot(label_ids, num_classes=len(label2id))
            labels_onehot_acc: torch.Tensor = torch.zeros(len(label2id))
            for label_tensor in label_onehot:
                labels_onehot_acc.add_(label_tensor)
            label_id_array.append(labels_onehot_acc)
        label_id_array: torch.Tensor = torch.stack(label_id_array)

        self._data["tensors"] = encoded_sents
        self._data["tensorEncodedLabels"] = label_id_array
        self._data["ID"] = list(range(1, len(all_sents) + 1))
        self._data["isAugmented"] = [0] * len(
            self._data["ID"]
        )  # <- Indicates whether a sample has been additionally added from another dataset
        del tokenizer

    def augment_data_from_pubmed(
        self,
        pubmed_dataset: PubmedAZDataset,
        percentage_to_add=0.3,
        seed=23081861,
        pad_token_id=0,
        mask_non_pubmed_labels=False,
    ) -> None:
        """
        Adds data samples from the Pubmed dataset to the MLMS AZ dataset.

        Args:
            pubmed_dataset (PubmedDataset): Dataloader for Pubmed dataset
            percentage_to_add (float, optional): Percentage of samples to add from Pubmed RELATIVE to the size OF THIS dataset. Defaults to 0.3.
            seed (int, optional): Random seed for selecting Pubmed samples. Defaults to 23081861.
            pad_token_id (int, optional): Padding ID. Defaults to 0.
            mask_non_pubmed_labels (bool, optional): Whether to mask label positions with negative value if there is no corresponding mapping to Pubmed labels. Defaults to False.
        """
        samples_to_add: int = int(percentage_to_add * len(self))
        samples_added: int = 0
        pubmed_tensors: list[dict] = list(pubmed_dataset)

        augmented_input_tensors: list[torch.Tensor] = [None] * samples_to_add
        augmented_attention_mask_tensors: list[torch.Tensor] = [None] * samples_to_add
        augmented_token_type_tensors: list[torch.Tensor] = [None] * samples_to_add
        augmented_labels: list = [None] * samples_to_add

        random.seed(seed)
        idx: int = 0
        add_sample: bool = False

        mapping_indices: dict[int, int] = dict(
            [
                (pubmed_label2id[k], az_filtered_label2id[v])
                for k, v in pubmed_to_az_mapping.items()
            ]
        )
        while samples_added < samples_to_add:
            random_idx: int = random.randint(0, len(pubmed_tensors) - 1)
            one_hot_label: list[float] = [0.0] * len(self._data["tensorEncodedLabels"][0])
            for i, v in enumerate(pubmed_tensors[random_idx]["label"][0]):
                if v != 0 and i in mapping_indices.keys():
                    one_hot_label[mapping_indices[i]] = 1.0
                    add_sample = True

            if not add_sample:
                continue

            augmented_input_tensors[idx] = pubmed_tensors[random_idx]["tensor"]["input_ids"]
            augmented_attention_mask_tensors[idx] = pubmed_tensors[random_idx]["tensor"][
                "attention_mask"
            ]
            augmented_token_type_tensors[idx] = pubmed_tensors[random_idx]["tensor"][
                "token_type_ids"
            ]
            augmented_labels[idx] = one_hot_label
            idx += 1
            samples_added += 1
            add_sample = False

        padding_size: int = (
            pubmed_tensors[0]["tensor"]["input_ids"][0].shape[0]
            - self._data["tensors"]["input_ids"][0].shape[0]
        )
        self._data["tensors"]["input_ids"] = F.pad(
            self._data["tensors"]["input_ids"], (0, padding_size), value=pad_token_id
        )
        self._data["tensors"]["attention_mask"] = F.pad(
            self._data["tensors"]["attention_mask"], (0, padding_size), value=pad_token_id
        )
        self._data["tensors"]["token_type_ids"] = F.pad(
            self._data["tensors"]["token_type_ids"], (0, padding_size), value=pad_token_id
        )
        self._data["tensors"]["input_ids"] = torch.cat(
            (self._data["tensors"]["input_ids"], *augmented_input_tensors)
        )
        self._data["tensors"]["attention_mask"] = torch.cat(
            (self._data["tensors"]["attention_mask"], *augmented_attention_mask_tensors)
        )
        self._data["tensors"]["token_type_ids"] = torch.cat(
            (self._data["tensors"]["token_type_ids"], *augmented_token_type_tensors)
        )
        self._data["ID"].extend(
            list(range(self._data["ID"][-1] + 1, self._data["ID"][-1] + 1 + samples_added))
        )
        self._data["isAugmented"].extend([1] * samples_added)

        tensor_encoded_augmented_labels = torch.tensor(augmented_labels)
        if mask_non_pubmed_labels:
            mask: torch.Tensor = torch.tensor(
                [
                    i
                    for i in range(len(self._data["tensorEncodedLabels"][0]))
                    if i not in mapping_indices.values()
                ]
            )
            tensor_encoded_augmented_labels[:, mask] = -1

        self._data["tensorEncodedLabels"] = torch.cat(
            (self._data["tensorEncodedLabels"], tensor_encoded_augmented_labels)
        )
