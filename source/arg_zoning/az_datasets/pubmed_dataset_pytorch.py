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
This module contains the Pytorch dataset for the PUBMED AZ corpus.
"""
import random

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from source.arg_zoning.az_datasets.pubmed_dataset import load_pubmed_dataset


class PubmedAZDataset(Dataset):
    """
    Dataset which stores all tensors for the Pubmed AZ dataset.
    """

    def __init__(
        self,
        tokenizer_model_name: str,
        split: str,
        pubmed_dataset=None,
        subsample_rate=None,
        seed=12344321,
        pad_to_max_length: bool = False,
        max_seq_length: int = 512,
    ) -> None:
        """
        Initializes Pubmed AZ Pytorch dataset.

        Args:
            tokenizer_model_name (str): Name or path of BERT-based tokenizer.
            split (str): Corresponding split.
            pubmed_dataset (PubmedDataset, optional): Pubmed AZ dataset object. Will be loaded if None. Defaults to None.
            subsample_rate (float, optional): Subsample rate; if set: must be between 0 and 1
            seed (int, optional): Seed for creating random subset
            pad_to_max_length (bool, optional): Whether to pad tensors to maximum model input length. Defaults to False.
            max_seq_length (int, optional): The maximum sequence input length of the BERT model. Defaults to 512.
        """
        assert (
            tokenizer_model_name is not None and split is not None
        ), "Tokenizer name and split must not be None."
        assert split in ["train", "validation", "test"]
        super().__init__()
        if pubmed_dataset is None:
            pubmed_dataset = load_pubmed_dataset()
        self.name: str = "pubmed"
        self._raw_dataset = pubmed_dataset
        self._data: dict = {}
        self._num_labels: int = len(pubmed_dataset.idx2cat)
        self._split_indices: tuple = None
        self._split = split
        self._remove_from_gpu: list = []
        if self._split == "train":
            self._split_indices = (0, self._raw_dataset.data_split[0])
        elif self._split == "validation":
            self._split_indices = (
                self._raw_dataset.data_split[0],
                self._raw_dataset.data_split[1],
            )
        else:
            self._split_indices = (
                self._raw_dataset.data_split[1],
                len(self._raw_dataset.documents),
            )
        self._load_pubmed_az_dataset(
            tokenizer_model_name=tokenizer_model_name, pad_to_max_length=pad_to_max_length
        )
        self._data_indices: list = list(range(len(self._data["az_label"])))
        if subsample_rate is not None:
            random.seed(seed)
            random.shuffle(self._data_indices)
            self._data_indices = self._data_indices[
                : int(subsample_rate * len(self._data_indices))
            ]

    def __len__(self) -> int:
        """
        Returns the (subsampled) length

        Returns:
            int: Length of the dataset
        """
        return len(self._data_indices)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a sample of the dataset.

        Args:
            index (int): (Random) Index to get a sample.

        Returns:
            dict: Sample containing ID, BERT tensors and label.
        """
        idx: int = self._data_indices[index]
        return {
            "doc_id": self._data["doc_id"][idx],
            "sent_id": self._data["sent_id"][idx],
            "tensor": {
                "input_ids": self._data["tensors"]["input_ids"][idx],
                "attention_mask": self._data["tensors"]["attention_mask"][idx],
                "token_type_ids": self._data["tensors"]["token_type_ids"][idx],
            },
            "label": self._data["tensorEncodedLabels"][idx],
        }

    def _load_pubmed_az_dataset(
        self, tokenizer_model_name: str, pad_to_max_length: bool = False, max_seq_length: int = 512
    ) -> None:
        """
        Encodes Pubmed AZ dataset as tensors.

        Args:
            tokenizer_model_name (str): Name or path of BERT-based tokenizer.
            pad_to_max_length (bool, optional): Whether to pad tensors to maximum model input length. Defaults to False.
            max_seq_length (int, optional): The maximum sequence input length of the BERT model. Defaults to 512.
        """
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_model_name)
        self._data["tokens"] = []
        self._data["doc_id"] = []
        self._data["sent_id"] = []
        self._data["az_label"] = []
        for i, doc in enumerate(
            self._raw_dataset.documents[self._split_indices[0] : self._split_indices[1]]
        ):
            for j, sent in enumerate(doc):
                self._data["tokens"].append([self._raw_dataset.idx2word[t] for t in sent[0]])
                self._data["doc_id"].append(i)
                self._data["sent_id"].append(j)
                self._data["az_label"].append(self._raw_dataset.cat2idx[sent[2]])
        self._data["tensors"] = tokenizer.batch_encode_plus(
            self._data["tokens"],
            is_split_into_words=True,
            return_tensors="pt",
            padding=("max_length" if pad_to_max_length else True),
            truncation=True,
            max_length=max_seq_length,
        )
        self._data["oneHotAZLabels"] = [None] * len(self._data["az_label"])
        for i, az in enumerate(self._data["az_label"]):
            self._data["oneHotAZLabels"][i] = [0.0] * len(self._raw_dataset.cat2idx)
            self._data["oneHotAZLabels"][i][az] = 1.0
        self._data["tensorEncodedLabels"] = torch.tensor(self._data["oneHotAZLabels"])
        del tokenizer
