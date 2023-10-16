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
This module contains the Pytorch dataset class for the all
additional AZ datasets (ART, AZ-CL, DRI).
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class AZ_Dataset(Dataset):
    """
    Pytorch dataset class for all additional AZ corpora (ART, AZ-CL, DRI) that returns batches
    of the dataset for training and evaluation.
    """

    def __init__(
        self,
        name: str,
        path: Path,
        tokenizer_model_name: str,
        label2id: dict,
        id2label: dict,
        subsample_rate: float = 1.0,
        pad_to_max_length: bool = False,
        max_seq_length: int = 512,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            name (str): Name of the dataset.
            path (Path): Path to the dataset files.
            tokenizer_model_name (str): Name of the BERT-based model for the tokenizer.
            label2id (dict): Lookup to map label to ID.
            id2label (dict): Lookup to map ID to label.
            subsample_rate (float, optional): Subsample rate between 0.0 and 1.0. Defaults to 1.0.
            pad_to_max_length (bool, optional): Whether to pad tensors to maximum model input length. Defaults to False.
            max_seq_length (int, optional): The maximum sequence input length of the BERT model. Defaults to 512.
        """
        super().__init__()
        self.name: str = name
        self._df: pd.DataFrame = None
        self._label2id: dict = label2id
        self._id2label: dict = id2label
        self._sentences: list[str] = None
        self._labels: list[int] = None
        self._one_hot_label_tensor: torch.Tensor = None
        self._encoded_batch = None

        self._load_dataset(tokenizer_model_name, path, pad_to_max_length, max_seq_length)

        self._ids: list[int] = list(range(1, len(self._labels) + 1))
        if subsample_rate != 1.0:
            # Only take the first m samples of this dataset
            count: int = int(len(self._ids) * subsample_rate)
            self._ids = self._ids[:count]
            self._sentences = self._sentences[:count]
            self._labels = self._labels[:count]
            self._one_hot_label_tensor = self._one_hot_label_tensor[:count]
            self._encoded_batch["input_ids"] = self._encoded_batch["input_ids"][:count]
            self._encoded_batch["attention_mask"] = self._encoded_batch["attention_mask"][:count]
            self._encoded_batch["token_type_ids"] = self._encoded_batch["token_type_ids"][:count]

        self._one_hot_label_tensor = self._one_hot_label_tensor.float()

    def __len__(self) -> int:
        """
        Returns the (subsampled) length

        Returns:
            int: Length of the dataset
        """
        # Reflects the subsampled length
        return len(self._ids)

    def __getitem__(self, index: int) -> dict:
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
            "label": self._one_hot_label_tensor[index],
        }

    def _load_dataset(
        self, tokenizer_model_name: str, path: Path, pad_to_max_length: bool, max_seq_length: int
    ) -> None:
        """
        Loads the AZ dataset by reading all files. (automatically called by constructor)

        Args:
            tokenizer_model_name (str): Name of the BERT-based model for the tokenizer
            path (Path): Path to the dataset files
            pad_to_max_length (bool): Whether to pad tensors to maximum model input length
            max_seq_length (int): The maximum sequence input length of the BERT model
        """
        self._df = pd.read_csv(path, delimiter="\t", names=["Label", "Sentence"])
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_model_name)
        self._sentences = self._df["Sentence"].tolist()
        self._labels = [self._label2id[label] for label in self._df["Label"].tolist()]
        self._one_hot_label_tensor = F.one_hot(torch.tensor(self._labels))
        self._encoded_batch = tokenizer.batch_encode_plus(
            self._sentences,
            return_tensors="pt",
            padding=("max_length" if pad_to_max_length else True),
            truncation=True,
            max_length=max_seq_length,
        )
        del tokenizer
