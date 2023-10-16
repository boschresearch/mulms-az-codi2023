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
This module contains the all-task AZ model that learns the MuLMS-AZ dataset and all other AZ datasets
by jointly training one BERT embedding model and multiple linear output layers on top.
"""
from typing import Tuple

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig

from source.constants.constants import CPU


class AllTaskAZClassifier(nn.Module):
    """
    Pytorch model with four output heads for four AZ tasks.
    It contains four linear output heads on top of the jointly trained and used
    BERT-based embedding model where each of them is responsible for a different
    AZ datasets.
    """

    def __init__(
        self,
        model_path: str,
        num_labels_mulms: int,
        num_labels_art: int,
        num_labels_az_cl: int,
        num_labels_dri: int,
        dropout_rate=0.1,
        device=CPU,
    ) -> None:
        """
        Initializes the all-task AZ classifier.

        Args:
            model_path (str): Path or name to/of the BERT-based model.
            num_labels_mulms (int): Number of MuLMS-AZ labels.
            num_labels_art (int): Number of ART AZ labels.
            num_labels_az_cl (int): Number of AZ-CL AZ labels.
            num_labels_dri (int): Number of DRI AZ labels.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            device (str, optional): Backend device. Must be either "cpu" or "cuda". Defaults to "cpu".
        """
        assert (
            model_path is not None and type(model_path) == str
        ), "Model Path must be a valid string."
        super(AllTaskAZClassifier, self).__init__()
        config: PretrainedConfig = BertConfig.from_pretrained(model_path)
        cls_size: int = int(config.hidden_size)

        self._encoder: BertModel = BertModel.from_pretrained(model_path)
        self._dropout: nn.Dropout = nn.Dropout(p=dropout_rate)
        self._mulms_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_mulms
        )
        self._art_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_art
        )
        self._az_cl_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_az_cl
        )
        self._dri_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_dri
        )
        self._device = device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        dataset_flag: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes forward pass for multi-task training loop.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            dataset_flag (int): Determines which output head is responsible for the current batch. All other outputs are set to -100.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits, Sigmoid, CLS embedding
        """
        logits: torch.Tensor = None
        embeddings = self._encoder(input_ids, attention_mask, token_type_ids)
        cls_embedding: torch.Tensor = embeddings.last_hidden_state[:, 0]
        dp_output: torch.Tensor = self._dropout(cls_embedding)
        if dataset_flag == 0:
            logits = self._mulms_classifier(dp_output)
        elif dataset_flag == 1:
            logits = self._art_classifier(dp_output)
        elif dataset_flag == 2:
            logits = self._az_cl_classifier(dp_output)
        elif dataset_flag == 3:
            logits = self._dri_classifier(dp_output)

        return logits, torch.sigmoid(logits), cls_embedding
