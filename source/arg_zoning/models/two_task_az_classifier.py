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
This module contains the two-task AZ model that learns the MuLMS-AZ dataset and one further AZ dataset.
"""
from typing import Tuple

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig

from source.constants.constants import CPU


class TwoTaskAZClassifier(nn.Module):
    """
    Pytorch model with two output heads for two AZ tasks.
    It contains two linear output heads on top of the jointly trained and used
    BERT-based embedding model where one of them is responsible for the MuLMS-AZ
    task.
    """

    def __init__(
        self,
        model_path: str,
        num_labels_mulms: int,
        num_labels_second_task: int,
        dropout_rate=0.1,
        device=CPU,
    ) -> None:
        """
        Initializes the two-task AZ model.

        Args:
            model_path (str): Path or name to/of the BERT-based model.
            num_labels_mulms (int): Number of MuLMS-AZ labels.
            num_labels_second_task (int): Number of AZ labels of the second dataset.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            device (str, optional): Backend device. Must be either "cpu" or "cuda". Defaults to "cpu".
        """
        assert (
            model_path is not None and type(model_path) == str
        ), "Model Path must be a valid string."
        super(TwoTaskAZClassifier, self).__init__()
        config: PretrainedConfig = BertConfig.from_pretrained(model_path)
        cls_size: int = int(config.hidden_size)

        self._encoder: BertModel = BertModel.from_pretrained(model_path)
        self._dropout: nn.Dropout = nn.Dropout(p=dropout_rate)
        self._mulms_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_mulms
        )
        self._other_classifier: nn.Linear = nn.Linear(
            in_features=cls_size, out_features=num_labels_second_task
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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits, Sigmoid scores, CLS embedding
        """
        logits: torch.Tensor = None
        embeddings = self._encoder(input_ids, attention_mask, token_type_ids)
        cls_embedding: torch.Tensor = embeddings.last_hidden_state[:, 0]
        dp_output: torch.Tensor = self._dropout(cls_embedding)
        if dataset_flag == 0:
            logits = self._mulms_classifier(dp_output)
        else:
            logits = self._other_classifier(dp_output)

        return logits, torch.sigmoid(logits), cls_embedding
