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
This module contains the binary AZ classifier used for the AZ transfer experiment.
"""
from typing import Tuple

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig


class BinaryAZClassifier(nn.Module):
    """
    BERT-based binary AZ classifier that uses a single output node to decide
    whether a sample belongs to a target AZ class or not.
    """

    def __init__(self, model_path: str, dropout_rate=0.1) -> None:
        """
        Initializes the binary AZ classifier.

        Args:
            model_path (str): Path or name to/of the BERT-based model.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        assert (
            model_path is not None and type(model_path) == str
        ), "Model Path must be a valid string."
        super(BinaryAZClassifier, self).__init__()
        self.encoder: BertModel = BertModel.from_pretrained(model_path)
        config: PretrainedConfig = BertConfig.from_pretrained(model_path)
        config.output_hidden_states = False
        self.cls_size: int = int(config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)
        self.linear_layer: nn.Linear = nn.Linear(self.cls_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the classifier

        Args:
            input_ids (torch.Tensor): Input IDs as returned by a tokenizer
            attention_mask (torch.Tensor): Attention mask values
            token_type_ids (torch.Tensor): Token Type IDs

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits, Sigmoid scores, CLS embedding
        """
        model_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        encoded_cls: torch.Tensor = model_outputs.last_hidden_state[:, 0]
        encoded_cls_dp: torch.Tensor = self.dropout(encoded_cls)
        logits: torch.Tensor = self.linear_layer(encoded_cls_dp)
        sigmoid_scores: torch.Tensor = torch.sigmoid(logits)
        return logits, sigmoid_scores, encoded_cls
