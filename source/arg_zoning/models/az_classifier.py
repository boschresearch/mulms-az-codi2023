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
This modules contains the Pytorch AZ classifier that builds upon
contextualized BERT-based embeddings.
"""
from typing import Tuple

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig


class AZClassifier(nn.Module):
    """
    BERT-based classifier that uses a linear classification layer on top to
    predict AZ labels.
    """

    def __init__(self, model_path: str, num_labels=2, dropout_rate=0.1) -> None:
        """
        Initializes the AZ classifier.

        Args:
            model_path (str): Path or name to/of the BERT-based model.
            num_labels (int, optional): Number of AZ labels in the dataset. Defaults to 2.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        assert (
            model_path is not None and type(model_path) == str
        ), "Model Path must be a valid string."
        super(AZClassifier, self).__init__()
        self._encoder: BertModel = BertModel.from_pretrained(model_path)
        config: PretrainedConfig = BertConfig.from_pretrained(model_path)
        config.output_hidden_states = False
        self._cls_size: int = int(config.hidden_size)
        self._dropout: nn.Dropout = nn.Dropout(p=dropout_rate)
        self._linear_layer: nn.Linear = nn.Linear(self._cls_size, num_labels)

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
        model_outputs = self._encoder(input_ids, attention_mask, token_type_ids)
        encoded_cls: torch.Tensor = model_outputs.last_hidden_state[:, 0]
        encoded_cls_dp: torch.Tensor = self._dropout(encoded_cls)
        logits: torch.Tensor = self._linear_layer(encoded_cls_dp)
        sigmoid_scores: torch.Tensor = torch.sigmoid(logits)
        return logits, sigmoid_scores, encoded_cls
