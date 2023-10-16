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
This module contains the loss function for the AZ augmentation experiment.
"""
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss


class BCEWithLogitsLossWithIgnore(_Loss):
    """Custom BCEWithLogitsLoss that ignores indices where target tensor is negative.
    Useful when working with padding.

    Additionally, makes BCE loss work with integer targets.
    """

    def __init__(self, reduction="mean"):
        """
        Initializes the BCE loss that ignores values whose indices are negative.

        Args:
            reduction (str, optional): Type of reduction. Defaults to "mean".
        """
        super(BCEWithLogitsLossWithIgnore, self).__init__()
        self.bce_with_logits_loss = BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss between target and loss.

        Args:
            input (torch.Tensor): Predicted AZ labels.
            target (torch.Tensor): True AZ labels.

        Returns:
            torch.Tensor: BCE Loss.
        """
        assert input.shape == target.shape

        input_non_ignored = input[target >= 0]
        target_non_ignored = target[target >= 0]

        assert input_non_ignored.shape == target_non_ignored.shape

        if target_non_ignored.dtype != torch.float32:
            target_non_ignored = target_non_ignored.float()

        return self.bce_with_logits_loss(input_non_ignored, target_non_ignored)


class AugmentedLoss(_Loss):
    """
    Specialized augmented loss function that can separate between augmented and non-augmented samples
    in the AZ augmentation experiment.

    """

    def __init__(
        self,
        reduction: str = "mean",
        distinguish_augmented_samples: bool = True,
        loss_weights: list[float] = [1.0, 1.0],
    ):
        """
        Initializes the augmented loss function.

        Args:
            reduction (str, optional): How to calculate the loss. Defaults to "mean".
            distinguish_augmented_samples (bool, optional): Whether to distinguish augmented samples and separately calculate two loss terms. Defaults to True.
            loss_weights (list[float], optional): If augmented samples are distinguished, both loss terms are weighted according to these values. Defaults to [1.0, 1.0].
        """
        super(AugmentedLoss, self).__init__()
        self._bce_with_logits_loss: BCEWithLogitsLossWithIgnore = BCEWithLogitsLossWithIgnore(
            reduction=reduction
        )
        self._reduction = reduction
        self._distinguish_augmented_samples = distinguish_augmented_samples
        self._loss_weights = loss_weights

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, augmentation_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the loss between input tensor and target tensor when using data augmentation.
        Additionally, if desired, the loss is distinguished between non-augmented samples
        and augmented samples and weighted accordingly given the weights.

        Args:
            input (torch.Tensor): Predicted AZ labels.
            target (torch.Tensor): True AZ labels.
            augmentation_mask (torch.Tensor): Booelan values that indicate wheter samples have been augmented or not.

        Returns:
            torch.Tensor: Calculated loss
        """
        if not self._distinguish_augmented_samples:
            loss: torch.Tensor = self._bce_with_logits_loss(input, target)
            return loss

        else:
            non_augmented_samples: torch.Tensor = input[augmentation_mask != 1]
            non_augmented_targets: torch.Tensor = target[augmentation_mask != 1]
            non_augmented_loss: torch.Tensor = self._bce_with_logits_loss(
                non_augmented_samples, non_augmented_targets
            )

            augmented_samples: torch.Tensor = input[augmentation_mask == 1]
            augmented_targets: torch.Tensor = target[augmentation_mask == 1]
            augmented_loss: torch.Tensor = self._bce_with_logits_loss(
                augmented_samples, augmented_targets
            )

        if augmented_loss.isnan():
            return non_augmented_loss

        overall_loss: torch.Tensor = None

        overall_loss = (
            self._loss_weights[0] * non_augmented_loss + self._loss_weights[1] * augmented_loss
        )

        return overall_loss
