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
This module contains the multi-task dataloader.
"""
from itertools import zip_longest


class MultitaskDataloader:
    """
    This class stores multiple dataloaders and yields batches from each one
    in an alternating fashion.
    """

    def __init__(self, *dataloader) -> None:
        """
        Initializes the multi-task dataloader by taking
        an iterable of Pytorch dataloaders.
        """
        self._dataloader = dataloader
        self._batch_idx_to_dataset: dict = dict(
            [(k, v.dataset.name) for k, v in enumerate(self._dataloader)]
        )

    def __iter__(self):
        """
        Returns a batch from one of the dataloaders. The order of dataloaders
        is cyclic, i.e., it draws a batch from one dataset, then the next one
        until it stars again from the beginning dataset.

        Yields:
            A batch from one of the dataloaders.
        """
        for batches in zip_longest(*self._dataloader, fillvalue=None):
            for i, batch in enumerate(batches):
                if batch is not None:
                    batch["dataset"] = i
                    yield batch

    def __len__(self) -> int:
        """
        Returns the sum of legths of each dataset.

        Returns:
            int: The sum of lengths
        """
        return sum([len(dl) for dl in self._dataloader])
