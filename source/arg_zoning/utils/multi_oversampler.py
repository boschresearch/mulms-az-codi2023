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
This module contains the oversampling algorithm to address class imbalance
in the MuLMS-AZ corpus.
"""
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import TensorDataset


class MultiLabelOversampler(Sampler):
    """
    This class builds upon the Pytorch sampler to implement oversampling
    in order to address class imbalance.
    """

    def __init__(
        self,
        dataset: TensorDataset,
        num_labels: int,
        percentage: float,
        use_probabilities=False,
        fix_meanIR=False,
        include_new_minlabels=True,
        single_label_only=False,
        sublabel_indices=[],
        oversample_each_epoch=False,
    ):
        """
        Initializes the Multi-Label Oversampler instance

        Args:
            dataset (TensorDataset): AZ dataset
            num_labels (int): Number of distinct labels
            percentage (float): Oversampling percentage
            use_probabilities (bool, optional): Whether to use cloning decision probabilities for weighted dynamic ML-ROS. Defaults to False.
            fix_meanIR (bool, optional): Whether to use an intial and fixed mean IR value. Defaults to False.
            include_new_minlabels (bool, optional): Whether to dynamically add new minority labels. Defaults to True.
            single_label_only (bool, optional): Whether to only oversample single-label instances. Defaults to False.
            sublabel_indices (list, optional): Hierarchy of AZ labels. Defaults to [].
            oversample_each_epoch (bool, optional): Whether to oversample after epoch; changes cloned instances every time.
        """
        self._current_label_counts: dict[int, int] = {}
        self._labels_per_index: dict[int, list] = {}
        self._IRLbl: dict[int, float] = {}
        self._bags: dict[int, list] = None
        self._fix_meanIR: bool = fix_meanIR
        self._include_new_minlabels: bool = include_new_minlabels
        self._num_labels: int = num_labels
        self._dataset: TensorDataset = dataset
        self._single_label_only: bool = single_label_only
        self._use_probabilities: bool = use_probabilities
        self._sublabel_indices: list[int] = sublabel_indices
        self._percentage: float = percentage
        self._samples_to_clone: int = int(len(self._dataset) * self._percentage)
        self._oversample_each_epoch: bool = oversample_each_epoch

        self._indices: list = []
        self._initial_minLabels: list = []
        self._initialMeanIR: float = 0

        self._dataset_length: int = len(self._dataset) + self._samples_to_clone

    def _clone_samples(self) -> None:
        """
        Applies the Mulit-Label Oversampling algorithm and randomly clones instances of minority labels
        """
        print(f"Samples to clone is {self._samples_to_clone}")
        self._indices = []
        self._bags = {}
        for sample_label in range(self._num_labels):
            self._bags[sample_label] = []
            self._IRLbl[sample_label] = []
            self._current_label_counts[sample_label] = 0

        # add samples to labelbags and create initial label counts
        for i, sample in enumerate(self._dataset):
            self._labels_per_index[i] = []
            sample_labels = self._onehot2labellist(sample["label"])
            # append index for all occuring labels
            for sample_label in sample_labels:
                self._bags[int(sample_label)].append(i)
                self._labels_per_index[i].append(int(sample_label))
                self._current_label_counts[sample_label] += 1

        # add all indices once
        stats_label_counts: list = []
        stats_IRLbls: list = []
        stats_MeanIR: list = []

        self._indices += self._labels_per_index.keys()
        samples_to_clone: int = self._samples_to_clone

        # clone random sample from each minority bag
        while samples_to_clone > 0:
            current_minBags = self._get_minBags()

            # for label count, IRLbl and MeanIR stats during oversampling process
            stats_label_counts.append(list(self._current_label_counts.values()))
            stats_IRLbls.append(list(self._IRLbl.values()))
            sum_IRLbl: int = sum([v for v in self._IRLbl.values() if v is not None])
            stats_MeanIR.append(sum_IRLbl / len(self._IRLbl))

            for label in current_minBags:
                minBag: list = current_minBags[label]
                sample_found: bool = False
                while not sample_found:
                    # Pick random sample from minbag
                    min_sample_index: int = random.choice(minBag)
                    if self._dataset._data["isAugmented"][min_sample_index]:
                        continue
                    min_sample_labels: list = self._labels_per_index[min_sample_index]
                    if self._single_label_only:
                        # Don't clone sample if it's not a single-labelled instance (or sublabel)
                        sample_found = not (
                            len(min_sample_labels) > 2
                            if label in self._sublabel_indices
                            else len(min_sample_labels) > 1
                        )
                    elif self._use_probabilities:
                        # when applying weighted dynamic ML-ROS
                        sample_found = self._calculate_sample_score(min_sample_labels)
                    else:
                        sample_found = True
                # Cloned samples are added to the index s.t. they are appearing at least twice during iteration over the dataset
                self._indices.append(min_sample_index)
                # update sample counts for all labels in the sample
                for min_sample_label in min_sample_labels:
                    self._current_label_counts[min_sample_label] += 1
                # decrease remaining counter
                samples_to_clone -= 1
                if samples_to_clone == 0:
                    break
            if len(current_minBags) == 0:
                # Can only happen when MeanIR is not updated or labels cant become new minlabels during oversampling
                print(
                    f"Stopped oversampling after {len(self._indices) - len(self._dataset)} samples because there are no minBags left. Total samples are now {len(self._indices)}"
                )
                break
        # for label count, IRLbl and MeanIR stats during oversampling process
        stats_IRLbls.append(list(self._IRLbl.values()))
        stats_label_counts.append(list(self._current_label_counts.values()))
        sum_IRLbl: int = sum([v for v in self._IRLbl.values() if v is not None])
        stats_MeanIR.append(sum_IRLbl / len(self._IRLbl))
        random.shuffle(self._indices)

    def __iter__(self) -> Iterator:
        """
        Returns an iterator.

        Yields:
            Iterator: Iterator over the sample indices.
        """
        if self._oversample_each_epoch or self._indices == []:
            self._clone_samples()
        return iter(self._indices)

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Size of dataset.
        """
        return self._dataset_length

    def _update_IRLbls(self) -> None:
        """
        Updates the Imbalance Ration score for all classes
        """
        highest_label_count: int = max(self._current_label_counts.values())
        for label in self._current_label_counts:
            self._IRLbl[label] = (
                highest_label_count / self._current_label_counts[label]
                if self._current_label_counts[label] > 0
                else None
            )

    def _onehot2labellist(self, onehot_tensor: torch.Tensor) -> list[int]:
        """
        Generates a list of labels from one-hot encoded tensor

        Args:
            onehot_tensor (torch.Tensor): One-hot encoded label tensor

        Returns:
            list: List which contains labels as numeric values
        """
        return np.argwhere(onehot_tensor == 1).tolist()[0]

    def _get_minBags(self) -> dict:
        """
        Updates IRLbl values and adds label bags to minority bag list if label is minority label (i.e., IRLbl > MeanIR)

        Returns:
            dict: Minority label bag
        """
        minBags: dict = {}
        first_iteration = len(self._initial_minLabels) == 0
        self._update_IRLbls()
        sum_IRLbl: int = sum([v for v in self._IRLbl.values() if v is not None])
        if self._initialMeanIR == 0:
            self._initialMeanIR = sum_IRLbl / len(self._IRLbl)
        MeanIR: float = self._initialMeanIR if self._fix_meanIR else sum_IRLbl / len(self._IRLbl)
        for label in self._current_label_counts:
            if self._IRLbl[label] is not None and self._IRLbl[label] > MeanIR:
                if first_iteration:
                    self._initial_minLabels.append(label)
                    minBags[label] = self._bags[label]
                # New Minority labels are only added if corresponding flag is set to true; otherwise the initial list stays the same
                elif label in self._initial_minLabels or self._include_new_minlabels:
                    minBags[label] = self._bags[label]
        minBagList: list = list(minBags.items())
        random.shuffle(minBagList)
        minBags = dict(minBagList)
        return minBags

    def _calculate_sample_score(self, sample_labels: list) -> bool:
        """
        Calculates weight for weighted dynamic ML-ROS and decided whether to use the current sampling for cloning

        Args:
            sample_labels (list): List of labels corresponding to the current sample

        Returns:
            bool: Whether to accept this sample for cloning
        """
        avg_count = sum(self._current_label_counts.values()) / len(self._current_label_counts)
        avg_count_sample = 0
        for sample_label in sample_labels:
            avg_count_sample += self._current_label_counts[sample_label]
        avg_count_sample = avg_count_sample / len(sample_labels)
        sample_probability = (
            0.5
            * (-0.03 * (avg_count_sample - avg_count))
            / (abs(0.03 * (avg_count_sample - avg_count)) + 1)
            + 0.5
        )
        return random.randrange(100) < sample_probability * 100
