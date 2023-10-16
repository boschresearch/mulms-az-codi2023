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
This module contains the dataset class for the PUBMED dataset.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from source.arg_zoning.az_datasets.az_dataset import AZ_Dataset  # noqa: F401
from source.constants.constants import AZ_DATASET_PATHS

CURRENT_PATH: Path = Path(__file__).parent


class PubmedDataset:
    """
    Class which represents the Pubmed AZ dataset saved as pickle file.
    """

    def __init__(self, name: str, data_split: Tuple) -> None:
        """
        Initializes the Pubmed dataset.

        Args:
            name (str): Name of the dataset object
            data_split (Tuple): Tuple which contains start and end indices for specific split
        """
        self.name = name
        self.data_split = data_split  # tuple with indices (end_of_train, end_of_dev)
        self.categories = []  # list of categories, corresponding to indices below
        self.cat2idx = (
            {}
        )  # map: category --> index, the list of categories (class labels) in the corpus
        self.idx2cat = {}
        self.documents = (
            []
        )  # one list item per document, each document is a list of tuples: (tokenId, tags, categoryId)
        self.doc_ids = []  # IDs of documents
        self.vocab = defaultdict(
            int
        )  # vocabulary of words that occur in the entire data set, with their counts
        self.posVocab = set()  # set of POS tags that have been seen
        self.word2idx = {"<pad>": 0}  # dictionary, entry for padding
        self.idx2word = {0: "<pad>"}  # reverse dictionary (for debugging etc.)
        # vocabulary, but determined only based on training data: mapping from word indices determined earlier
        # to the word indices as they will be used for the weights matrix in the embedding
        self.trainidx2idx = {0: 0}  # 0 for <pad>
        self.idx2trainidx = {0: 0}


def load_pubmed_dataset(path_to_pickle_file=None) -> PubmedDataset:
    """
    Reads Pubmed pickle dataset from disk and returns dataset object.

    Args:
        path_to_pickle_file (str, optional): Path to pickle file on disk. Defaults to None.

    Returns:
        PubmedDataset: PubmedDataset object which contains all data
    """
    if path_to_pickle_file is None:
        path_to_pickle_file = AZ_DATASET_PATHS["PUBMED"]

    sys.path.append(str(CURRENT_PATH))

    with open(path_to_pickle_file, mode="rb") as f:
        dataset: PubmedDataset = pickle.load(f, fix_imports=True)

    return dataset
