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
This module contains helper functions for the data reader.
"""

from puima.uimatypes import *  # noqa: F403


def get_token_indices_for_annot(annot, sent_tokens, doc):
    """
    Retrieves indices of tokens within the list of sententence tokens (sent_tokens) for
    the annotation annot.
    param: annot - puima Annotation object
    param: sent_tokens - list of puima Annotation objects (tokens of the sentence)
    """
    indices = [None, None]
    for annot_token in doc.select_covered(TOKEN_TYPE, annot):  # noqa: F405
        token_index = sent_tokens.index(annot_token)
        if indices[0] is None or indices[0] > token_index:
            indices[0] = token_index
        if indices[1] is None or indices[1] < token_index:
            indices[1] = token_index
    return tuple(indices)


def get_token_index_for_annot_if_subtoken(annot, sent_tokens, doc):
    """
    Retrieves indices of tokens within the list of sententence tokens (sent_tokens) for
    the annotation annot.
    param: annot - puima Annotation object
    param: sent_tokens - list of puima Annotation objects (tokens of the sentence)
    """
    annot_token = next(doc.select_covering(TOKEN_TYPE, annot))  # noqa: F405
    token_index = sent_tokens.index(annot_token)
    return (token_index, token_index)
