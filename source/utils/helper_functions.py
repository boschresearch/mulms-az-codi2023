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
This module contains useful helper functions.
"""
from argparse import Namespace
from typing import Iterable, Tuple

import torch

from source.constants.constants import CPU, CUDA


def print_cmd_args(args: Namespace) -> None:
    """
    Prints all command line arguments to stdout. Useful for writing training arguments into log files.

    Args:
        args (Namespace): The arguments object
    """
    for arg in args._get_kwargs():
        print(arg[0], ":", arg[1])


def get_executor_device(disable_cuda: bool) -> str:
    """
    Returns the string constant that corresponds to the suited
    computation device.

    Args:
        disable_cuda (bool): Whether to disable GPU backend.

    Returns:
        str: Device string constant
    """
    return CUDA if not disable_cuda and torch.cuda.is_available() else CPU


def move_to_device(device: str, *tensors: Iterable[torch.Tensor]) -> Tuple[torch.Tensor]:
    """
    Moves a tuple of tensors to the specified memory space.

    Args:
        device (str): The memory space to move to (GPU or CPU)

    Returns:
        Tuple[torch.Tensor]: References to the moved tensors
    """
    return tuple([t.to(device) for t in tensors])
