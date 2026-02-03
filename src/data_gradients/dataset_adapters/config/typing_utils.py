from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import numpy.typing as npt
import torch

SupportedDataType: TypeAlias = tuple | list | Mapping
JSONValue: TypeAlias = str | int | float | bool | None | dict[str, "JSONValue | list[JSONValue]"]
JSONDict: TypeAlias = dict[str, JSONValue]


ClassNamesToUseType: TypeAlias = list[str] | tuple[str] | npt.NDArray | torch.Tensor
ClassNamesType: TypeAlias = ClassNamesToUseType | Mapping[int, str]

ExtractorType: TypeAlias = str | Callable[[SupportedDataType], torch.Tensor]
PathLike: TypeAlias = str | Path
