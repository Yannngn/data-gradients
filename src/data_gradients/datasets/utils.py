from pathlib import Path
from typing import Protocol, TypeAlias

import cv2
import numpy as np
import torch
from PIL import Image

PathLike: TypeAlias = str | Path

ImageType: TypeAlias = Image.Image | torch.Tensor


class TransformType(Protocol):
    # Can be any callable that takes in a PIL Image and returns a torch Tensor
    # ToTensor of Compose with ToTensor both satisfy this Protocol
    def __call__(self, img: Image.Image) -> torch.Tensor: ...


def load_image_rgb(path: PathLike) -> np.ndarray:
    """Load an image from a path in a RGB.

    :return: The image as a numpy array. (H, W, 3)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Image unable to load: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
