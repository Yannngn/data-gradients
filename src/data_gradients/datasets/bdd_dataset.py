from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data_gradients.datasets.utils import ImageType, PathLike, TransformType

NORMALIZATION_MEANS = [0.485, 0.456, 0.406]
NORMALIZATION_STDS = [0.229, 0.224, 0.225]


class BDDDataset(Dataset):
    """
    PyTorch Dataset implementation of the BDD100K dataset.
    The BDD100K data and annotations can be obtained at https://bdd-data.berkeley.edu/.
    """

    CLASS_NAMES = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "<?>",
    ]

    def __init__(
        self,
        data_folder: PathLike,
        split: str,
        ignore_label: int = 19,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
    ):
        """
        :param data_folder: Folder where data files are stored
        :param split: 'train' or 'test'
        :param ignore_label: label to ignore for certain metrics
        """
        data_location = Path(data_folder) / str(split)
        files_list = data_location.glob("*.jpg")
        self.ignore_label = ignore_label
        self.samples_fn = []

        for f in files_list:
            self.samples_fn.append([data_location / f, data_location / (f.stem + ".png")])

        self.transforms = transform or transforms.Compose([])
        if not isinstance(transform, transforms.Compose):
            self.transforms: TransformType = transforms.Compose([transform])

        self.target_transforms = target_transform or transforms.Compose([])
        if not isinstance(target_transform, transforms.Compose):
            self.target_transforms: TransformType = transforms.Compose([target_transform])

    def get_target(self, target: Image.Image) -> torch.Tensor:
        # Mask as normalized tensor
        mask = self.target_transforms(target)
        mask[mask == 1.0] = self.ignore_label / 255.0
        return mask

    def __getitem__(self, i: int) -> tuple[ImageType, ImageType]:
        image = Image.open(self.samples_fn[i][0]).convert("RGB")
        label = Image.open(self.samples_fn[i][1])
        if self.transforms:
            image = self.transforms(image)
            label = self.get_target(label)

        return image, label

    def __iter__(self) -> Iterator[tuple[ImageType, ImageType]]:
        yield from self

    def __len__(self) -> int:
        return len(self.samples_fn)
