from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from torch.utils.data.dataset import Dataset

from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS, ImageLabelFilesIterator
from data_gradients.datasets.utils import PathLike, load_image_rgb


class BaseImageLabelDirectoryDataset(Dataset, ABC):
    """Base class for any dataset that is primarily made of an image and label directories."""

    def __init__(
        self,
        root_dir: PathLike,
        images_subdir: PathLike,
        labels_subdir: PathLike,
        label_extensions: Iterable[str],
        image_extensions: Iterable[str] = DEFAULT_IMG_EXTENSIONS,
        config_path: PathLike | None = None,
        verbose: bool = False,
    ):
        """
        :param root_dir:            Where the data is stored.
        :param images_subdir:       Local path to directory that includes all the images. Path relative to `root_dir`.
                                    Can be the same as `labels_subdir`.
        :param labels_subdir:       Local path to directory that includes all the labels. Path relative to `root_dir`.
                                    Can be the same as `images_subdir`.
        :param image_extensions:    List of image file extensions to load from.
        :param label_extensions:    List of label file extensions to load from.
        :param config_path:         Path to an optional config file. This config file should contain the list of file ids to include.
                                    If None, all the available images and labels will be loaded.
        :param verbose:             Whether to show extra information during loading.
        """
        root_dir = Path(root_dir)
        config_path = root_dir / config_path if config_path is not None else None
        self.image_label_tuples = ImageLabelFilesIterator(
            images_dir=root_dir / images_subdir,
            labels_dir=root_dir / labels_subdir,
            config_path=config_path,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
            verbose=verbose,
        )

    def load_image(self, path: PathLike) -> np.ndarray:
        """Load an image from the given path into RGB format."""
        return load_image_rgb(path)

    @abstractmethod
    def load_labels(self, path: PathLike) -> np.ndarray: ...

    def __len__(self) -> int:
        return len(self.image_label_tuples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        paths = self.image_label_tuples[index]
        image_path, labels_path = paths[0]  # FIXME: if it breaks the error is here
        image = self.load_image(path=image_path)
        labels = self.load_labels(path=labels_path)
        return image, labels

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self


###
###
###
###
###
###
