from dataclasses import dataclass

import numpy as np
import torch

from data_gradients.dataset_adapters.config.typing_utils import ClassNamesType
from data_gradients.dataset_adapters.formatters.utils import FloatImageFormat, ImageFormat, ScaledFloatImageFormat, Uint8ImageFormat
from data_gradients.utils.data_classes.contour import Contour
from data_gradients.utils.data_classes.image_channels import ImageChannels


@dataclass
class Image:
    data: torch.Tensor | np.ndarray
    format: ImageFormat
    channels: ImageChannels

    def to_uint8(self) -> "Image":
        return self._to_format(target_format=Uint8ImageFormat())

    def to_float(self) -> "Image":
        return self._to_format(target_format=FloatImageFormat())

    def to_scaled_float(self, mean: list[float], std: list[float]) -> "Image":
        return self._to_format(target_format=ScaledFloatImageFormat(mean=mean, std=std))

    def _to_format(self, target_format: ImageFormat) -> "Image":
        if isinstance(target_format, type(self.format)):
            return self
        else:
            float_image = self.format.convert_image_to_float(images=self.data)
            return Image(data=target_format.convert_image_from_float(images=float_image), format=target_format, channels=self.channels)

    @property
    def shape(self):
        return self.data.shape

    def as_numpy(self) -> np.ndarray:
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data

    def as_torch(self) -> torch.Tensor:
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        return self.data

    def as_rgb(self) -> np.ndarray:
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"`image_as_rgb` is only available for numpy arrays. Got `{type(self.data)}`.")
        nd_image: np.ndarray = np.array(self.to_uint8().data)
        return self.channels.convert_image_to_rgb(image=nd_image)

    @property
    def channels_to_visualize(self) -> np.ndarray:
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"`channels_to_visualize` is only available for numpy arrays. Got `{type(self.data)}`.")
        nd_image: np.ndarray = np.array(self.to_uint8().data)
        return self.channels.get_channels_to_visualize(image=nd_image)

    @property
    def mean_intensity(self) -> float:
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"`mean_intensity` is only available for numpy arrays. Got `{type(self.data)}`.")
        nd_image: np.ndarray = np.array(self.to_uint8().data)
        return self.channels.compute_mean_image_intensity(image=nd_image)


@dataclass
class ImageSample:
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image.

    :attr sample_id:The unique identifier of the sample. Could be the image path or the image name.
    :attr split:    The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:    np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    """

    sample_id: str
    split: str
    image: Image

    def __repr__(self) -> str:
        return f"ImageSample(sample_id={self.sample_id}, image={self.image.data.shape}, format={self.image.channels})"


@dataclass
class SegmentationSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    :attr sample_id:        The unique identifier of the sample. Could be the image path or the image name.
    :attr split:            The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:            np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr mask:             np.ndarray of shape [H, W], categorical representation of the mask.
    :attr contours:         A list of contours for each class in the mask.
    :attr class_names:      List of all class names in the dataset. The index should represent the class_id.
    """

    mask: np.ndarray

    contours: list[list[Contour]]
    class_names: dict[int, str]

    def __repr__(self) -> str:
        return f"SegmentationSample(sample_id={self.sample_id}, image={self.image.shape}, mask={self.mask.shape})"


@dataclass
class DetectionSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    :attr sample_id:    The unique identifier of the sample. Could be the image path or the image name.
    :attr split:        The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:        np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr bboxes_xyxy:  np.ndarray of shape [N, 4] (X, Y, X, Y)
    :attr class_ids:    np.ndarray of shape [N, ]
    :attr class_names:  Dict[int, str] of all class names in the dataset. The key should represent the class_id.
    """

    bboxes_xyxy: np.ndarray
    class_ids: np.ndarray
    class_names: dict[int, str]

    def __repr__(self) -> str:
        return (
            f"DetectionSample(sample_id={self.sample_id}, image={self.image.shape}, "
            f"bboxes_xyxy={self.bboxes_xyxy.shape}, class_ids={self.class_ids.shape})"
        )


@dataclass
class ClassificationSample(ImageSample):
    """
    This is a dataclass that represents a single classification sample of the dataset where input to the model is
    a single image and the target is an image label.

    :attr sample_id:    The unique identifier of the sample. Could be the image path or the image name.
    :attr split:        The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:        np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr class_label:  Class label (int)
    :attr class_names:  Dict[int, str] of all class names in the dataset. The key should represent the class_id.
    """

    class_id: int
    class_names: ClassNamesType

    def __post_init__(self):
        if self.class_id not in self.class_names:
            raise ValueError(f"Class ID {self.class_id} not found in class names dictionary.")
        # Coerce class_names into dict[int, str]
        cn = self.class_names
        if cn is None:
            self.class_names = {}
        elif isinstance(cn, dict):
            coerced: dict[int, str] = {}
            for k, v in cn.items():
                try:
                    ik = int(k)
                except Exception as e:
                    raise ValueError(f"class_names dict keys must be ints or int-convertible strings. Got key={k!r}") from e
                coerced[ik] = str(v)
            self.class_names = coerced
        elif isinstance(cn, list | tuple):
            self.class_names = {int(i): str(name) for i, name in enumerate(cn)}
        else:
            raise ValueError(f"Unsupported ClassNamesType: {type(cn)}. Expected dict[int,str] or list[str].")

        if self.class_id not in self.class_names:
            raise ValueError(f"Class ID {self.class_id} not found in class names dictionary.")

    def __repr__(self) -> str:
        return (
            f"ClassificationSample(sample_id={self.sample_id}, image={self.image.shape}, "
            f"label={self.class_id}, name={self.class_names.get(self.class_id, 'Unknown')})"  # type: ignore
        )


def images_list_to_tensor(images: list[Image]) -> torch.Tensor:
    """Convert a list of Image dataclasses to a single tensor.

    :param images:  List of Image dataclasses.
    :return:        Tensor of shape [BS, C, H, W].
    """
    image_tensors = [img.as_torch() for img in images]
    return torch.stack(image_tensors)
