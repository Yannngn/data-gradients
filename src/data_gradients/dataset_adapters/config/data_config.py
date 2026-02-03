import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import platformdirs
import torch

import data_gradients
from data_gradients.dataset_adapters.config.caching_utils import TensorExtractorResolver, XYXYConverterResolver
from data_gradients.dataset_adapters.config.questions import FixedOptionsQuestion, OpenEndedQuestion, text_to_yellow
from data_gradients.dataset_adapters.config.typing_utils import (
    ClassNamesToUseType,
    ClassNamesType,
    ExtractorType,
    JSONDict,
    SupportedDataType,
)
from data_gradients.dataset_adapters.formatters.utils import (
    FloatImageFormat,
    ImageFormat,
    ImageFormatFactory,
    ScaledFloatImageFormat,
    Uint8ImageFormat,
)
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.utils.detection import XYXYConverter
from data_gradients.utils.utils import PathLike, safe_json_load, write_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_cache_dir() -> str:
    return platformdirs.user_cache_dir("DataGradients", "Yannngn")


@dataclass
class DataConfig:
    """Data class for handling Dataset/Dataloader configuration.

    Works as a regular dataclass, but with some additional features:
        - Getter functions that ask the user for information if this information was not provided yet.
        - Caching system, that supports saving and loading of any non-callable attribute.
            Also supports saving and loading from callable defined within DataGradients.
    """

    images_extractor: ExtractorType | None = None
    labels_extractor: ExtractorType | None = None
    is_batch: bool | None = None

    image_channels: ImageChannels | None = None
    image_format: ImageFormat | None = None

    n_classes: int | None = None
    class_names: ClassNamesType | None = None
    class_names_to_use: ClassNamesToUseType | None = None

    cache_path: PathLike | None = None

    def __post_init__(self):
        # Once the object is initialized, we check if the cache is activated or not.
        if self.cache_path is not None and Path(self.cache_path).is_file():
            self.update_from_cache_file()

        else:
            logger.info(f"Cache deactivated for `{self.__class__.__name__}`.")

        self._setup_class_related_params()  # casts class_names to dict and class_names_to_use to list

    def update_from_cache_file(self):
        """Update the values that are not set yet, using the cache file."""
        if self.cache_path is not None and Path(self.cache_path).is_file():
            self._fill_missing_params_with_cache(self.cache_path)

    def dump_cache_file(self):
        """Save the current state to the cache file."""
        if self.cache_path is not None:
            self.write_to_json(self.cache_path)

    def get_caching_info(self) -> str:
        """Get information about the status of the caching."""
        if self.cache_path is None:
            return f"`{self.__class__.__name__}` cache is not enabled because `cache_path={self.cache_path}` was not set."

        return f"`{self.__class__.__name__}` cache is set to `cache_path={self.cache_path}`."

    @classmethod
    def load_from_json(cls, cache_path: PathLike) -> "DataConfig":
        """Load an instance of DataConfig directly from a cache file.
        :param cache_path: Path to the cache file. This should include ".json" extension.
        :return: An instance of DataConfig loaded from the cache file.
        """
        try:
            return cls(**cls._load_json_dict(path=cache_path))
        except TypeError as e:
            raise TypeError(f"{e}\n\t => Could not load `{cls.__name__}` from cache.") from e

    @staticmethod
    def _load_json_dict(path: PathLike) -> dict:
        """Load cache if available."""
        json_dict = safe_json_load(path=path)
        metadata: dict = json_dict.get("metadata", {})
        if not json_dict:
            return {}

        if metadata.get("__version__") == data_gradients.__version__:
            return json_dict.get("attributes", {})

        logger.info(
            f"{path} was not loaded from cache due to data-gradients missmatch between cache and current version"
            f"cache={json_dict.get('__version__')}!={data_gradients.__version__}=installed"
        )
        return {}

    def write_to_json(self, cache_path: PathLike):
        """Save the serializable representation of the class to a .json file.
        :param cache_path: Full path to the cache file. This should end with ".json" extension
        """
        if not Path(cache_path).suffix == ".json":
            raise ValueError(f"`{cache_path}` should end with `.json`")

        json_dict = {"metadata": {"__version__": data_gradients.__version__}, "attributes": self.to_json()}
        write_json(json_dict=json_dict, path=cache_path)

    def to_json(self) -> JSONDict:
        """Convert the dataclass into a serializable representation that can be saved and loaded safely.
        :return: JSON like dictionary, that can be used to create a new instance of the object.
        """
        json_dict = {
            "images_extractor": TensorExtractorResolver.to_string(self.images_extractor),
            "labels_extractor": TensorExtractorResolver.to_string(self.labels_extractor),
            "is_batch": self.is_batch,
            "image_channels": None if self.image_channels is None else self.image_channels.channels_str,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "class_names_to_use": self.class_names_to_use,
            "image_format": None if self.image_format is None else self.image_format.to_json(),
        }
        return json_dict

    @property
    def is_completely_initialized(self) -> bool:
        """Check if all the attributes are set or not."""
        return all(v is not None for v in self.to_json().values())

    def _fill_missing_params_with_cache(self, path: PathLike):
        """Load an instance of DataConfig directly from a cache file.
        :param path: Full path of the cache file. This should end with ".json" extension.
        :return: An instance of DataConfig loaded from the cache file.
        """
        cache_dict = self._load_json_dict(path=path)
        if cache_dict:
            self._fill_missing_params(json_dict=cache_dict)

    def _fill_missing_params(self, json_dict: JSONDict):
        """Overwrite every attribute that is equal to `None`.
        This is the safe way of loading cache, since it will prioritize attributes already set by the user.

        :param json_dict: JSON like dictionary. It's values will overwrite the attributes if these attributes are None
        """
        if self.images_extractor is None:
            self.images_extractor = str(json_dict.get("images_extractor"))

        if self.labels_extractor is None:
            self.labels_extractor = str(json_dict.get("labels_extractor"))

        if self.is_batch is None:
            self.is_batch = bool(json_dict.get("is_batch"))

        if self.n_classes is None:
            n_classes = json_dict.get("n_classes")
            self.n_classes = int(n_classes) if isinstance(n_classes, int | float | str) else None

        if self.class_names is None:
            class_names = json_dict.get("class_names")

            if isinstance(class_names, Mapping):
                self.class_names = {int(k): str(v) for k, v in class_names.items()}
            elif isinstance(class_names, Iterable):
                self.class_names = {int(i): str(name) for i, name in enumerate(class_names)}

        if self.class_names_to_use is None:
            names = json_dict.get("class_names_to_use")

            if isinstance(names, Mapping):
                self.class_names_to_use = [str(name) for name in names.values()]
            elif isinstance(names, Iterable):
                self.class_names_to_use = [str(name) for name in names]

        if self.image_channels is None:
            if json_dict.get("image_channels"):
                self.image_channels = ImageChannels.from_str(str(json_dict.get("image_channels")))

        if self.image_format is None:
            image_format = json_dict.get("image_format", {})
            if isinstance(image_format, Mapping):
                self.image_format = ImageFormatFactory.get_normalizer_from_cache(json_data=image_format)

    def get_images_extractor(
        self, question: FixedOptionsQuestion | None = None, hint: str = ""
    ) -> Callable[[SupportedDataType], torch.Tensor] | None:
        if self.images_extractor is None and question is not None:
            self.images_extractor = question.ask(hint=hint)

        if self.images_extractor is None:
            return None

        return TensorExtractorResolver.to_callable(tensor_extractor=self.images_extractor)

    def get_labels_extractor(
        self, question: FixedOptionsQuestion | None = None, hint: str = ""
    ) -> Callable[[SupportedDataType], torch.Tensor] | None:
        if self.labels_extractor is None and question is not None:
            self.labels_extractor = question.ask(hint=hint)

        if self.labels_extractor is None:
            return None

        return TensorExtractorResolver.to_callable(tensor_extractor=self.labels_extractor)

    def get_image_channels(self, image: torch.Tensor | np.ndarray) -> ImageChannels | None:
        if isinstance(self.image_channels, ImageChannels):
            return self.image_channels

        if 1 in image.shape or len(image.shape) == 2:
            self.image_channels = ImageChannels.from_str("G")

            return self.image_channels

        if 3 in image.shape:
            question = FixedOptionsQuestion(
                question="In which format are your images loaded?",
                options={
                    "RGB": ImageChannels.from_str("RGB"),
                    "BGR": ImageChannels.from_str("BGR"),
                    "LAB": ImageChannels.from_str("LAB"),
                    "Other": ImageChannels.from_str("OOO"),
                },
            )
            self.image_channels = question.ask()

            return self.image_channels

        def _validate_image_channels(channels_str: str) -> bool:
            if len(channels_str) not in image.shape:
                return False
            try:
                ImageChannels.from_str(channels_str=channels_str)
                print(f"image_channels_str={channels_str} is valid with {image.shape}")
                return True
            except ValueError:
                return False

        question = OpenEndedQuestion(question="Please describe your image channels?", validation=_validate_image_channels)
        hint = (
            f"Image Shape: {tuple(image.shape)}\n\n"
            "Enter the channel format representing your image:\n"
            "\n"
            "  > RGB  : Red, Green, Blue\n"
            "  > BGR  : Blue, Green, Red\n"
            "  > G    : Grayscale\n"
            "  > LAB  : Luminance, A and B color channels\n"
            "\n"
            "ADDITIONAL CHANNELS?\n"
            "If your image contains channels other than the standard ones listed above (e.g., Depth, Heat), "
            "prefix them with 'O'. \n"
            "For instance:\n"
            "  > ORGBO: Can represent (Heat, Red, Green, Blue, Depth).\n"
            "  > OBGR:  Can represent (Alpha, Blue, Green, Red).\n"
            "  > BGRO:  Can represent (Blue, Green, Red, Alpha).\n"
            "  > GO:    Can represent (Gray, Depth).\n\n"
            f"IMPORTANT: Make sure that your answer represents all the image channels."
        )

        image_channels_str = question.ask(hint=hint)
        self.image_channels = ImageChannels.from_str(channels_str=image_channels_str)

        return self.image_channels

    def get_is_batch(self, hint: str = "") -> bool:
        if self.is_batch is None:
            question = FixedOptionsQuestion(
                question="Does your dataset provide a batch or a single sample?",
                options={
                    "Batch of Samples (e.g. torch Dataloader)": True,
                    "Single Sample (e.g. torch Dataset)": False,
                },
            )
            self.is_batch = question.ask(hint=hint)

        return self.is_batch or False

    def get_class_names(self) -> dict[int, str] | None:
        if self.class_names is None:
            self._setup_class_related_params()

        assert isinstance(self.class_names, dict | None)

        return self.class_names  # type: ignore[returnType]

    def get_n_classes(self) -> int | None:
        if self.n_classes is None:
            self._setup_class_related_params()
        return self.n_classes

    def get_class_names_to_use(self) -> list[str] | None:
        if self.class_names_to_use is None:
            self._setup_class_related_params()

        assert isinstance(self.class_names_to_use, list | None)

        return self.class_names_to_use

    def _setup_class_related_params(self):
        """Resolve class related params.

        All the parameters are set up together because strongly related -
        knowing only `class_names` or `n_classes` is enough to set the values of the other 2.
        """

        self.class_names = resolve_class_names(class_names=self.class_names, n_classes=self.n_classes)
        self.n_classes = len(self.class_names)
        self.class_names_to_use = resolve_class_names_to_use(class_names=self.class_names, class_names_to_use=self.class_names_to_use)

    def get_image_format(self, images: torch.Tensor | np.ndarray) -> ImageFormat:
        if self.image_format is not None:
            return self.image_format

        # Check if images are already in the range 0-1
        if 0 <= images.min() and images.max() <= 1:
            self.image_format = FloatImageFormat()
            return self.image_format

        # Check if images are in the range 0-255
        if 0 <= images.min() and images.max() <= 255:
            self.image_format = Uint8ImageFormat()
            return self.image_format

        # For standardized normalizer, we need to ask user for mean and std

        question = OpenEndedQuestion(
            question="Enter the mean values for image normalization (comma-separated, e.g., `0.485, 0.456, 0.406`):",
            validation=_validate_float_list,
        )
        mean_str: str = question.ask()
        mean = [float(x.strip()) for x in mean_str.split(",")]
        logger.debug("mean: ", mean)

        question = OpenEndedQuestion(
            question="Enter the std deviation values for image normalization (comma-separated, e.g., `0.229, 0.224, 0.225`):",
            validation=_validate_float_list,
        )
        std_str: str = question.ask()
        std = [float(x.strip()) for x in std_str.split(",")]
        logger.debug("std: ", std)

        self.image_format = ScaledFloatImageFormat(mean=mean, std=std)
        return self.image_format


def _validate_float_list(value_str: str) -> bool:
    try:
        values = [float(x.strip()) for x in value_str.split(",")]
        return len(values) > 0
    except ValueError:
        return False


@dataclass
class ImageChannel:
    channel_names: list[str]
    channels_idx_to_visualize: list[str]
    rgb_converter: Callable[[np.ndarray], np.ndarray]


@dataclass
class ClassificationDataConfig(DataConfig):
    pass


@dataclass
class SegmentationDataConfig(DataConfig):
    pass


@dataclass
class DetectionDataConfig(DataConfig):
    is_label_first: bool | None = None
    xyxy_converter: str | Callable[[torch.Tensor], torch.Tensor] | None = None

    def to_json(self) -> JSONDict:
        parent_json = super().to_json()
        result: dict = {**cast(dict, parent_json)}
        result.update(
            {
                "is_label_first": self.is_label_first,
                "xyxy_converter": XYXYConverterResolver.to_string(self.xyxy_converter),
            }
        )
        return result

    def _fill_missing_params(self, json_dict: JSONDict):
        super()._fill_missing_params(json_dict=json_dict)

        if self.is_label_first is None:
            self.is_label_first = bool(json_dict.get("is_label_first"))

        if self.xyxy_converter is None:
            self.xyxy_converter = str(json_dict.get("xyxy_converter"))

    def get_is_label_first(self, hint: str = "") -> bool:
        if self.is_label_first is None:
            question = FixedOptionsQuestion(
                question=f"{text_to_yellow('Which comes first')} in your annotations, the class id or the bounding box?",
                options={
                    "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
                    "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
                },
            )
            self.is_label_first = question.ask(hint=hint)

            if self.is_label_first is None:
                raise RuntimeError("`is_label_first` could not be determined.")

        return self.is_label_first

    def get_xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor] | None:
        if self.xyxy_converter is None:
            question = FixedOptionsQuestion(
                question=f"What is the {text_to_yellow('bounding box format')}?",
                options=XYXYConverter.get_available_options(),
            )
            self.xyxy_converter = question.ask(hint=hint)

        if self.xyxy_converter is None:
            return None

        return XYXYConverterResolver.to_callable(self.xyxy_converter)


def resolve_class_names(class_names: ClassNamesType | None, n_classes: int | None = None) -> dict[int, str]:
    """Ensure that either `class_names` or `n_classes` is specified, but not both. Return the list of class names that will be used."""
    if n_classes and class_names and (len(class_names) != n_classes):
        raise RuntimeError(f"`len(class_names)={len(class_names)} != n_classes`.")

    if n_classes is None and class_names is None:

        def _represents_int(s: str) -> bool:
            """Check if a string represents an integer."""
            try:
                int(s)
            except ValueError:
                return False
            else:
                return True

        question = OpenEndedQuestion(
            question="How many classes does your dataset include?",
            validation=lambda answer: _represents_int(answer) and int(answer) > 0,
        )
        n_classes = int(question.ask())
        return {i: f"class_{i}" for i in range(n_classes)}

    if class_names:
        if isinstance(class_names, Mapping):
            return dict(class_names)

        if isinstance(class_names, torch.Tensor):
            class_names = dict(enumerate(map(str, class_names.tolist())))

        if isinstance(class_names, Iterable):
            return dict(enumerate(map(str, class_names)))

    if n_classes is None:
        raise RuntimeError("Either `class_names` or `n_classes` must be specified.")

    return {i: f"class_{i}" for i in range(n_classes)}


def resolve_class_names_to_use(class_names: ClassNamesType, class_names_to_use: ClassNamesToUseType | None) -> list[str]:
    """Define `class_names_to_use` from `class_names` if it is specified. Otherwise, return the list of class names that will be used."""
    class_names = resolve_class_names(class_names=class_names, n_classes=None)
    if class_names_to_use:
        if isinstance(class_names_to_use, torch.Tensor):
            class_names_to_use = list(map(str, class_names_to_use.tolist()))
        invalid_class_names_to_use = set(class_names_to_use) - set(class_names.values())
        if invalid_class_names_to_use != set():
            raise RuntimeError(
                f"You defined `class_names_to_use` with classes that are not listed in `class_names`: {invalid_class_names_to_use}"
            )
        return list(class_names_to_use)

    return list(class_names.values())
