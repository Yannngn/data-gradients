import time
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor
from data_gradients.sample_preprocessor.utils.contours import get_contours
from data_gradients.utils.data_classes import SegmentationSample


class SegmentationSampleProcessor(AbstractSamplePreprocessor):
    def __init__(self, data_config: SegmentationDataConfig, threshold_soft_labels: float):
        self.data_config = data_config
        self.adapter = SegmentationDatasetAdapter(data_config=data_config, threshold_soft_labels=threshold_soft_labels)
        super().__init__(data_config=self.adapter.data_config)

    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[SegmentationSample]:
        for data in dataset:
            images, labels = self.adapter.adapt(data)

            if isinstance(labels, list):
                labels = torch.stack(labels, dim=0)

            labels = labels.cpu().numpy().astype(np.uint8)

            for image, mask in zip(images, labels, strict=False):
                contours = get_contours(mask, class_ids=list(self.data_config.get_class_names()))  # type: ignore

                # TODO: Abstract the fact the images are channel last/first and add it to the Image class
                image.data = np.transpose(image.as_numpy(), (1, 2, 0)).astype(np.uint8)
                yield SegmentationSample(
                    image=image,
                    mask=mask,
                    contours=contours,
                    class_names=self.data_config.get_class_names(),  # type: ignore
                    split=split,
                    sample_id=str(time.time()),
                )
