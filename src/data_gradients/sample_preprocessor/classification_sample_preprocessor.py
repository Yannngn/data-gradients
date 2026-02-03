import time
from collections.abc import Iterable, Iterator

import numpy as np

from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor
from data_gradients.utils.data_classes.data_samples import ClassificationSample


class ClassificationSamplePreprocessor(AbstractSamplePreprocessor):
    def __init__(self, data_config: ClassificationDataConfig):
        self.data_config = data_config
        self.adapter = ClassificationDatasetAdapter(data_config=data_config)
        super().__init__(data_config=self.adapter.data_config)

    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[ClassificationSample]:
        for data in dataset:
            images, labels = self.adapter.adapt(data)

            for image, target in zip(images, labels, strict=False):
                class_id = int(target)

                # TODO: Abstract the fact the images are channel last/first and add it to the Image class
                image.data = np.transpose(image.as_numpy(), (1, 2, 0)).astype(np.uint8)
                sample = ClassificationSample(
                    image=image,
                    class_id=class_id,
                    class_names=self.data_config.get_class_names(),  # type: ignore
                    split=split,
                    sample_id=str(time.time()),
                )
                yield sample
