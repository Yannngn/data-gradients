from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from jinja2 import Template

from data_gradients.assets import assets
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors import NoSourceFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes.data_samples import DetectionSample, ImageSample, SegmentationSample


@dataclass
class BasicStatistics:
    num_samples: int = 0
    classes_count: int = 0
    classes_in_use: int = 0
    classes: list[int] = field(default_factory=list)
    num_annotations: int = 0
    images_without_annotation: int = 0
    images_resolutions: list[tuple[int, int]] = field(default_factory=list)
    annotations_sizes: list[float] = field(default_factory=list)
    annotations_per_image: list[int] = field(default_factory=list)
    med_image_resolution: int = 0
    smallest_annotations: int = 0
    largest_annotations: int = 0
    most_annotations: int = 0
    least_annotations: int = 0

    def update(self, sample: ImageSample):
        height, width = sample.image.shape[:2]
        self.images_resolutions.append((height, width))
        self.num_samples += 1

        if isinstance(sample, SegmentationSample):
            contours = [contour for sublist in sample.contours for contour in sublist]
            self.annotations_per_image.append(len(contours))

            for contour in contours:
                self.annotations_sizes.append(contour.area)
                self.classes.append(contour.class_id)

            self.classes_count = len(sample.class_names)
            return

        if isinstance(sample, DetectionSample):
            labels = sample.class_ids
            self.classes.extend(labels)
            boxes = sample.bboxes_xyxy
            self.annotations_per_image.append(len(boxes))
            for box in boxes:
                self.annotations_sizes.append((box[2] - box[0]) * (box[3] - box[1]))

            self.classes_count = len(sample.class_names)
            return

    def to_dict(self) -> dict | None:
        if self.num_samples > 0:
            areas = np.array(self.images_resolutions)[:, 0] * np.array(self.images_resolutions)[:, 1]
            sorted_idx = np.argsort(areas)
            index_of_med = int(sorted_idx[len(sorted_idx) // 2])

            return {
                "classes_in_use": len(set(self.classes)),
                "num_annotations": int(np.sum(self.annotations_per_image)),
                "images_without_annotation": np.count_nonzero(self.annotations_per_image == 0),
                "smallest_annotations": int(np.min(self.annotations_sizes)),
                "largest_annotations": int(np.max(self.annotations_sizes)),
                "most_annotations": int(np.max(self.annotations_per_image)),
                "least_annotations": int(np.min(self.annotations_per_image)),
                "med_image_resolution": self.format_resolution(self.images_resolutions[index_of_med]),
                "annotations_per_image": f"{np.mean(self.annotations_per_image) if len(self.annotations_per_image) > 0 else 0:.2f}",
                "num_samples": int(self.num_samples),
                # To support JSON - delete arrays
                "classes": None,  # np.array(self.classes)
                "images_resolutions": None,  # np.array(self.images_resolutions),
                "annotations_sizes": None,  # np.array(self.annotations_sizes),
            }

    @staticmethod
    def format_resolution(array: Iterable) -> str:
        return "x".join([str(int(x)) for x in array])


@register_feature_extractor()
class SummaryStats(NoSourceFeatureExtractor):
    """
    Gathers basic statistical data from the dataset.

    This extractor compiles essential statistics from the image samples. It counts the number of images, annotations, and classes,
    assesses the diversity of image resolutions, and measures the size of annotations. This data is crucial for getting a high-level
    overview of the dataset's characteristics and composition.
    """

    def __init__(self):
        super().__init__()
        self.stats = {"train": BasicStatistics(), "val": BasicStatistics()}

        self.template: Template = Template(source=assets.html.basic_info_fe)

    def update(self, sample: ImageSample):
        basic_stats = self.stats[sample.split]

        basic_stats.update(sample)

    def aggregate(self) -> Feature:
        json_res = {}
        for split, basic_stats in self.stats.items():
            json_res[split] = basic_stats.to_dict()

        feature = Feature(
            data=None,
            plot_options=None,
            json=json_res,
            title="General Statistics",
            description=self.template.render(**self.stats),
        )
        return feature


###


###


###


###
