from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from data_gradients.feature_extractors.abstract_feature_extractor import Feature, NoSourceFeatureExtractor
from data_gradients.visualize.images import combine_images_per_split_per_class
from data_gradients.visualize.seaborn_renderer import FigureRenderer


class BaseClassHeatmap(NoSourceFeatureExtractor, ABC):
    def __init__(self, n_rows: int = 12, n_cols: int = 2, heatmap_shape: tuple[int, int] = (200, 200)):
        """
        :param n_rows:          How many rows per split.
        :param n_cols:          How many columns per split.
        :param heatmap_shape:   Heatmap, in (H, W) format. Increase for more resolution, at the expense of processing speed.
        """
        self.heatmap_shape = heatmap_shape
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.class_names: dict[int, str] = {}
        self.heatmaps_per_split: dict[str, np.ndarray] = {}  # Each heatmap should be of shape (n_class, heatmap_shape[0], heatmap_shape[1])

    @abstractmethod
    def update(self, sample: Any):  # SegmentationSample
        ...

    def aggregate(self) -> Feature:
        # Select top k heatmaps by appearance
        split_count: np.ndarray = np.sum([split_heatmap.sum(axis=(1, 2)) for split_heatmap in self.heatmaps_per_split.values()], axis=0)
        most_used_class_ids = (-split_count).argsort()[: self.n_rows * self.n_cols]

        # Normalize (0-1)
        normalized_heatmaps_per_split_per_cls = defaultdict(dict)
        for split, heatmaps in self.heatmaps_per_split.items():
            for class_id, heatmap in enumerate(heatmaps):
                if class_id in most_used_class_ids:
                    class_name = self.class_names[class_id]
                    normalized_heatmaps_per_split_per_cls[class_name][split] = (255 * (heatmap / (heatmap.max() + 1e-6))).astype(np.uint8)

        fig = combine_images_per_split_per_class(images_per_split_per_class=normalized_heatmaps_per_split_per_cls, n_cols=self.n_cols)
        plot_options = FigureRenderer()
        json = dict.fromkeys(normalized_heatmaps_per_split_per_cls.keys(), "No Data")

        feature = Feature(
            data=fig,
            plot_options=plot_options,
            json=json,
            title=self._generate_title(),
            description=self._generate_description(),
            notice=self._generate_notice(),
        )
        return feature

    @abstractmethod
    def _generate_title(self) -> str: ...

    @abstractmethod
    def _generate_description(self) -> str: ...

    @abstractmethod
    def _generate_notice(self) -> str: ...
