from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from data_gradients.visualize.plot_options import CommonPlotOptions


@dataclass
class Feature:
    """Feature extracted from the whole dataset."""

    data: dict | pd.DataFrame | np.ndarray | Figure | None
    plot_options: CommonPlotOptions | None

    json: dict | list

    title: str
    description: str
    notice: str | None = None
    warning: str | None = None


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def update(self, sample: Any):  # ImageSample
        """Accumulate information about samples"""
        raise NotImplementedError()

    @abstractmethod
    def aggregate(self) -> Feature:
        raise NotImplementedError()

    @abstractmethod
    def setup_data_sources(self, train_data: Iterable, val_data: Iterable):
        """
        Called in AnalysisManagerAbstract.__init__ for the purpose of exposing train_data and val_data
         to the feature in case some information is needed.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__


FeatureExtractorsType: TypeAlias = (
    list[str | AbstractFeatureExtractor | type[AbstractFeatureExtractor]] | str | AbstractFeatureExtractor | type[AbstractFeatureExtractor]
)


class NoSourceFeatureExtractor(AbstractFeatureExtractor):
    def update(self, sample: Any):
        """No source feature extractor does not need to update anything."""
        pass

    def aggregate(self) -> Feature:
        """No source feature extractor does not have any data to aggregate."""
        return Feature(
            data=None,
            plot_options=None,
            json={},
            title="No Source Feature Extractor",
            description="This feature extractor does not extract any data.",
        )

    def setup_data_sources(self, train_data: Iterable, val_data: Iterable):
        """
        Called in AnalysisManagerAbstract.__init__ for the purpose of exposing train_data and val_data
         to the feature in case some information is needed.
        """
        return


#
