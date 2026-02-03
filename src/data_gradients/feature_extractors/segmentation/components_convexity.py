import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature, NoSourceFeatureExtractor
from data_gradients.sample_preprocessor.utils import contours
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import KDEPlotOptions


@register_feature_extractor()
class SegmentationComponentsConvexity(NoSourceFeatureExtractor):
    """
    Assesses the convexity of segmented objects within images of a dataset and presents the distribution across dataset splits.

    Higher convexity values suggest complex structures that may pose challenges for accurate segmentation.
    """

    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for class_channel in sample.contours:
            for contour in class_channel:
                convex_hull = contours.get_convex_hull(contour)
                convex_hull_perimeter = contours.get_contour_perimeter(convex_hull)
                convexity_measure = (contour.perimeter - convex_hull_perimeter) / contour.perimeter
                self.data.append(
                    {
                        "split": sample.split,
                        "convexity_measure": convexity_measure,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = KDEPlotOptions(
            x_label_key="convexity_measure",
            x_label_name="Convexity",
            x_ticks_rotation=None,
            labels_key="split",
            common_norm=False,
            fill=True,
            sharey=True,
        )

        json = {
            "train": df[df["split"] == "train"]["convexity_measure"].describe().to_dict(),
            "val": df[df["split"] == "val"]["convexity_measure"].describe().to_dict(),
        }

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
            title="Object Convexity",
            description=(
                "This graph depicts the convexity distribution of objects in both training and validation sets. \n"
                "Higher convexity values suggest complex structures that may pose challenges for accurate segmentation."
            ),
        )
        return feature
