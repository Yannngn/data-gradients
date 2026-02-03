from collections.abc import Iterable
from pathlib import Path

from torch.utils.data import DataLoader

from data_gradients.config.utils import get_grouped_feature_extractors
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig, get_default_cache_dir
from data_gradients.dataset_adapters.config.typing_utils import (
    ClassNamesToUseType,
    ClassNamesType,
    ExtractorType,
    PathLike,
    SupportedDataType,
)
from data_gradients.dataset_adapters.formatters.utils import ImageFormat
from data_gradients.datasets import COCOFormatSegmentationDataset, COCOSegmentationDataset, VOCSegmentationDataset
from data_gradients.feature_extractors import FeatureExtractorsType
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.sample_preprocessor.segmentation_sample_preprocessor import SegmentationSampleProcessor
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.utils.summary_writer import SummaryWriter


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    """
    Main semantic segmentation manager class.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        report_title: str,
        train_data: Iterable[SupportedDataType],
        val_data: Iterable[SupportedDataType],
        report_subtitle: str | None = None,
        config_path: PathLike | None = None,
        feature_extractors: FeatureExtractorsType | None = None,
        log_dir: PathLike | None = None,
        use_cache: bool = False,
        class_names: ClassNamesType | None = None,
        class_names_to_use: ClassNamesToUseType | None = None,
        n_classes: int | None = None,
        images_extractor: ExtractorType | None = None,
        labels_extractor: ExtractorType | None = None,
        is_batch: bool | None = None,
        image_channels: ImageChannels | None = None,
        image_format: ImageFormat | None = None,
        threshold_soft_labels: float = 0.5,
        batches_early_stop: int | None = None,
        remove_plots_after_report: bool | None = True,
    ):
        """
        Constructor of semantic-segmentation manager which controls the analyzer

        :param report_title:            Title of the report. Will be used to save the report
        :param report_subtitle:         Subtitle of the report
        :param class_names:             Either the list of all class names in the dataset OR dictionary mapping of `class_id` -> `class_name`.
                                        The index should represent the class_id. Mutually exclusive with `n_classes`
        :param class_names_to_use:      List of class names that we should use for analysis.
        :param n_classes:               Number of classes. Mutually exclusive with `class_names`.
                                        If set, `class_names` will be a list of `class_ids`.
        :param train_data:              Iterable object contains images and labels of the training dataset
        :param val_data:                Iterable object contains images and labels of the validation dataset
        :param config_path:             Full path the hydra configuration file. If None, the default configuration will be used.
                                        Mutually exclusive with feature_extractors
        :param feature_extractors:      One or more feature extractors to use. If None, the default configuration will be used.
                                        Mutually exclusive with config_path
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param use_cache:               Whether to use cache or not for the configuration of the data.
        :param images_extractor:        Function extracting the image(s) out of the data output.
        :param labels_extractor:        Function extracting the label(s) out of the data output.
        :param image_channels:          Image channels to use.
        :param image_format:            Image format to use. Can be Uint8ImageFormat, FloatImageFormat, ScaledFloatImageFormat.
        :param threshold_soft_labels:   Threshold for converting soft labels to binary labels
        :param remove_plots_after_report:  Delete the plots from the report directory after the report is generated. By default, True
        """
        if feature_extractors is not None and config_path is not None:
            raise RuntimeError("`feature_extractors` and `config_path` cannot be specified at the same time")

        summary_writer = SummaryWriter(report_title=report_title, report_subtitle=report_subtitle, log_dir=log_dir)
        cache_path = Path(get_default_cache_dir()) / f"{summary_writer.run_name}.json" if use_cache else None
        data_config = SegmentationDataConfig(
            class_names=class_names,
            n_classes=n_classes,
            cache_path=cache_path,
            class_names_to_use=class_names_to_use,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            is_batch=is_batch,
            image_channels=image_channels,
            image_format=image_format,
        )

        sample_preprocessor = SegmentationSampleProcessor(data_config=data_config, threshold_soft_labels=threshold_soft_labels)
        grouped_feature_extractors = get_grouped_feature_extractors(
            default_config_name="segmentation", config_path=config_path, feature_extractors=feature_extractors
        )

        super().__init__(
            train_data=train_data,
            val_data=val_data,
            sample_preprocessor=sample_preprocessor,
            summary_writer=summary_writer,
            grouped_feature_extractors=grouped_feature_extractors,
            batches_early_stop=batches_early_stop,
            remove_plots_after_report=remove_plots_after_report,
        )

    @classmethod
    def analyze_coco(
        cls,
        *,
        root_dir: PathLike,
        year: str | int,
        report_title: str,
        report_subtitle: str | None = None,
        config_path: PathLike | None = None,
        feature_extractors: FeatureExtractorsType | None = None,
        log_dir: PathLike | None = None,
        use_cache: bool = False,
        class_names_to_use: ClassNamesToUseType | None = None,
        batches_early_stop: int | None = None,
        remove_plots_after_report: bool | None = True,
    ):
        """
        Class method to create a semantic-segmentation manager instance from data in COCO format.

        :param root_dir:                       Directory containing the COCO formatted dataset.
        :param year:                           Year or version of the COCO dataset.
        :param report_title:                   Title of the report. Will be used to save the report.
        :param report_subtitle:                Subtitle of the report.
        :param config_path:                    Full path to the hydra configuration file. If None, the default configuration will be used.
                                               Mutually exclusive with feature_extractors.
        :param feature_extractors:             One or more feature extractors to use. If None, the default configuration will be used.
                                               Mutually exclusive with config_path.
        :param log_dir:                        Directory where to save the logs. By default, uses the current working directory.
        :param use_cache:                      Whether to use cache or not for the configuration of the data.
        :param class_names_to_use:             List of class names that we should use for analysis.
        :param batches_early_stop:             Maximum number of batches to run in training (early stop).
        :param remove_plots_after_report:      Delete the plots from the report directory after the report is generated. By default, True.

        :return:                               An instance of the semantic-segmentation manager configured for the specified
                                               COCO-formatted dataset.
        """

        train_data = COCOSegmentationDataset(root_dir=root_dir, split="train", year=year)
        val_data = COCOSegmentationDataset(root_dir=root_dir, split="val", year=year)

        train_loader = DataLoader(train_data, num_workers=8, batch_size=1)
        val_loader = DataLoader(val_data, num_workers=8, batch_size=1)

        cls(
            train_data=train_loader,
            val_data=val_loader,
            #
            report_title=report_title,
            report_subtitle=report_subtitle,
            config_path=config_path,
            feature_extractors=feature_extractors,
            log_dir=log_dir,
            use_cache=use_cache,
            #
            class_names=train_data.class_names,
            class_names_to_use=class_names_to_use,
            #
            batches_early_stop=batches_early_stop,
            remove_plots_after_report=remove_plots_after_report,
        ).run()

    @classmethod
    def analyze_coco_format(
        cls,
        *,
        # DATA
        root_dir: PathLike,
        train_images_subdir: str,
        train_annotation_file_path: PathLike,
        val_images_subdir: str,
        val_annotation_file_path: PathLike,
        # Report
        report_title: str,
        report_subtitle: str | None = None,
        config_path: PathLike | None = None,
        feature_extractors: FeatureExtractorsType | None = None,
        log_dir: PathLike | None = None,
        use_cache: bool = False,
        class_names_to_use: ClassNamesToUseType | None = None,
        batches_early_stop: int | None = None,
        remove_plots_after_report: bool | None = True,
    ):
        """
        Class method to create a semantic-segmentation manager instance from data in custom COCO format.

        :param root_dir:                      Root directory containing the COCO formatted dataset.
        :param train_images_subdir:           Subdirectory containing the training images.
        :param train_annotation_file_path:    Path to the training annotation file.
        :param val_images_subdir:             Subdirectory containing the validation images.
        :param val_annotation_file_path:      Path to the validation annotation file.
        :param report_title:                  Title of the report. Will be used to save the report.
        :param report_subtitle:               Subtitle of the report.
        :param config_path:                   Full path to the hydra configuration file. If None, the default configuration will be used.
                                              Mutually exclusive with feature_extractors.
        :param feature_extractors:            One or more feature extractors to use. If None, the default configuration will be used.
                                              Mutually exclusive with config_path.
        :param log_dir:                       Directory where to save the logs. By default, uses the current working directory.
        :param use_cache:                     Whether to use cache or not for the configuration of the data.
        :param class_names_to_use:            List of class names that we should use for analysis.
        :param batches_early_stop:            Maximum number of batches to run in training (early stop).
        :param remove_plots_after_report:     Delete the plots from the report directory after the report is generated. By default, True.

        :return:                              An instance of the semantic-segmentation manager configured for the specified
                                              custom COCO-formatted dataset.
        """

        train_data = COCOFormatSegmentationDataset(
            root_dir=root_dir,
            images_subdir=train_images_subdir,
            annotation_file_path=train_annotation_file_path,
        )
        val_data = COCOFormatSegmentationDataset(
            root_dir=root_dir,
            images_subdir=val_images_subdir,
            annotation_file_path=val_annotation_file_path,
        )

        cls(
            train_data=train_data,
            val_data=val_data,
            #
            report_title=report_title,
            report_subtitle=report_subtitle,
            config_path=config_path,
            feature_extractors=feature_extractors,
            log_dir=log_dir,
            use_cache=use_cache,
            #
            class_names=train_data.class_names,
            class_names_to_use=class_names_to_use,
            #
            batches_early_stop=batches_early_stop,
            remove_plots_after_report=remove_plots_after_report,
        ).run()

    @classmethod
    def analyze_voc(
        cls,
        *,
        root_dir: PathLike,
        year: str | int,
        report_title: str,
        report_subtitle: str | None = None,
        config_path: PathLike | None = None,
        feature_extractors: FeatureExtractorsType | None = None,
        log_dir: PathLike | None = None,
        use_cache: bool = False,
        class_names_to_use: ClassNamesToUseType | None = None,
        batches_early_stop: int | None = None,
        remove_plots_after_report: bool | None = True,
    ):
        """
        Class method to create a semantic-segmentation manager instance from data in VOC format.

        :param root_dir:                  Root directory containing the VOC formatted dataset.
        :param year:                      Dataset release year or specific identifier for the VOC dataset.
        :param report_title:              Title of the report. Will be used to save the report.
        :param report_subtitle:           Subtitle of the report.
        :param config_path:               Full path to the hydra configuration file. If None, the default configuration will be used.
                                          Mutually exclusive with feature_extractors.
        :param feature_extractors:        One or more feature extractors to use. If None, the default configuration will be used.
                                          Mutually exclusive with config_path.
        :param log_dir:                   Directory where to save the logs. By default, uses the current working directory.
        :param use_cache:                 Whether to use cache or not for the configuration of the data.
        :param class_names_to_use:        List of class names that we should use for analysis.
        :param batches_early_stop:        Maximum number of batches to run in training (early stop).
        :param remove_plots_after_report: Delete the plots from the report directory after the report is generated. By default, True.

        :return:                          An instance of the semantic-segmentation manager configured for the specified VOC-formatted dataset.
        """

        train_data = VOCSegmentationDataset(root_dir=root_dir, split="train", year=year)
        val_data = VOCSegmentationDataset(root_dir=root_dir, split="val", year=year)

        cls(
            train_data=train_data,
            val_data=val_data,
            #
            report_title=report_title,
            report_subtitle=report_subtitle,
            config_path=config_path,
            feature_extractors=feature_extractors,
            log_dir=log_dir,
            use_cache=use_cache,
            #
            class_names=train_data.class_names,
            class_names_to_use=class_names_to_use,
            #
            batches_early_stop=batches_early_stop,
            remove_plots_after_report=remove_plots_after_report,
        ).run()


###
