from collections.abc import Iterable
from pathlib import Path

from data_gradients.config.utils import get_grouped_feature_extractors
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig, get_default_cache_dir
from data_gradients.dataset_adapters.config.typing_utils import ClassNamesType, ExtractorType, SupportedDataType
from data_gradients.dataset_adapters.formatters.utils import ImageFormat
from data_gradients.feature_extractors import FeatureExtractorsType
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.sample_preprocessor.classification_sample_preprocessor import ClassificationSamplePreprocessor
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.utils.summary_writer import SummaryWriter


class ClassificationAnalysisManager(AnalysisManagerAbstract):
    """Implementation of analysys manager for image classification task.
    Definition of task name, task-related preprocessor and parsing related configuration file
    """

    def __init__(
        self,
        *,
        report_title: str,
        train_data: Iterable[SupportedDataType],
        val_data: Iterable[SupportedDataType] | None = None,
        report_subtitle: str | None = None,
        config_path: str | Path | None = None,
        feature_extractors: FeatureExtractorsType | None = None,
        log_dir: str | Path | None = None,
        use_cache: bool = False,
        class_names: ClassNamesType | None = None,
        n_classes: int | None = None,
        images_extractor: ExtractorType | None = None,
        labels_extractor: ExtractorType | None = None,
        is_batch: bool | None = None,
        image_channels: ImageChannels | None = None,
        image_format: ImageFormat | None = None,
        batches_early_stop: int | None = None,
        remove_plots_after_report: bool | None = True,
    ):
        """
        Constructor of detection manager which controls the analyzer
        :param report_title:            Title of the report. Will be used to save the report
        :param report_subtitle:         Subtitle of the report
        :param class_names:             Either the list of all class names in the dataset OR dictionary mapping of `class_id` -> `class_name`.
                                        The index should represent the class_id. Mutually exclusive with `n_classes`
        :param n_classes:               Number of classes. Mutually exclusive with `class_names`.
        :param train_data:              Iterable object contains images and labels of the training dataset
        :param val_data:                Iterable object contains images and labels of the validation dataset
        :param config_path:             Full path the hydra configuration file. If None, the default configuration will be used.
                        Mutually exclusive with feature_extractors
        :param feature_extractors:      One or more feature extractors to use. If None, the default configuration will be used.
                        Mutually exclusive with config_path
        :param log_dir:                 Directory where to save the logs. By default uses the current working directory
        :param batches_early_stop:      Maximum number of batches to run in training (early stop)
        :param use_cache:               Whether to use cache or not for the configuration of the data.
        :param image_format:            Image format to use. Can be Uint8ImageFormat, FloatImageFormat, ScaledFloatImageFormat.
        :param images_extractor:        Function extracting the image(s) out of the data output.
        :param labels_extractor:        Function extracting the label(s) out of the data output.
        :param image_channels:          Image channels to use.
        :param remove_plots_after_report:  Delete the plots from the report directory after the report is generated. By default, True
        """

        if feature_extractors is not None and config_path is not None:
            raise RuntimeError("`feature_extractors` and `config_path` cannot be specified at the same time")

        summary_writer = SummaryWriter(report_title=report_title, report_subtitle=report_subtitle, log_dir=log_dir)
        cache_path = Path(get_default_cache_dir()) / f"{summary_writer.run_name}.json" if use_cache else None
        data_config = ClassificationDataConfig(
            cache_path=cache_path,
            n_classes=n_classes,
            class_names=class_names,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            is_batch=is_batch,
            image_channels=image_channels,
            image_format=image_format,
        )

        sample_preprocessor = ClassificationSamplePreprocessor(data_config=data_config)
        grouped_feature_extractors = get_grouped_feature_extractors(
            default_config_name="classification", feature_extractors=feature_extractors, config_path=config_path
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
