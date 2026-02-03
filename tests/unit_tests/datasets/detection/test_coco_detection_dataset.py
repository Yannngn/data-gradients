import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from data_gradients.datasets.detection.coco_detection_dataset import COCODetectionDataset
from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.utils.data_classes.image_channels import ImageChannels


class COCOSegmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[5]
        self.train_set = COCODetectionDataset(root_dir=project_root / "example_dataset" / "tinycoco", split="train", year="2017")
        self.val_set = COCODetectionDataset(root_dir=project_root / "example_dataset" / "tinycoco", split="val", year="2017")

    def test_coco_dataset(self):
        da = DetectionAnalysisManager(
            report_title="COCO 2017 Detection Report",
            train_data=self.train_set,
            val_data=self.val_set,
            class_names=self.train_set.class_names,
            image_channels=ImageChannels.from_str("rgb"),
            is_label_first=True,
            bbox_format="xywh",
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()

    def test_coco_dataset_batch(self):
        da = DetectionAnalysisManager(
            report_title="COCO 2017 Detection Report Batch",
            train_data=DataLoader(self.train_set, batch_size=1),
            val_data=DataLoader(self.val_set, batch_size=1),
            class_names=self.train_set.class_names,
            image_channels=ImageChannels.from_str("rgb"),
            is_label_first=True,
            bbox_format="xywh",
            batches_early_stop=10,
            use_cache=False,
        )

        da.run()


if __name__ == "__main__":
    unittest.main()
###
###
###
