import random
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from data_gradients.managers.classification_manager import ClassificationAnalysisManager


class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EndToEndTest(unittest.TestCase):
    """ """

    def test_classification_task(self):
        class_names = ["class_1", "class_2", "class_3", "class_4"]

        train_samples = []
        for _ in range(100):
            dummy_image = torch.randn((3, random.randint(100, 500), random.randint(100, 500)), dtype=torch.float32)
            train_samples += [(dummy_image, 0)]

        for _ in range(100):
            dummy_image = torch.randn((3, random.randint(300, 600), random.randint(200, 300)), dtype=torch.float32)
            train_samples += [(dummy_image, 1)]

        for _ in range(100):
            dummy_image = torch.randn((3, random.randint(100, 200), random.randint(700, 800)), dtype=torch.float32)
            train_samples += [(dummy_image, 2)]

        valid_samples = []
        for _ in range(100):
            dummy_image = torch.randn((3, random.randint(200, 250), random.randint(200, 250)), dtype=torch.float32)
            valid_samples += [(dummy_image, 3)]

        for _ in range(100):
            dummy_image = torch.randn((220, 230, 3), dtype=torch.float32)
            valid_samples += [(dummy_image, 0)]

        manager = ClassificationAnalysisManager(
            train_data=DataLoader(ImageDataset(train_samples)),
            val_data=DataLoader(ImageDataset(valid_samples)),
            report_title="End to End Classification Test",
            class_names=class_names,
            batches_early_stop=None,
        )

        manager.run()


if __name__ == "__main__":
    unittest.main()
