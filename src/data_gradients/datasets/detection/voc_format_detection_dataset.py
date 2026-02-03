import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from data_gradients.dataset_adapters.config.typing_utils import ClassNamesType
from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import PathLike

logger = logging.getLogger(__name__)


class VOCFormatDetectionDataset(BaseImageLabelDirectoryDataset):
    """The VOC format Detection Dataset supports datasets where labels are stored in XML following according to VOC standard.

    #### Expected folder structure
    Any structure including at least one sub-directory for images and one for xml labels. They can be the same.

    Example 1: Separate directories for images and labels
    ```
        dataset_root/
            ├── images/
            │   ├── train/
            │   │   ├── 1.jpg
            │   │   ├── 2.jpg
            │   │   └── ...
            │   ├── test/
            │   │   ├── ...
            │   └── validation/
            │       ├── ...
            └── labels/
                ├── train/
                │   ├── 1.xml
                │   ├── 2.xml
                │   └── ...
                ├── test/
                │   ├── ...
                └── validation/
                    ├── ...
    ```

    Example 2: Same directory for images and labels
    ```
        dataset_root/
            ├── train.txt
            ├── validation.txt
            ├── train/
            │   ├── 1.jpg
            │   ├── 1.xml
            │   ├── 2.jpg
            │   ├── 2.xml
            │   └── ...
            └── validation/
                ├── ...
    ```

    **Note**: The label file need to be stored in XML format, but the file extension can be different.

    #### Expected label files structure
    The label files must be structured in XML format, like in the following example:

    ``` xml
    <annotation>
        <object>
            <name>chair</name>
            <bndbox>
                <xmin>1</xmin>
                <ymin>213</ymin>
                <xmax>263</xmax>
                <ymax>375</ymax>
            </bndbox>
        </object>
        <object>
            <name>sofa</name>
            <bndbox>
                <xmin>104</xmin>
                <ymin>151</ymin>
                <xmax>334</xmax>
                <ymax>287</ymax>
            </bndbox>
        </object>
    </annotation>
    ```

    The (optional) config file should include the list image ids to include.
    ```
    1
    5
    6
    ...
    34122
    ```
    The associated images/labels will then be loaded from the images_subdir and labels_subdir.
    If config_path is not provided, all images will be used.

    #### Instantiation
    ```
    dataset_root/
        ├── train.txt
        ├── validation.txt
        ├── images/
        │   ├── train/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── test/
        │   │   ├── ...
        │   └── validation/
        │       ├── ...
        └── labels/
            ├── train/
            │   ├── 1.txt
            │   ├── 2.txt
            │   └── ...
            ├── test/
            │   ├── ...
            └── validation/
                ├── ...
    ```


    ```python
    from data_gradients.datasets.detection import VOCFormatDetectionDataset

    train_set = VOCFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/train", labels_subdir="labels/train", config_path="train.txt"
    )
    val_set = VOCFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/validation", labels_subdir="labels/validation", config_path="validation.txt"
    )
    ```
    """

    def __init__(
        self,
        root_dir: PathLike,
        images_subdir: PathLike,
        labels_subdir: PathLike,
        class_names: ClassNamesType,
        config_path: PathLike | None = None,
        verbose: bool = False,
        image_extensions: Iterable[str] = DEFAULT_IMG_EXTENSIONS,
        label_extensions: Iterable[str] = ("xml",),
    ):
        """
        :param root_dir:            Where the data is stored.
        :param images_subdir:       Local path to directory that includes all the images. Path relative to `root_dir`.
            Can be the same as `labels_subdir`.
        :param labels_subdir:       Local path to directory that includes all the labels. Path relative to `root_dir`.
            Can be the same as `images_subdir`.
        :param class_names:         List of class names. This is required to be able to parse the class names into class ids.
        :param config_path:         Path to an optional config file. This config file should contain the list of file ids to include.
                                    If None, all the available images and tagets will be loaded.
        :param verbose:             Whether to show extra information during loading.
        :param image_extensions:    List of image file extensions to load from.
        :param label_extensions:    List of label file extensions to load from.
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=images_subdir,
            labels_subdir=labels_subdir,
            config_path=config_path,
            verbose=verbose,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
        )
        self.class_names = class_names

    def load_labels(self, path: PathLike) -> np.ndarray:
        with Path(path).open(encoding="utf-8") as f:
            xml_parser = ElementTree.parse(f).getroot()

        labels = []
        for obj in xml_parser.iter("object"):
            obj_name = obj.find("name")
            class_name = obj_name.text if obj_name is not None else ""
            xml_box = obj.find("bndbox")
            obj_difficult = obj.find("difficult")

            # TODO: understand if we want difficult!=1 or not
            if class_name is None:
                continue

            if class_name in self.class_names and (obj_difficult is None or obj_difficult.text != "1") and xml_box is not None:
                if isinstance(self.class_names, list | set | tuple):
                    class_id = self.class_names.index(class_name)
                elif isinstance(self.class_names, Mapping):
                    class_id = next((k for k, v in self.class_names.items() if v == class_name), -1)
                    if class_id == -1:
                        continue
                elif hasattr(self.class_names, "numpy"):  # npt.NDArray
                    class_names_array = self.class_names
                    class_id = int(np.where(class_names_array == class_name)[0][0])  # type: ignore[attr-defined]

                elif hasattr(self.class_names, "nonzero"):  # torch.Tensor
                    class_names_tensor = self.class_names
                    class_id = int((class_names_tensor == class_name).nonzero(as_tuple=True)[0][0])  # type: ignore[attr-defined]

                obj_labels = [
                    xml_box.find("xmin"),
                    xml_box.find("ymin"),
                    xml_box.find("xmax"),
                    xml_box.find("ymax"),
                ]

                if any(obj is None for obj in obj_labels):
                    continue

                labels.append(
                    [class_id, float(obj_labels[0].text), float(obj_labels[1].text), float(obj_labels[2].text), float(obj_labels[3].text)]  # type: ignore[ArgType]
                )

        return np.array(labels, dtype=float) if labels else np.zeros((0, 5), dtype=float)
