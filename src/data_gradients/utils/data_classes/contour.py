from dataclasses import dataclass

import numpy as np


@dataclass()
class Contour:
    points: np.ndarray
    area: float
    w: float
    h: float
    center: tuple[int, int]
    perimeter: float
    class_id: int
    bbox_area: float
