__version__ = "0.3.2"

from .managers.classification_manager import ClassificationAnalysisManager
from .managers.detection_manager import DetectionAnalysisManager

__all__ = ["DetectionAnalysisManager", "ClassificationAnalysisManager"]
