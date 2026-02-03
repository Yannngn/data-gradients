from ..registry.registry import FEATURE_EXTRACTORS
from .base_factory import BaseFactory


class FeatureExtractorsFactory(BaseFactory):
    def __init__(self):
        super().__init__(FEATURE_EXTRACTORS)
