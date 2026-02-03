from collections.abc import Sequence

from data_gradients.common.factories.base_factory import AbstractFactory, ConfigType


class ListFactory(AbstractFactory):
    def __init__(self, factory: AbstractFactory):
        self.factory = factory

    def get(self, conf: ConfigType):
        if isinstance(conf, Sequence):
            return [self.factory.get(c) for c in conf]

        return self.factory.get(conf)

    #
    #
