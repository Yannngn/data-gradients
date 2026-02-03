import inspect
from collections.abc import Callable


def create_register_decorator(registry: dict[str, Callable]) -> Callable:
    """
    Create a decorator that registers object of specified type (model, metric, ...)

    :param registry:    Dict including registered objects (maps name to object that you register)
    :return:            Register function
    """

    def register(name: str | None = None) -> Callable:
        """
        Set up a register decorator.

        :param name: If specified, the decorated object will be registered with this name.
        :return:     Decorator that registers the callable.
        """

        def decorator(cls: Callable) -> Callable:
            """Register the decorated callable"""
            cls_name = name if name is not None else cls.__name__

            if cls_name in registry:
                ref = registry[cls_name]
                the_module = inspect.getmodule(ref)
                module_name = the_module.__name__ if the_module is not None else "unknown module"
                raise Exception(f"`{cls_name}` is already registered and points to `{module_name}.{ref.__name__}")

            registry[cls_name] = cls
            return cls

        return decorator

    return register


FEATURE_EXTRACTORS = {}
register_feature_extractor = create_register_decorator(registry=FEATURE_EXTRACTORS)
