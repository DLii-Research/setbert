
from pathlib import Path
import tensorflow as tf
from typing import Any, cast, TypeVar

# Import layers and model submodules to ensure everything is registered correctly.
from .. import layers, losses, registry
from . import dnabert, setbert, setbert, utils

ModelType = TypeVar("ModelType", bound=tf.keras.Model)

def load_model(
    path: str|Path,
    type: type[ModelType] = tf.keras.Model,
    custom_objects: dict[str, Any]|None = None,
    compile: bool = True,
    options: tf.saved_model.LoadOptions|None = None
) -> ModelType:
    """
    Load a custom model, providing the necessary custom object layers
    """
    objects = registry.custom_objects()
    if custom_objects is not None:
        objects.update(custom_objects)
    return cast(type, tf.keras.models.load_model(path, objects, compile, options))
