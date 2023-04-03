
from pathlib import Path
import tensorflow as tf
from typing import Any, cast

def load_model(
    path: str|Path,
    custom_objects: dict[str, Any]|None = None,
    compile: bool = True,
    options: tf.saved_model.LoadOptions|None = None
) -> tf.keras.Model:
    """
    Load a custom model, providing the necessary custom object layers
    """
    # Import custom layers and models
    from .. import layers, registry
    from . import dnabert, setbert
    objects = registry.custom_objects()
    if custom_objects is not None:
        objects.update(custom_objects)
    return cast(tf.keras.Model,tf.keras.models.load_model(path, custom_objects, compile, options))
