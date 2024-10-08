from pathlib import Path
import tensorflow as tf
from typing import Any, cast, Dict, Optional, Type, TypeVar, Union

from . import custom_model

from . import contrastive
from . import dnabert
from . import setbert
from . import taxonomy
from . import transformer

from .. import registry

# Import layers and model submodules to ensure everything is registered correctly.

ModelType = TypeVar("ModelType", bound=tf.keras.Model)

def load_model(
    path: Union[str, Path],
    type: Type[ModelType] = tf.keras.Model,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = True,
    options: Optional[tf.saved_model.LoadOptions] = None
) -> ModelType:
    """
    Load a custom model, providing the necessary custom object layers
    """
    objects = registry.custom_objects()
    if custom_objects is not None:
        objects.update(custom_objects)
    return cast(type, tf.keras.models.load_model(path, objects, compile, options))
