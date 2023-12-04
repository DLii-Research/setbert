from typing import Any, Dict

__custom_objects: Dict[str, Any] = {}

def custom_objects():
    """
    Fetch custom object instances.
    """
    return __custom_objects.copy()

def register_custom_object(name: str, obj: Any):
    """
    Register a single object.
    """
    __custom_objects[name] = obj

def register_custom_objects(objs: Dict[str, Any]):
    """
    Register several objects at once.
    """
    __custom_objects.update(objs)

def CustomObject(Object):
    """
    A decorator for keeping track of custom objects.
    """
    register_custom_object(Object.__name__, Object)
    return Object
