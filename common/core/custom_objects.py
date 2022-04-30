__custom_objects = {}

def custom_objects():
    """
    Fetch custom layer instances
    """
    return __custom_objects

def register_custom_object(name, obj):
    __custom_objects[name] = obj
    
def register_custom_objects(objs):
    __custom_objects.update(objs)
    
def CustomObject(Object):
    """
    A decorator for keeping track of custom objects
    """
    register_custom_object(Object.__name__, Object)
    return Object