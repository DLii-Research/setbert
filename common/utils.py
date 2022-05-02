import tensorflow.keras as keras
from . core.custom_objects import custom_objects

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def load_model(path, custom_objs={}):
    objs = custom_objects()
    objs.update(custom_objs)
    return keras.models.load_model(path, objs)

def str_to_bool(s):
    return s.strip().lower() in {'1', 'true', 'y', 'yes'}