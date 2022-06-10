import io
from PIL import Image
import tensorflow.keras as keras
from . core.custom_objects import custom_objects as registered_custom_objects

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def load_model(path, custom_objects={}):
    objs = registered_custom_objects()
    objs.update(custom_objects)
    return keras.models.load_model(path, objs)

def str_to_bool(s):
    return s.strip().lower() in {'1', 't', 'true', 'y', 'yes'}

def plt_to_image(plt, format="png"):
    buf = io.BytesIO()
    plt.savefig(buf, format=format)
    buf.seek(0)
    return Image.open(buf)
