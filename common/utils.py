import io
from PIL import Image
import tensorflow as tf
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

def accumulate(a, b):
    if type(a) in (list, tuple):
        return [accumulate(x, y) for x, y in zip(a, b)]
    return a + b

def subbatch_predict(model, batch, subbatch_size, concat=lambda old, new: tf.concat((old, new), axis=0)):
    def predict(i, result=None):
        n = i + subbatch_size
        pred = tf.stop_gradient(model(batch[i:n]))
        if result is None:
            return [n, pred]
        return [n, concat(result, pred)]
    
    # First pass to obtain initial results
    i, result = predict(0)
    
    batch_size = tf.shape(batch)[0]
    i, result = tf.while_loop(
        cond=lambda i, _: i < batch_size,
        body=predict,
        loop_vars=[i, result],
        parallel_iterations=1)
    
    return result

def accumulate_train_step(train_step, batch, subbatch_size, models):
    """
    A generic implementation of accumulating/gradients via sub-batching.
    
    train_step should return a list of variables, followed by a list of gradients
    """
    def step(i, accum_variables=None, accum_grads=None):
        n = i + subbatch_size
        subbatch = (batch[0][i:n], batch[1][i:n])
        
        # Perform training step
        variables, grads = train_step(subbatch)
        
        # Accumulate variables and gradients
        if accum_variables is not None:
            variables = accumulate(variables, accum_variables)
        grads = [[(g + ag) for g, ag in zip(gs, ags)] for gs, ags in zip(grads, accum_grads)]    
        return [n, variables, grads]
    
    if type(models) not in (list, tuple):
        models = (models,)
    
    # First pass to obtain variables/gradients
    accum_grads = [[tf.zeros_like(w) for w in model.trainable_weights] for model in models]
    i, variables, accum_grads = step(0, None, accum_grads)
    
    # Loop for any additional updates
    batch_size = tf.shape(batch[0])[0]
    i, variables, accum_grads = tf.while_loop(
        cond=lambda i, *_: i < batch_size,
        body=step,
        loop_vars=[i, variables, accum_grads],
        parallel_iterations=1)
    
    return variables, accum_grads