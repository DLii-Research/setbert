"""
A Tensorflow implementation of sef attention attribution: https://arxiv.org/abs/2004.11207
"""

from collections import defaultdict
from graphviz import Digraph
from lmdbm import Lmdb
from numba import njit
import numpy as np
from pathlib import Path
import pickle
import settransformer as st
import tensorflow as tf
import time
from tqdm import tqdm, trange
from typing import Any, Callable, Iterable


def find_mha_layers(model) -> list[tf.keras.layers.Layer]:
    """
    Find all multi-head attention layers within a model.
    """
    result = []
    for layer in model.layers:
        if isinstance(layer, st.SetAttentionBlock):
            result.append(layer.att)
        elif isinstance(layer, st.InducedSetAttentionBlock):
            result.append(layer.mab2.att)
        elif isinstance(layer, st.InducedSetEncoder):
            result.append(layer.mab.att)
        elif isinstance(layer, tf.keras.Model):
            result += find_mha_layers(layer)
    return result


def attention_attribution(
    output_path: str|Path,
    data_generator: Iterable[Any],
    call: callable[Any, tf.Tensor],
    mha_layers: Iterable[tf.keras.layers.Layer],
    integration_steps=20
) -> None:
    """
    Compute self-attention attribution.

    Arguments:
      output_path: The output DB path to store the results.
      data_generator: an iterable that returns a tuple containing metadata and the data.
      call: a callable that accepts a data item as input, passing it through the model,
            and returning the attention scores.
      mha_layers: The MHA layers to monitor.
      integration_steps: The number of integration approximation steps to perform.
    """
    mha_layers = tuple(mha_layers)
    # Create the compute_attention_attribution function to gather attribution scores for each layer/head
    if not hasattr(attention_attribution, "call") or id(attention_attribution.call) != id(call):
        print("Compiling execution graph... This may take several minutes.")
        attention_attribution.call = call
        attention_attribution.compute_attention_attribution = _compute_attention_attribution_factory(call)
    compute_attention_attribution = attention_attribution.compute_attention_attribution

    store = Lmdb.open(output_path, "n")

    i = -1
    for i, (metadata, x) in enumerate(data_generator):
        print(f"\rComputing attention attribution: Step {i+1}", end="")
        # Ensure token names have a corresponding ID
        attrs = compute_attention_attribution(
            tf.expand_dims(x, axis=0),
            mha_layers,
            integration_steps)[0]

        store[f"{i}_x"] = pickle.dumps(x)
        store[f"{i}_metadata"] = pickle.dumps(metadata)
        store[f"{i}_attr_shape"] = pickle.dumps(attrs.shape)
        store[f"{i}_attrs"] = np.array(attrs).tobytes()

    store["length"] = i + 1
    store.close()


def token_attribution(attribution_path: str|Path, metadata_key = lambda x: x, tau=0.4):
    """
    Compute token attribution scores and build the attribution graph.
    """
    store = Lmdb.open(attribution_path)
    length = store["length"] if "length" in store else len(store)//4

    token_ids = {}
    for n in range(length):
        token_names = metadata_key(pickle.loads(store[f"{n}_metadata"]))
        for token_name in token_names:
            if token_name not in token_ids:
                token_ids[token_name] = len(token_ids)

    token_total_attrs = defaultdict(float)
    top_node = None
    top_node_value = -np.inf

    taus = [tau]*(pickle.loads(store["0_attr_shape"])[0] - 1) + [0.0]
    edges_by_layer = [set() for _ in range(len(taus))]

    t1 = t2 = 0.0
    for n in trange(length, leave=False):
        print(f"\r{n+1}/{length}   IO: {t1:.3f}s   Total: {t2:.3f}s", end="")
        s = time.time()
        shape = pickle.loads(store[f"{n}_attr_shape"])
        token_names = metadata_key(pickle.loads(store[f"{n}_metadata"]))
        attrs = np.frombuffer(store[f"{n}_attrs"], dtype=np.float32).reshape(shape)
        t1 = time.time() - s

        attrs_by_layer = np.sum(attrs, axis=1)[:,1:,1:] # strip off class tokens
        token_attrs = _compute_token_attributions(attrs_by_layer)

        # Sum up total attribution for each token
        for i, value in enumerate(token_attrs):
            token_total_attrs[token_ids[token_names[i]]] += value

        # Update the top node
        max_value = np.max(token_attrs)
        if max_value > top_node_value:
            top_node = token_ids[token_names[np.argmax(token_attrs)]]
            top_node_value = max_value

        # Create tree adges
        for l, (tau, edges) in enumerate(zip(taus, edges_by_layer)):
            a_ij = attrs_by_layer[l]
            max_attr_l = np.max(attrs_by_layer[l])
            for (i, j) in np.argwhere(a_ij/max_attr_l > tau):
                i_universal = token_ids[token_names[i]]
                j_universal = token_ids[token_names[j]]
                edges.add((i_universal, j_universal))

        t2 = time.time() - s

    assert top_node is not None

    NotAppear, Appear, Fixed = "NotAppear", "Appear", "Fixed"
    state = {id: NotAppear for id in token_ids.values()}
    state[top_node] = Appear
    edges = set()
    vertices = set([top_node])
    for l in range(len(taus) - 2, -1, -1):
        for (i, j) in edges_by_layer[l]:
            if i == j:
                continue
            if state[i] is Appear and state[j] is NotAppear:
                edges.add((i, j))
                vertices.add(j)
                state[i] = Fixed
                state[j] = Appear
            if state[i] is Fixed and state[j] is NotAppear:
                edges.add((i, j))
                vertices.add(j)
                state[j] = Appear

    # Inject class token identifier
    vertices.add(-1)
    for j in range(len(state) - 1, -1, -1):
        if state[j] in (Appear, Fixed):
            edges.add((-1, j))

    # Compute the reversed ID map
    token_id_reverse_map = {v: k for k, v in token_ids.items()}
    token_id_reverse_map[-1] = "[CLS]"

    store.close()

    return {
        "token_id_map": token_id_reverse_map,
        "token_attrs": token_total_attrs,
        "edges": edges,
        "vertices": vertices
    }


def build_attribution_tree(vertices, edges, node_labels):
    """
    Build the attention attribution tree given the vertices/edges
    """
    # Create the tree graphic
    tree = Digraph()
    for vertex in vertices:
        label = node_labels[vertex]
        tree.node(label)
    edge_set = set()
    for (i, j) in edges:
        i = node_labels[i]
        j = node_labels[j]
        if (i, j) in edge_set:
            continue
        edge_set.add((i, j))
        tree.edge(j, i)
    tree.graph_attr["rankdir"] = "BT"
    return tree


@njit
def _compute_token_attributions(attrs_by_layer):
    """
    Compute the attribution scores by token.
    """
    num_tokens = attrs_by_layer.shape[-1]
    attr_all = np.zeros(num_tokens)
    for i in range(num_tokens):
        for l in range(attrs_by_layer.shape[0]):
            for j in range(num_tokens):
                if j == i:
                    continue
                attr_all[i] += attrs_by_layer[l,i,j]
    return attr_all


def _compute_attention_attribution_factory(call):
    if hasattr(call, "__iter__"):
        # attention attribution shift
        call_a, call_b = call
        @tf.function
        def compute_attention_attribution(x, mha_layers, integration_steps=20):
            # Initialize attention attribution weights
            for (layer_a, layer_b) in mha_layers:
                layer_a.enable_attribution()
                layer_b.enable_attribution()
                layer_a._alpha.assign([1.0]*layer_a.num_heads)
                layer_b._alpha.assign([1.0]*layer_b.num_heads)
            # Get the gradient shapes using a single call
            shapes = []
            _, scores = call_a(x)
            for layer_scores in scores: # [batch_index, head_index, n, n]
                batch_size = tf.shape(layer_scores)[0]
                n = tf.shape(layer_scores)[2]
                shapes.append((batch_size, n, n))
            # Compute attribution scores
            attrs = []
            for layer_index, (layer_a, layer_b) in enumerate(mha_layers):
                attrs.append([])
                for head in range(layer_a.num_heads):
                    grad_sum_a = tf.zeros(shapes[layer_index])
                    grad_sum_b = tf.zeros(shapes[layer_index])
                    for alpha in tf.linspace(0.0, 1.0, integration_steps):
                        layer_a.set_head_attribution_weight(head, alpha)
                        layer_b.set_head_attribution_weight(head, alpha)
                        with tf.GradientTape() as tape_a:
                            y_pred_a, scores_a = call_a(x)
                        with tf.GradientTape() as tape_b:
                            y_pred_b, scores_b = call_b(x)
                        grads_a = tape_a.gradient(y_pred_a, scores_a[layer_index])[:,head,:,:]
                        grads_b = tape_b.gradient(y_pred_b, scores_b[layer_index])[:,head,:,:]
                        grad_sum_a += grads_a
                        grad_sum_b += grads_b
                    attr_h_a = scores_a[layer_index][:,head,:,:]/integration_steps * grad_sum_a
                    attr_h_b = scores_b[layer_index][:,head,:,:]/integration_steps * grad_sum_b
                    attrs[-1].append(attr_h_b - attr_h_a)
            attrs = tf.transpose(attrs, (2, 0, 1, 3, 4))
            # attrs.shape == [sample_index, layer_index, head_index, n, n]
            return attrs

    else:
        @tf.function
        def compute_attention_attribution(x, mha_layers, integration_steps=20):
            # Initialize attention attribution weights
            for layer in mha_layers:
                layer.enable_attribution()
                layer._alpha.assign([1.0]*layer.num_heads)
            # Get the gradient shapes using a single call
            shapes = []
            _, scores = call(x)
            for layer_scores in scores: # [batch_index, head_index, n, n]
                batch_size = tf.shape(layer_scores)[0]
                n = tf.shape(layer_scores)[2]
                shapes.append((batch_size, n, n))
            # Compute attribution scores
            attrs = []
            for layer_index, layer in enumerate(mha_layers):
                attrs.append([])
                for head in range(layer.num_heads):
                    grad_sum = tf.zeros(shapes[layer_index])
                    for alpha in tf.linspace(0.0, 1.0, integration_steps):
                        layer.set_head_attribution_weight(head, alpha)
                        with tf.GradientTape() as tape:
                            y_pred, scores = call(x)
                        grads = tape.gradient(y_pred, scores[layer_index])[:,head,:,:]
                        grad_sum += grads
                    attr_h = scores[layer_index][:,head,:,:]/integration_steps * grad_sum
                    attrs[-1].append(attr_h)
            attrs = tf.transpose(attrs, (2, 0, 1, 3, 4))
            # attrs.shape == [sample_index, layer_index, head_index, n, n]
            return attrs

    return compute_attention_attribution
