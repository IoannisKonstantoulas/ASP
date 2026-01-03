import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def get_latest_model_date_string(base_folder=None):
    if base_folder is None:
        base_folder = os.path.join(os.path.dirname(__file__), "..", "model", "models")
    base_folder = os.path.abspath(base_folder)

    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Base folder does not exist: {base_folder}")

    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    if not folders:
        raise FileNotFoundError(f"No model folders found in {base_folder}")

    folders.sort()
    latest_folder = folders[-1]  # get the last (latest) folder
    return latest_folder

def save_heatmap(matrix, title, filename, xlabel=None, ylabel=None, cmap='viridis'):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect='auto', cmap=cmap)
    plt.colorbar()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def get_layer_shapes(model):
    layer_info = []
    for layer in model.layers:
        if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
            layer_info.append({
                'W_shape': layer.kernel.shape,
                'b_shape': layer.bias.shape
            })
    return layer_info


def unpack_weights(weights_step, model):
    layer_info = get_layer_shapes(model)
    unpacked = []
    idx = 0
    for info in layer_info:
        W_size = int(np.prod(info['W_shape']))
        b_size = int(np.prod(info['b_shape']))
        W = weights_step[idx:idx + W_size].reshape(info['W_shape'])
        idx += W_size
        b = weights_step[idx:idx + b_size].reshape(info['b_shape'])
        idx += b_size
        unpacked.append({'W': W, 'b': b})
    return unpacked
