import numpy as np

from neural_logic_entropy.util.inspect_utils import unpack_weights


def discover_neuron_roles(inputs, weights_mm, model):
    roles = {}

    for h in range(21):
        activations = []
        preferred_inputs = []

        for step in range(weights_mm.shape[0]):
            layer_weights = unpack_weights(weights_mm[step], model)
            W1 = layer_weights[0]["W"]
            b1 = layer_weights[0]["b"]

            z = inputs @ W1[:, h] + b1[h]
            active = z > 0

            activations.append(active.mean())

            if active.any():
                preferred_inputs.append(inputs[active].mean(axis=0))

        roles[h] = {
            "activation_rate": float(np.mean(activations)),
            "stability": float(np.std(activations)),
            "input_signature": (
                np.mean(preferred_inputs, axis=0)
                if preferred_inputs else None
            )
        }

    return roles
