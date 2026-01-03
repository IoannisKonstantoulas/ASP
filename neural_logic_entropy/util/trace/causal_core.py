import numpy as np


def causal_explanation_onehidden(input_row, layer_weights):

    W1 = layer_weights[0]["W"]  # (20, 21)
    b1 = layer_weights[0]["b"]  # (21,)
    W2 = layer_weights[1]["W"]  # (21, 21)
    b2 = layer_weights[1]["b"]  # (21,)

    # ---------- Forward pass ----------
    z1 = input_row @ W1 + b1           # (21,)
    a1 = np.maximum(0.0, z1)           # ReLU

    z2 = a1 @ W2 + b2                  # (21,)
    y_hat = 1.0 / (1.0 + np.exp(-z2))  # Sigmoid

    # ---------- Causal flows ----------

    # Input → Hidden (each input's contribution to each hidden neuron)
    input_to_hidden = input_row[:, None] * W1  # (20, 21)

    # ReLU gate: if hidden neuron inactive, all its incoming contributions are zero
    relu_gate = (z1 > 0).astype(float)         # (21,)
    input_to_hidden *= relu_gate[None, :]      # column-wise gating

    # Hidden → Output (each hidden neuron's contribution to each output)
    hidden_to_output = a1[:, None] * W2         # (21, 21)

    return {
        # core causal tensors
        "input_to_hidden": input_to_hidden,     # (20, 21)
        "hidden_to_output": hidden_to_output,   # (21, 21)

        # activations
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "y_hat": y_hat,
    }
