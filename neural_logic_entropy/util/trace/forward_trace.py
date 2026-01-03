import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_trace_onehidden(
        input_row,
        layer_weights
):
    # ---- unpack weights ----
    W1 = layer_weights[0]["W"]  # (20, 21)
    b1 = layer_weights[0]["b"]  # (21,)
    W2 = layer_weights[1]["W"]  # (21, 21)
    b2 = layer_weights[1]["b"]  # (21,)

    # ---- hidden layer ----
    z_hidden = input_row @ W1 + b1  # (21,)
    a_hidden = np.maximum(0.0, z_hidden)  # ReLU

    # ---- output layer ----
    z_out = a_hidden @ W2 + b2  # (21,)
    y_hat = sigmoid(z_out)  # (21,)

    # ---- routed contributions ----

    # input → hidden contribution per (input_i → hidden_j)
    # shape: (20, 21)
    input_hidden_contrib = input_row[:, None] * W1

    # hidden → output contribution per (hidden_j → output_k)
    # shape: (21, 21)
    hidden_output_contrib = a_hidden[:, None] * W2

    # contribution of EACH hidden neuron to EACH output
    # (this is the most important object)
    per_hidden_to_output = hidden_output_contrib

    return {
        "input": input_row,

        "hidden_pre": z_hidden,
        "hidden_act": a_hidden,

        "output_pre": z_out,
        "output": y_hat,

        "input_hidden_contrib": input_hidden_contrib,
        "hidden_output_contrib": hidden_output_contrib,
        "per_hidden_to_output": per_hidden_to_output,
    }
