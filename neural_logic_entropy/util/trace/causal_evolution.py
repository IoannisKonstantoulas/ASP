import numpy as np

from neural_logic_entropy.util.trace.causal_core import causal_explanation_onehidden
from neural_logic_entropy.util.inspect_utils import unpack_weights


def causal_entropy(contrib):
    p = np.abs(contrib)
    p /= (p.sum() + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))


def causal_evolution(
    input_row,
    weights_mm,
    model,
    output_idx
):
    history = []

    for step in range(weights_mm.shape[0]):
        layer_weights = unpack_weights(weights_mm[step], model)
        core = causal_explanation_onehidden(input_row, layer_weights)

        contrib = core["input_output"][:, output_idx]

        history.append({
            "step": step,
            "entropy": causal_entropy(contrib),
            "active_hidden": (core["a1"] > 0).sum(),
            "top_input": int(np.argmax(np.abs(contrib))),
            "logit": core["z2"][output_idx]
        })

    return history
