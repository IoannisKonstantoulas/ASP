from neural_logic_entropy.util.trace.causal_core import causal_explanation_onehidden


def build_sankey(input_row, layer_weights, output_idx, threshold=1e-5):
    core = causal_explanation_onehidden(input_row, layer_weights)

    links = []

    # Input → Hidden
    for i in range(20):
        for h in range(21):
            v = core["input_hidden"][i, h]
            if abs(v) > threshold:
                links.append(("I" + str(i), "H" + str(h), v))

    # Hidden → Output
    for h in range(21):
        v = core["hidden_output"][h, output_idx]
        if abs(v) > threshold:
            links.append(("H" + str(h), "O" + str(output_idx), v))

    return links
