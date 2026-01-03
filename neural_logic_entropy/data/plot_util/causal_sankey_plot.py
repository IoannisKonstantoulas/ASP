import plotly.graph_objects as go

from neural_logic_entropy.util.trace.causal_core import causal_explanation_onehidden


def plot_causal_sankey(
        input_row,
        layer_weights,
        output_idx,
        threshold=1e-4,
        title=None,
        save_path=None
):
    core = causal_explanation_onehidden(input_row, layer_weights)

    input_hidden = core["input_to_hidden"]  # (20, 21)
    hidden_output = core["hidden_to_output"]  # (21, 21)

    labels = []
    index = {}

    def add(name):
        if name not in index:
            index[name] = len(labels)
            labels.append(name)
        return index[name]

    sources = []
    targets = []
    values = []
    colors = []

    # ----- Input → Hidden -----
    for i in range(input_hidden.shape[0]):
        for h in range(input_hidden.shape[1]):
            v = input_hidden[i, h]
            if abs(v) > threshold:
                s = add(f"I{i}")
                t = add(f"H{h}")
                sources.append(s)
                targets.append(t)
                values.append(abs(v))
                colors.append("rgba(0,120,255,0.5)" if v > 0 else "rgba(255,50,50,0.5)")

    # ----- Hidden → Output -----
    for h in range(hidden_output.shape[0]):
        v = hidden_output[h, output_idx]
        if abs(v) > threshold:
            s = add(f"H{h}")
            t = add(f"O{output_idx}")
            sources.append(s)
            targets.append(t)
            values.append(abs(v))
            colors.append("rgba(0,180,0,0.6)" if v > 0 else "rgba(200,0,0,0.6)")

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=18,
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )
    )

    fig.update_layout(
        title=title or f"Causal Sankey → Output {output_idx}",
        font_size=11
    )

    if save_path:
        fig.write_html(save_path)

    return fig
