import os
import numpy as np
import plotly.graph_objects as go

from neural_logic_entropy.model.model_definition import OneHiddenDense, OneHiddenDenseWithHidden
from neural_logic_entropy.util.inspect_utils import (
    unpack_weights,
    get_latest_model_date_string
)

# ============================================================
# ========================== CONFIG ==========================
# ============================================================

STEP_START = 800000
STEP_END = 810000

UNSAT_THRESHOLD = 0.2        # <<< ONLY include frames where unsatisfiable < X

LINK_SCALE = 5
NODE_COLOR_SCALE = 0.8
MIN_NODE_ALPHA = 0.05
MIN_LINK = 1e-6
MAX_LINK_THICKNESS = 10

NODE_PADDING = 0.5           # spacing between nodes within a layer
LAYER_MARGIN = 0.02          # top/bottom margin per layer

# ============================================================
# ===================== ATOMS / LABELS =======================
# ============================================================

atoms = [
    "man(a)", "not_man(a)", "woman(a)", "not_woman(a)",
    "man(b)", "not_man(b)", "woman(b)", "not_woman(b)",
    "parent(a,b)", "not_parent(a,b)", "parent(b,a)", "not_parent(b,a)",
    "father(a,b)", "not_father(a,b)", "father(b,a)", "not_father(b,a)",
    "mother(a,b)", "not_mother(a,b)", "mother(b,a)", "not_mother(b,a)"
]

outputs = atoms.copy() + ["unsatisfiable"]
UNSAT_INDEX = outputs.index("unsatisfiable")

# ============================================================
# ====================== MODEL PATHS =========================
# ============================================================

DATE_STRING = get_latest_model_date_string()

PLOTS_FOLDER = os.path.join("plots", DATE_STRING)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

current_folder = os.path.dirname(os.path.abspath(__file__))

MEMMAP_PATH_WEIGHTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "weights.memmap")
)
MEMMAP_PATH_INPUTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "inputs.memmap")
)
MEMMAP_PATH_OUTPUTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "outputs.memmap")
)

# ============================================================
# ====================== LOAD MEMMAPS ========================
# ============================================================

model = OneHiddenDenseWithHidden(input_size=20, hidden_size=10, output_size=21)

total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
dtype_size = np.dtype(np.float32).itemsize
file_size = os.path.getsize(MEMMAP_PATH_WEIGHTS)

num_steps_saved = file_size // (total_params * dtype_size)
STEP_END = min(STEP_END, num_steps_saved)

weights_mm = np.array(
    np.memmap(
        MEMMAP_PATH_WEIGHTS,
        dtype=np.float32,
        mode="r",
        shape=(num_steps_saved, total_params)
    )[STEP_START:STEP_END]
)

inputs_mm = np.array(
    np.memmap(
        MEMMAP_PATH_INPUTS,
        dtype=np.float32,
        mode="r",
        shape=(num_steps_saved, 20)
    )[STEP_START:STEP_END]
)

outputs_mm = np.array(
    np.memmap(
        MEMMAP_PATH_OUTPUTS,
        dtype=np.float32,
        mode="r",
        shape=(num_steps_saved, 21)
    )[STEP_START:STEP_END]
)

# ============================================================
# ===================== OUTPUT SELECTION =====================
# ============================================================

selected_outputs = outputs.copy()  # default: all outputs
selected_output_indices = [outputs.index(o) for o in selected_outputs]

# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def get_neuron_roles(hidden_W, hidden_b, top_k=3):
    hidden_activations = np.maximum(0, np.eye(hidden_W.shape[0]) @ hidden_W + hidden_b)
    hidden_activations = hidden_activations.T
    return {
        h: np.argsort(-hidden_activations[h])[:top_k]
        for h in range(hidden_W.shape[1])
    }
def compute_sankey_data(input_row, layer_weights, neuron_roles):
    hidden_W, hidden_b = layer_weights[0]["W"], layer_weights[0]["b"]
    output_W, output_b = layer_weights[1]["W"], layer_weights[1]["b"]

    # Pre-activation and activation for hidden layer
    hidden_input = input_row @ hidden_W + hidden_b        # pre-ReLU
    hidden_activ = np.maximum(0, hidden_input)           # ReLU

    # Pre-activation for output layer
    output_input = hidden_activ @ output_W + output_b

    # Numerically stable sigmoid for output layer
    from scipy.special import expit
    output_activ = expit(output_input)

    sources, targets, values, labels, link_colors = [], [], [], [], []
    num_hidden = hidden_W.shape[1]

    # ---------------- Input -> Hidden -------------------
    for h in range(num_hidden):
        # Avoid division by zero if hidden neuron is inactive
        total_input_h = hidden_input[h] if hidden_input[h] != 0 else 1e-6

        for i, val in enumerate(input_row):
            # Contribution of input i to hidden h
            contrib = (val * hidden_W[i, h]) / total_input_h * hidden_activ[h]

            sources.append(i)
            targets.append(len(atoms) + h)
            values.append(max(abs(contrib), MIN_LINK))

            top_inputs = [atoms[idx] for idx in neuron_roles[h]]
            labels.append(
                f"{atoms[i]} → h{h} | contrib={contrib:.2f}, b={hidden_b[h]:+.2f}, top:{top_inputs}"
            )

            alpha = min(1.0, max(MIN_NODE_ALPHA, abs(contrib) / LINK_SCALE))
            link_colors.append(
                f"rgba({255 if contrib > 0 else 0},0,{255 if contrib < 0 else 0},{alpha:.3f})"
            )

    # ---------------- Hidden -> Output ------------------
    num_outputs_selected = len(selected_output_indices)
    for o_idx, o in enumerate(selected_output_indices):
        # Total input to output neuron for proportion scaling
        total_hidden_input_o = output_input[o] if output_input[o] != 0 else 1e-6

        for h in range(num_hidden):
            # Contribution of hidden neuron h to output o
            contrib = (hidden_activ[h] * output_W[h, o]) / total_hidden_input_o * output_activ[o_idx]

            sources.append(len(atoms) + h)
            targets.append(len(atoms) + num_hidden + o_idx)
            values.append(max(abs(contrib), MIN_LINK))

            labels.append(
                f"h{h} → {outputs[o]} | contrib={contrib:.2f}, b={output_b[o]:+.2f}"
            )

            alpha = min(1.0, max(MIN_NODE_ALPHA, abs(contrib) / LINK_SCALE))
            link_colors.append(
                f"rgba({255 if contrib > 0 else 0},0,{255 if contrib < 0 else 0},{alpha:.3f})"
            )

    # ---------------- Normalize values for Sankey -----------------
    values = np.array(values)
    if values.max() > 0:
        values = (values / values.max()) * MAX_LINK_THICKNESS

    return (
        sources,
        targets,
        values.tolist(),
        labels,
        link_colors,
        hidden_activ,
        output_activ[selected_output_indices],
        output_activ
    )

def get_node_colors(hidden_activ, output_activ, input_row):
    input_colors = [
        f"rgba(0,0,255,{max(v / max(input_row.max(), 1e-6), MIN_NODE_ALPHA) * NODE_COLOR_SCALE:.2f})"
        for v in input_row
    ]

    hidden_colors = [
        f"rgba(0,255,0,{max(h / max(hidden_activ.max(), 1e-6), MIN_NODE_ALPHA) * NODE_COLOR_SCALE:.2f})"
        for h in hidden_activ
    ]

    output_colors = [
        f"rgba(255,0,0,{max(o / max(output_activ.max(), 1e-6), MIN_NODE_ALPHA) * NODE_COLOR_SCALE:.2f})"
        for o in output_activ
    ]

    return input_colors + hidden_colors + output_colors


def get_node_labels(input_row, hidden_activ, output_activ):
    labels = [f"{a}\n{v:.2f}" for a, v in zip(atoms, input_row)]
    labels += [f"h{i}\n{hidden_activ[i]:.2f}" for i in range(len(hidden_activ))]
    labels += [f"{selected_outputs[i]}\n{o:.2f}" for i, o in enumerate(output_activ)]
    return labels


def compute_group_y_positions(sizes, padding=NODE_PADDING, margin=LAYER_MARGIN):
    total = sizes.sum() + padding * (len(sizes) - 1)
    available = 1.0 - 2 * margin

    y_positions = []
    cumulative = 0

    for size in sizes:
        y = margin + (cumulative + size / 2) / total * available
        y_positions.append(y)
        cumulative += size + padding

    return y_positions

# ============================================================
# ====================== BUILD FIGURE ========================
# ============================================================

num_inputs = len(atoms)
num_hidden = model.trainable_variables[0].shape[1]
num_outputs_selected = len(selected_output_indices)

num_nodes = num_inputs + num_hidden + num_outputs_selected
node_x = [0.05] * num_inputs + [0.5] * num_hidden + [0.95] * num_outputs_selected

fig = go.Figure()
frames = []

for step_idx in range(weights_mm.shape[0]):
    layer_weights = unpack_weights(weights_mm[step_idx], model)
    neuron_roles = get_neuron_roles(layer_weights[0]["W"], layer_weights[0]["b"])

    (
        src,
        tgt,
        val,
        lbl,
        link_colors,
        hidden_activ,
        output_activ_sel,
        output_activ_full
    ) = compute_sankey_data(inputs_mm[step_idx], layer_weights, neuron_roles)

    # ---------- UNSAT FILTER ----------
    if output_activ_full[UNSAT_INDEX] >= UNSAT_THRESHOLD:
        continue
    # ---------------------------------

    node_sizes = np.zeros(num_nodes)
    for s, t, v in zip(src, tgt, val):
        node_sizes[s] += v
        node_sizes[t] += v
    node_sizes += 1e-6

    input_nodes = list(range(num_inputs))
    hidden_nodes = list(range(num_inputs, num_inputs + num_hidden))
    output_nodes = list(range(num_inputs + num_hidden, num_nodes))

    node_y = (
        compute_group_y_positions(node_sizes[input_nodes])
        + compute_group_y_positions(node_sizes[hidden_nodes])
        + compute_group_y_positions(node_sizes[output_nodes])
    )

    frame = go.Frame(
        data=[
            go.Sankey(
                node=dict(
                    label=get_node_labels(inputs_mm[step_idx], hidden_activ, output_activ_sel),
                    color=get_node_colors(hidden_activ, output_activ_sel, inputs_mm[step_idx]),
                    pad=15,
                    thickness=20,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=src,
                    target=tgt,
                    value=val,
                    label=lbl,
                    color=link_colors,
                ),
            )
        ],
        name=str(step_idx),
    )

    frames.append(frame)

if not frames:
    raise ValueError("No frames passed the UNSAT_THRESHOLD filter.")

fig.add_trace(frames[0].data[0])

slider_steps = [
    dict(
        method="animate",
        args=[
            [f.name],
            {
                "mode": "immediate",
                "frame": {"duration": 250, "redraw": True},
                "transition": {"duration": 250},
            },
        ],
        label=f.name,
    )
    for f in frames
]

fig.update_layout(
    sliders=[dict(active=0, currentvalue={"prefix": "Step: "}, pad={"t": 50}, steps=slider_steps)],
    autosize=True,
)

fig.frames = frames

# ============================================================
# ========================= SAVE =============================
# ============================================================

output_file = os.path.join(PLOTS_FOLDER, "sankey_dynamic_per_layer_margin.html")
fig.write_html(output_file, include_plotlyjs="cdn")

print(f"Saved: {output_file}")
