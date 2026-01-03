import os
import numpy as np
import plotly.graph_objects as go
from neural_logic_entropy.model.model_definition import OneHiddenDense
from neural_logic_entropy.util.inspect_utils import unpack_weights, get_latest_model_date_string

# ---------------- CONFIG ----------------
STEP_START = 0
STEP_END = 10  # number of steps to animate
DATE_STRING = None
if DATE_STRING is None:
    DATE_STRING = get_latest_model_date_string()

# ---------------- PATHS ----------------
current_folder = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.normpath(os.path.join(current_folder, "..", "model", "models", DATE_STRING))

MEMMAP_PATH_WEIGHTS = os.path.join(MODEL_FOLDER, "weights.memmap")
MEMMAP_PATH_INPUTS = os.path.join(MODEL_FOLDER, "inputs.memmap")
MEMMAP_PATH_OUTPUTS = os.path.join(MODEL_FOLDER, "outputs.memmap")

# ---------------- LOAD DATA ----------------
model = OneHiddenDense(input_size=20, output_size=21)
total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

file_size = os.path.getsize(MEMMAP_PATH_WEIGHTS)
dtype_size = np.dtype(np.float32).itemsize
num_steps_saved = file_size // (total_params * dtype_size)

weights_mm = np.memmap(MEMMAP_PATH_WEIGHTS, dtype=np.float32, mode="r", shape=(num_steps_saved, total_params))[STEP_START:STEP_END]
inputs_mm = np.memmap(MEMMAP_PATH_INPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, 20))[STEP_START:STEP_END]
outputs_mm = np.memmap(MEMMAP_PATH_OUTPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, 21))[STEP_START:STEP_END]

# ---------------- ATOMS ----------------
atoms = [
    "man(a)", "not_man(a)", "woman(a)", "not_woman(a)",
    "man(b)", "not_man(b)", "woman(b)", "not_woman(b)",
    "parent(a,b)", "not_parent(a,b)", "parent(b,a)", "not_parent(b,a)",
    "father(a,b)", "not_father(a,b)", "father(b,a)", "not_father(b,a)",
    "mother(a,b)", "not_mother(a,b)", "mother(b,a)", "not_mother(b,a)"
]
hidden_labels = [f"h{i}" for i in range(21)]
output_labels = [f"o{i}" for i in range(20)] + ["unsat"]
labels = atoms + hidden_labels + output_labels

# ---------------- HELPER FUNCTION ----------------
def compute_step_contrib(input_row, layer_weights):
    hidden_W = layer_weights[0]['W']
    hidden_b = layer_weights[0]['b']
    output_W = layer_weights[1]['W']
    output_b = layer_weights[1]['b']

    # Input → Hidden
    z_hidden = input_row @ hidden_W + hidden_b
    hidden_activ = np.maximum(0, z_hidden)
    input_hidden_contrib = (input_row[:, None] * hidden_W.T).T
    input_hidden_contrib[z_hidden <= 0] = 0

    # Hidden → Output
    z_output = hidden_activ @ output_W + output_b
    output_activ = 1 / (1 + np.exp(-z_output))
    hidden_output_contrib = (hidden_activ[:, None] * output_W.T).T

    return input_hidden_contrib, hidden_output_contrib

# ---------------- BUILD ANIMATED SANKEY ----------------
frames = []

for step_idx in range(STEP_END - STEP_START):
    weights_step = unpack_weights(weights_mm[step_idx], model)
    input_row = inputs_mm[step_idx]

    input_hidden_contrib, hidden_output_contrib = compute_step_contrib(input_row, weights_step)

    source = []
    target = []
    value = []

    # Input → Hidden
    for i in range(20):
        for j in range(21):
            if input_hidden_contrib[i,j] != 0:
                source.append(i)
                target.append(20 + j)
                value.append(input_hidden_contrib[i,j])

    # Hidden → Output
    for j in range(21):
        for k in range(21):
            if hidden_output_contrib[j,k] != 0:
                source.append(20 + j)
                target.append(41 + k)
                value.append(hidden_output_contrib[j,k])

    frame = go.Frame(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value)
    )], name=f"step_{STEP_START + step_idx}")
    frames.append(frame)

# ---------------- INITIAL SANKEY ----------------
fig = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        title=f"Causal Sankey Evolution (steps {STEP_START}→{STEP_END-1})",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
    ),
    frames=frames
)

# ---------------- SAVE ----------------
output_file = os.path.join("plots", DATE_STRING, f"sankey_evolution_{STEP_START}_{STEP_END-1}.html")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
fig.write_html(output_file)
print(f"Animated Sankey saved to {output_file}")
