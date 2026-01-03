import os
import numpy as np
from neural_logic_entropy.model.model_definition import OneHiddenDense
from neural_logic_entropy.util.inspect_utils import unpack_weights, get_latest_model_date_string

# ---------------- CONFIG ----------------
DATE_STRING = None
if DATE_STRING is None:
    DATE_STRING = get_latest_model_date_string()

STEP_START = 0
STEP_END = 100  # number of steps to analyze
INPUT_SIZE = 20
HIDDEN_SIZE = 21
OUTPUT_SIZE = 21

# ---------------- PATHS ----------------
current_folder = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.normpath(os.path.join(current_folder, "..", "model", "models", DATE_STRING))
MEMMAP_PATH_WEIGHTS = os.path.join(MODEL_FOLDER, "weights.memmap")
MEMMAP_PATH_INPUTS = os.path.join(MODEL_FOLDER, "inputs.memmap")
MEMMAP_PATH_OUTPUTS = os.path.join(MODEL_FOLDER, "outputs.memmap")

# ---------------- LOAD MEMMAPS ----------------
model = OneHiddenDense(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

file_size = os.path.getsize(MEMMAP_PATH_WEIGHTS)
dtype_size = np.dtype(np.float32).itemsize
num_steps_saved = file_size // (total_params * dtype_size)
num_steps = min(STEP_END, num_steps_saved)

weights_mm = np.memmap(MEMMAP_PATH_WEIGHTS, dtype=np.float32, mode="r", shape=(num_steps_saved, total_params))[STEP_START:STEP_END]
inputs_mm = np.memmap(MEMMAP_PATH_INPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, INPUT_SIZE))[STEP_START:STEP_END]
outputs_mm = np.memmap(MEMMAP_PATH_OUTPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, OUTPUT_SIZE))[STEP_START:STEP_END]

# ---------------- NEURON ROLE DISCOVERY ----------------
hidden_activations = np.zeros((HIDDEN_SIZE, INPUT_SIZE))  # rows: hidden neurons, cols: inputs

for step_idx in range(num_steps):
    weights_step = unpack_weights(weights_mm[step_idx], model)
    hidden_W = weights_step[0]['W']
    hidden_b = weights_step[0]['b']

    for input_idx in range(inputs_mm.shape[1]):
        input_vector = np.zeros(INPUT_SIZE)
        input_vector[input_idx] = 1.0  # activate one feature at a time
        z_hidden = input_vector @ hidden_W + hidden_b
        activ_hidden = np.maximum(0, z_hidden)  # ReLU
        hidden_activations[:, input_idx] += activ_hidden

# Average across steps
hidden_activations /= num_steps

# ---------------- ANALYSIS ----------------
atoms = [
    "man(a)", "not_man(a)", "woman(a)", "not_woman(a)",
    "man(b)", "not_man(b)", "woman(b)", "not_woman(b)",
    "parent(a,b)", "not_parent(a,b)", "parent(b,a)", "not_parent(b,a)",
    "father(a,b)", "not_father(a,b)", "father(b,a)", "not_father(b,a)",
    "mother(a,b)", "not_mother(a,b)", "mother(b,a)", "not_mother(b,a)"
]

# Which input feature activates each hidden neuron most?
for h_idx in range(HIDDEN_SIZE):
    top_inputs = np.argsort(-hidden_activations[h_idx])[:3]  # top 3 inputs
    top_features = [atoms[i] for i in top_inputs]
    print(f"Hidden neuron h{h_idx} is most activated by: {top_features} (avg activation: {hidden_activations[h_idx, top_inputs]})")
