import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_logic_entropy.model.model_definition import OneHiddenDense
from neural_logic_entropy.util.inspect_utils import unpack_weights, save_heatmap, get_latest_model_date_string

matplotlib.use("Agg")

# ---------------- CONFIG ----------------
ROWS_TO_PLOT = 3           # how many input rows to plot
STEP_START = 7520             # first step to plot
STEP_END = 7520 + 1              # last step to plot (exclusive)
NUM_STEPS = 10000          # max steps to load
OUTPUT_INDEX = None        # None = all outputs, 0-20 = single output neuron

DATE_STRING = None
if DATE_STRING is None:
    DATE_STRING = get_latest_model_date_string()  # auto-select latest model folder

PLOTS_FOLDER = os.path.join("plots", DATE_STRING)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ---------------- CURRENT FOLDER ----------------
current_folder = os.path.dirname(os.path.abspath(__file__))

# ---------------- MEMMAP PATHS ----------------
MEMMAP_PATH_WEIGHTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "weights.memmap")
)
MEMMAP_PATH_INPUTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "inputs.memmap")
)
MEMMAP_PATH_OUTPUTS = os.path.normpath(
    os.path.join(current_folder, "..", "model", "models", DATE_STRING, "outputs.memmap")
)

# ---------------- LOAD MEMMAPS ----------------
model = OneHiddenDense(input_size=20, output_size=21)
total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

file_size = os.path.getsize(MEMMAP_PATH_WEIGHTS)
dtype_size = np.dtype(np.float32).itemsize
num_steps_saved = file_size // (total_params * dtype_size)
num_steps = min(NUM_STEPS, num_steps_saved)

weights_mm = np.array(
    np.memmap(MEMMAP_PATH_WEIGHTS, dtype=np.float32, mode="r", shape=(num_steps_saved, total_params))[STEP_START:STEP_END]
)
inputs_mm = np.array(
    np.memmap(MEMMAP_PATH_INPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, 20))[STEP_START:STEP_END]
)
outputs_mm = np.array(
    np.memmap(MEMMAP_PATH_OUTPUTS, dtype=np.float32, mode="r", shape=(num_steps_saved, 21))[STEP_START:STEP_END]
)

print(f"Loaded steps {STEP_START} → {STEP_END}, total: {weights_mm.shape[0]} steps.")

# ---------------- CONTRIBUTIONS ----------------
def compute_contributions(input_row, layer_weights, output_idx=None):
    hidden_W = layer_weights[0]['W']   # shape: (input_dim, hidden_dim)
    hidden_b = layer_weights[0]['b']   # shape: (hidden_dim,)
    output_W = layer_weights[1]['W']   # shape: (hidden_dim, output_dim)
    output_b = layer_weights[1]['b']   # shape: (output_dim,)

    # Input → Hidden (linear pre-activation)
    input_hidden_linear = input_row @ hidden_W
    hidden_activ = np.maximum(0, input_hidden_linear + hidden_b)  # ReLU

    # Hidden → Output
    if output_idx is None:
        hidden_output = hidden_activ @ output_W + output_b
        effective_input_output = np.zeros((input_row.size, output_W.shape[1]))
        for i in range(input_row.size):
            for j in range(output_W.shape[1]):
                # contribution of input i to output j through active hidden neurons
                effective_input_output[i, j] = np.sum(
                    input_row[i] * hidden_W[i, :] * output_W[:, j] * (input_hidden_linear + hidden_b > 0)
                )
        pred_output = 1 / (1 + np.exp(-hidden_output))  # sigmoid
    else:
        hidden_output = hidden_activ * output_W[:, output_idx]
        effective_input_output = np.zeros((input_row.size, 1))
        for i in range(input_row.size):
            effective_input_output[i, 0] = np.sum(
                input_row[i] * hidden_W[i, :] * output_W[:, output_idx] * (input_hidden_linear + hidden_b > 0)
            )
        pred_output = 1 / (1 + np.exp(-(hidden_output.sum() + output_b[output_idx])))  # sigmoid

    return {
        "input_hidden": input_hidden_linear.reshape(1, -1),
        "hidden_output": hidden_output.reshape(1, -1),
        "effective_input_output": effective_input_output,
        "pred_output": pred_output
    }

# ---------------- PLOTTING ----------------
for row_idx in range(min(ROWS_TO_PLOT, inputs_mm.shape[0])):
    input_row = inputs_mm[row_idx]
    actual_output = outputs_mm[row_idx] if OUTPUT_INDEX is None else outputs_mm[row_idx, OUTPUT_INDEX]

    for step_idx in range(weights_mm.shape[0]):
        weights_step = weights_mm[step_idx]
        layer_weights = unpack_weights(weights_step, model)

        result = compute_contributions(input_row, layer_weights, OUTPUT_INDEX)

        # --- Input → Hidden ---
        title = f"Input → Hidden Contribution (row={row_idx}, step={STEP_START + step_idx})"
        filename = os.path.join(PLOTS_FOLDER, f"input_hidden_dir(data_row={row_idx}, step={STEP_START + step_idx}).png")
        save_heatmap(result["input_hidden"], title=title, filename=filename,
                     xlabel="Hidden neuron", ylabel="Input feature")

        # --- Hidden → Output ---
        title = f"Hidden → Output Contribution (row={row_idx}, step={STEP_START + step_idx})"
        filename = os.path.join(PLOTS_FOLDER, f"hidden_to_output(data_row={row_idx}, step={STEP_START + step_idx}).png")
        save_heatmap(result["hidden_output"], title=title, filename=filename,
                     xlabel="Output neuron" if OUTPUT_INDEX is None else f"Output neuron {OUTPUT_INDEX}",
                     ylabel="Hidden neuron")

        # --- Effective Input → Output ---
        title = f"Effective Input → Output Contribution (row={row_idx}, step={STEP_START + step_idx})"
        filename = os.path.join(PLOTS_FOLDER, f"input_to_output_effective(data_row={row_idx}, step={STEP_START + step_idx}).png")
        save_heatmap(result["effective_input_output"], title=title, filename=filename,
                     xlabel="Output neuron" if OUTPUT_INDEX is None else f"Output neuron {OUTPUT_INDEX}",
                     ylabel="Input feature")

        # --- Predicted vs Actual ---
        plt.figure(figsize=(8, 4))
        if OUTPUT_INDEX is None:
            plt.plot(actual_output, label="Actual Output", marker='o')
            plt.plot(result["pred_output"], label="Predicted Output", marker='x')
        else:
            plt.bar([0], [actual_output], label="Actual Output")
            plt.bar([0], [result["pred_output"]], alpha=0.6, label="Predicted Output")
        plt.title(f"Predicted vs Actual Output (row={row_idx}, step={STEP_START + step_idx})")
        plt.xlabel("Output neuron" if OUTPUT_INDEX is None else f"Output neuron {OUTPUT_INDEX}")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, f"pred_vs_actual(data_row={row_idx}, step={STEP_START + step_idx}).png"))
        plt.close()

print(f"Plots saved to {PLOTS_FOLDER}")
