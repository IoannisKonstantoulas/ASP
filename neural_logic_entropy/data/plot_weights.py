import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_logic_entropy.model.model_definition import OneHiddenDense

matplotlib.use("Agg")

DATE_STRING = "2026-01-01_19.48.07"
NUM_STEPS = 10000
HIDDEN_NEURONS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
OUTPUT_NEURONS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

current_folder = os.path.dirname(__file__)
MEMMAP_PATH = os.path.join(current_folder, "..", "model", "models", DATE_STRING, "weights.memmap")
MEMMAP_PATH = os.path.normpath(MEMMAP_PATH)

plots_folder = os.path.join(current_folder, "plots", DATE_STRING)
os.makedirs(plots_folder, exist_ok=True)
print(f"Plots will be saved to: {plots_folder}")

# -------------------------------
# CREATE MODEL TO CALCULATE PARAMS
# -------------------------------
model = OneHiddenDense(input_size=20, output_size=21)
total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Total parameters dynamically calculated: {total_params}")

# -------------------------------
# DETERMINE MEMMAP ROWS
# -------------------------------
file_size_bytes = os.path.getsize(MEMMAP_PATH)
num_floats = file_size_bytes // 4  # float32
num_steps_in_file = num_floats // total_params
steps_to_load = min(NUM_STEPS, num_steps_in_file)
print(f"Memmap contains {num_steps_in_file} steps, loading first {steps_to_load}")

# -------------------------------
# LOAD MEMMAP SLICE
# -------------------------------
weights = np.memmap(
    MEMMAP_PATH,
    dtype=np.float32,
    mode="r",
    shape=(num_steps_in_file, total_params)
)
weights_slice = np.array(weights[:steps_to_load])

# -------------------------------
# EXTRACT HIDDEN & OUTPUT LAYER WEIGHTS
# -------------------------------
# Hidden layer
hidden_dense = model.layers[1]
hidden_W_shape = hidden_dense.kernel.shape
hidden_b_shape = hidden_dense.bias.shape

hidden_W_start = 0
hidden_W_end = int(np.prod(hidden_W_shape))
hidden_b_start = hidden_W_end
hidden_b_end = hidden_b_start + int(np.prod(hidden_b_shape))

hidden_W = weights_slice[:, hidden_W_start:hidden_W_end].reshape(steps_to_load, *hidden_W_shape)
hidden_b = weights_slice[:, hidden_b_start:hidden_b_end]

# Output layer
output_dense = model.layers[2]
output_W_shape = output_dense.kernel.shape
output_b_shape = output_dense.bias.shape

output_W_start = hidden_b_end
output_W_end = output_W_start + int(np.prod(output_W_shape))
output_b_start = output_W_end
output_b_end = output_b_start + int(np.prod(output_b_shape))

output_W = weights_slice[:, output_W_start:output_W_end].reshape(steps_to_load, *output_W_shape)
output_b = weights_slice[:, output_b_start:output_b_end]


# -------------------------------
# PLOT FUNCTIONS
# -------------------------------
def plot_neurons(weights_array, neuron_indices, layer_name="Layer"):
    plt.figure(figsize=(12, 6))
    for i in neuron_indices:
        if i >= weights_array.shape[1]:
            print(f"Neuron index {i} is out of bounds for {layer_name}, skipping.")
            continue
        plt.plot(weights_array[:, :, i].mean(axis=1), label=f"Neuron {i}")
    plt.title(f"{layer_name} neurons mean weight over steps")
    plt.xlabel("Step")
    plt.ylabel("Mean weight")
    plt.legend()
    filename = os.path.join(plots_folder, f"{layer_name.replace(' ', '_')}_neurons.png")
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def plot_biases(bias_array, neuron_indices, layer_name="Layer"):
    plt.figure(figsize=(12, 6))
    for i in neuron_indices:
        if i >= bias_array.shape[1]:
            print(f"Bias index {i} is out of bounds for {layer_name}, skipping.")
            continue
        plt.plot(bias_array[:, i], label=f"Neuron {i}")
    plt.title(f"{layer_name} biases over steps")
    plt.xlabel("Step")
    plt.ylabel("Bias value")
    plt.legend()
    filename = os.path.join(plots_folder, f"{layer_name.replace(' ', '_')}_biases.png")
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


# -------------------------------
# PLOT SELECTED NEURONS
# -------------------------------
plot_neurons(hidden_W, HIDDEN_NEURONS_TO_PLOT, layer_name="Hidden Layer")
plot_biases(hidden_b, HIDDEN_NEURONS_TO_PLOT, layer_name="Hidden Layer")

plot_neurons(output_W, OUTPUT_NEURONS_TO_PLOT, layer_name="Output Layer")
plot_biases(output_b, OUTPUT_NEURONS_TO_PLOT, layer_name="Output Layer")
