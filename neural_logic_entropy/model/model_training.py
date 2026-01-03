import datetime
import os

import pandas as pd
import tensorflow as tf

from neural_logic_entropy.model.custom_training_loop import train_with_full_weight_logging
from neural_logic_entropy.model.custom_training_loop_with_belief_consistency import train_with_full_weight_logging_belief
from neural_logic_entropy.model.model_definition import OneHiddenDense, OnlyOutputLayer, OneHiddenDenseWithHidden

file_path = "../data/dataset_satisfiable_and_unsatisfiable.csv"
df = pd.read_csv(file_path)
data = df.values.astype("float32")

input_data = data[:, 0:20]
output_data = data[:, 20:41]
mask = data[:, 41:62]

model = OneHiddenDenseWithHidden(input_size=20, hidden_size=10, output_size=21)
optimizer = tf.keras.optimizers.Adam()

now = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
run_folder = os.path.join("models", now)
os.makedirs(run_folder, exist_ok=True)
model_file = os.path.join(run_folder, "model.keras")
memmap_path_weights = os.path.join(run_folder, "weights.memmap")
memmap_path_inputs = os.path.join(run_folder, "inputs.memmap")
memmap_path_outputs = os.path.join(run_folder, "outputs.memmap")

train_with_full_weight_logging_belief(
    model=model,
    input_data=input_data,
    output_data=output_data,
    sample_weight=mask,
    optimizer=optimizer,
    memmap_path_weights=memmap_path_weights,
    memmap_path_inputs=memmap_path_inputs,
    memmap_path_outputs=memmap_path_outputs,
    val_split=0.1,
    epochs=1,
    batch_size=16,
    print_every=10000,
    early_stop_patience=10,
    random_seed=None,
    lambda_belief=1e-4
)

model.save(model_file)
