import os

import numpy as np
import pandas as pd
import tensorflow.keras.losses as losses
from tensorflow.keras.models import load_model

from neural_logic_entropy.util.inspect_utils import get_latest_model_date_string

file_path = '../data/dataset_satisfiable_and_unsatisfiable.csv'

# DATE_STRING = "2026-01-02_16.59.40"
DATE_STRING = None
if DATE_STRING is None:
    DATE_STRING = get_latest_model_date_string()
MODEL_PATH = os.path.join("models", DATE_STRING, "model.keras")

df = pd.read_csv(file_path)

input_data = df.iloc[8257, 0:20].values
input_data = np.array(input_data)
input_data = np.expand_dims(input_data, axis=0)

output_data = df.iloc[8257, 20:41].values
output_data = np.array(output_data)
output_data = np.expand_dims(output_data, axis=0)

model = load_model(MODEL_PATH, custom_objects={'masked_bce': losses.binary_crossentropy})

output_model = model.predict(input_data)

check_list = np.concatenate((output_data, output_model), axis=0)
print(check_list)
