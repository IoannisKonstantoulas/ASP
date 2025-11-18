import numpy as np
import pandas as pd
import tensorflow.keras.losses as losses
from tensorflow.keras.models import load_model

file_path = '../data/dataset_satisfiable_and_unsatisfiable.csv'

df = pd.read_csv(file_path)

input_data = df.iloc[2257, 0:20].values
input_data = np.array(input_data)
input_data = np.expand_dims(input_data, axis=0)

output_data = df.iloc[2257, 20:41].values
output_data = np.array(output_data)
output_data = np.expand_dims(output_data, axis=0)

model = load_model('first_model.keras', custom_objects={'masked_bce': losses.binary_crossentropy})

output_model = model.predict(input_data)

check_list = np.concatenate((output_data, output_model), axis=0)
print(check_list)
