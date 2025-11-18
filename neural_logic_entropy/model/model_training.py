import pandas as pd
import tensorflow as tf

from neural_logic_entropy.model.model_definition import OneHiddenDense

file_path = '../data/dataset_satisfiable_and_unsatisfiable.csv'

model = OneHiddenDense(20, 21)
df = pd.read_csv(file_path)

data = df.values

input_data = data[:, 0:20]
output_data = data[:, 20:41]
mask = data[:, 41:62]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

model.fit(input_data, output_data, sample_weight=mask, epochs=20, batch_size=1, validation_split=0.1)

model.save('first_model.keras')
