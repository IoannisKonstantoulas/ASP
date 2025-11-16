import pandas as pd

from neural_logic_entropy.model.model_definition import OneHiddenDense

file_path = '../data/dataset.csv'

model = OneHiddenDense(20)
df = pd.read_csv(file_path)

input_data = df.iloc[:, 0:20].values
output_data = df.iloc[:, 20:40].values

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['accuracy', 'mae']
)

model.fit(input_data, output_data, epochs=100, batch_size=32, validation_split=0.1)

model.save('first_model.keras')
