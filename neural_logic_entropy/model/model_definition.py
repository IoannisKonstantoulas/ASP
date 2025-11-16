import tensorflow as tf
from tensorflow.keras import layers, Model

def OneHiddenDense(size=20):
    input_set = layers.Input(shape=(size,))
    input_set = layers.Dense(size, activation='relu')(input_set)
    output_set = layers.Dense(size, activation='linear')(input_set)

    return Model(input_set, output_set, name=('OneHiddenDense' + str(size) +'Size'))