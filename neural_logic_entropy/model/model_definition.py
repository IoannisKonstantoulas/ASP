from tensorflow.keras import layers, Model


def OneHiddenDense(input_size=20, output_size=21):
    input_set = layers.Input(shape=(input_size,))
    temp_layer = layers.Dense(output_size, activation='relu')(input_set)
    output_set = layers.Dense(output_size, activation='sigmoid')(temp_layer)

    return Model(input_set, output_set, name=('OneHiddenDense' + str(input_size) + 'InputSize' + str(output_size) + 'OutputSize'))
