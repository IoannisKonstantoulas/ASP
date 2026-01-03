from tensorflow.keras import layers, Model


def OneHiddenDense(input_size=20, hidden_size=10, output_size=21):
    input_set = layers.Input(shape=(input_size,))
    temp_layer = layers.Dense(hidden_size, activation='relu')(input_set)
    output_set = layers.Dense(output_size, activation='sigmoid')(temp_layer)

    return Model(input_set, output_set, name=('OneHiddenDense' + str(input_size) + 'InputSize' + str(output_size) + 'OutputSize'))


def OnlyOutputLayer(input_size=20, output_size=21):
    input_set = layers.Input(shape=(input_size,))
    output_set = layers.Dense(output_size, activation='sigmoid')(input_set)

    return Model(input_set, output_set, name=('OnlyOutputLayer' + str(input_size) + 'InputSize' + str(output_size) + 'OutputSize'))

def OneHiddenDenseWithHidden(input_size=20, hidden_size=10, output_size=21):
    input_set = layers.Input(shape=(input_size,))
    temp_layer = layers.Dense(hidden_size, activation='relu', name='hidden')(input_set)
    output_set = layers.Dense(output_size, activation='sigmoid', name='output')(temp_layer)

    # Model now returns both output and hidden
    return Model(inputs=input_set, outputs=[output_set, temp_layer],
                 name=f'OneHiddenDense_{input_size}Input_{output_size}Output')