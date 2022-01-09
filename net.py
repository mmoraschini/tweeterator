from typing import List

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Layer, Embedding, SpatialDropout1D, Dense, concatenate


def word_branch(input_layer: Input, layer: Layer, input_dim: int, latent_dim: int,
                window: int, dropout: float, n_hidden: int, n_units: List[int]) -> Dense:

    x = Embedding(input_dim=input_dim, output_dim=latent_dim, input_length=window, name='w_embedding')(input_layer)
    if dropout > 0:
        x = SpatialDropout1D(dropout, name='w_dropout')(x)

    return_sequences = True
    for i in range(0, n_hidden):
        if i == n_hidden - 1:
            return_sequences = False
        x = layer(n_units[i], return_sequences=return_sequences, name=f'w_hl{i+1}')(x)

    # output_layer = Dense(input_dim, activation='softmax', name='w_output')(x)

    return x

def pos_branch(input_layer: Input, layer: Layer, input_dim: int, latent_dim: int,
               window: int, dropout: float, n_hidden: int, n_units: List[int]) -> Dense:

    x = Embedding(input_dim=input_dim, output_dim=latent_dim, input_length=window, name='pos_embedding')(input_layer)
    if dropout > 0:
        x = SpatialDropout1D(dropout, name='pos_dropout')(x)

    return_sequences = True
    for i in range(0, n_hidden):
        if i == n_hidden - 1:
            return_sequences = False
        x = layer(n_units[i], return_sequences=return_sequences, name=f'pos_hl{i+1}')(x)

    # output_layer = Dense(input_dim, activation='softmax', name='pos_output')(x)

    return x

def single_branch(window: int, layer: Layer, input_dim: int, latent_dim: int, n_units: List[int], dropout: float, n_hidden: int) -> Sequential:
    model = Sequential()

    model.add(Embedding(input_dim=input_dim, output_dim=latent_dim, input_length=window, name='embedding'))
    if dropout > 0:
        model.add(SpatialDropout1D(dropout, name='dropout'))

    return_sequences = True
    for i in range(0, n_hidden):
        if i == n_hidden - 1:
            return_sequences = False
        model.add(layer(n_units[i], return_sequences=return_sequences, name=f'hl{i+1}'))

    model.add(Dense(input_dim, activation='softmax', name='output'))

    return model


def double_branch(window: int,
              w_layer: Layer, w_input_dim: int, w_latent_dim: int, w_n_units: List[int], w_dropout: float, w_n_hidden: int,
              pos_layer: Layer, pos_input_dim: int, pos_latent_dim: int, pos_n_units: List[int], pos_dropout: float, pos_n_hidden: int)  -> Model:
    
    w_input = Input(shape=(window,), name='w_input')
    pos_input = Input(shape=(window,), name='pos_input')
    w_branch = word_branch(w_input, w_layer, w_input_dim, w_latent_dim, window, w_dropout, w_n_hidden, w_n_units)
    p_branch = pos_branch(pos_input, pos_layer, pos_input_dim, pos_latent_dim, window, pos_dropout, pos_n_hidden, pos_n_units)

    x = concatenate([w_branch, p_branch])

    outputs = Dense(w_input_dim, activation='softmax', name='output')(x)

    model = Model(
        inputs=[w_input, pos_input],
        outputs=outputs)
    
    return model