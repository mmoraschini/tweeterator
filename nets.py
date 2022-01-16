from typing import List

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Layer, GRU, SimpleRNN, LSTM, Embedding, SpatialDropout1D, Dense, concatenate


def get_layer_from_str(layer_name: str) -> Layer:
    if layer_name == 'GRU':
        layer = GRU
    elif layer_name == 'RNN':
        layer = SimpleRNN
    elif layer_name == 'LSTM':
        layer = LSTM
    
    return layer


def word_branch(input_layer: Input, layer: Layer, input_dim: int, latent_dim: int,
                window: int, dropout: float, n_hidden: int, n_units: List[int]) -> Layer:
    """
    Create the word prediction branch of the network

    Args:
        input_layer (Input): input layer
        layer (Layer): layer type
        input_dim (int): input dimensions, i.e. number of training words
        latent_dim (int): number of embeddings
        window (int): size of the training window
        dropout (float): dropout amount (0 to disable)
        n_hidden (int): number of hidden layers
        n_units (List[int]): number of units for each hidden layer

    Returns:
        Layer: the last hidden layer
    """

    x = Embedding(input_dim=input_dim, output_dim=latent_dim, input_length=window, name='w_embedding')(input_layer)
    if dropout > 0:
        x = SpatialDropout1D(dropout, name='w_dropout')(x)

    return_sequences = True
    for i in range(0, n_hidden):
        if i == n_hidden - 1:
            return_sequences = False
        x = layer(n_units[i], return_sequences=return_sequences, name=f'w_hl{i+1}')(x)

    return x


def pos_branch(input_layer: Input, layer: Layer, input_dim: int, latent_dim: int,
               window: int, dropout: float, n_hidden: int, n_units: List[int]) -> Layer:
    """
    Create the POS tag prediction branch of the network

    Args:
        input_layer (Input): input layer
        layer (Layer): layer type
        input_dim (int): input dimensions, i.e. number of total unique training words
        latent_dim (int): number of embeddings
        window (int): size of the training window
        dropout (float): dropout amount (0 to disable)
        n_hidden (int): number of hidden layers
        n_units (List[int]): number of units for each hidden layer

    Returns:
        Layer: the last hidden layer
    """

    x = Embedding(input_dim=input_dim, output_dim=latent_dim, input_length=window, name='pos_embedding')(input_layer)
    if dropout > 0:
        x = SpatialDropout1D(dropout, name='pos_dropout')(x)

    return_sequences = True
    for i in range(0, n_hidden):
        if i == n_hidden - 1:
            return_sequences = False
        x = layer(n_units[i], return_sequences=return_sequences, name=f'pos_hl{i+1}')(x)

    return x


def one_input_one_output(window: int, layer_name: str, input_dim: int, latent_dim: int, n_units: List[int], dropout: float, n_hidden: int) -> Sequential:
    """
    Create a next word prediction model

    Args:
        window (int): size of the training window
        layer (Layer): layer type
        input_dim (int): input dimensions, i.e. number of total unique training POS tags
        latent_dim (int): number of embeddings
        n_units (List[int]): number of units for each hidden layer        
        dropout (float): dropout amount (0 to disable)
        n_hidden (int): number of hidden layers

    Returns:
        Sequential: the full model
    """

    if type(n_units) == int:
        n_units = [n_units]

    layer = get_layer_from_str(layer_name)

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


def two_inputs_one_output(window: int,
              w_layer_name: Layer, w_input_dim: int, w_latent_dim: int, w_n_units: List[int], w_dropout: float, w_n_hidden: int,
              pos_layer_name: Layer, pos_input_dim: int, pos_latent_dim: int, pos_n_units: List[int], pos_dropout: float, pos_n_hidden: int)  -> Model:
    """
    Create a prediction model with two input branches, one for words and one for POS tags, and one output for the next word prediction.

    Args:
        window (int): [description]
        w_layer_name (str): type of layer of the word prediction part
        w_input_dim (int): input dimensions, i.e. number of total unique training words
        w_latent_dim (int): number of embeddings in the word prediction part
        w_n_units (List[int]): number of units for each hidden layer of the word prediction part
        w_dropout (float): dropout amount (0 to disable) in the word prediction part
        w_n_hidden (int): number of hidden layers in the word prediction part
        pos_layer_name (str): type of layer of the POS prediction part
        pos_input_dim (int): input dimensions, i.e. number of total unique training POS tags
        pos_latent_dim (int): number of embeddings in the POS prediction part
        pos_n_units (List[int]): number of units for each hidden layer in the POS prediction part
        pos_dropout (float): dropout amount (0 to disable) in the POS prediction part
        pos_n_hidden (int): number of hidden layers in the POS prediction part

    Returns:
        Model: the full model
    """

    if type(w_n_units) == int:
        w_n_units = [w_n_units]

    if type(pos_n_units) == int:
        pos_n_units = [pos_n_units]
    
    w_layer = get_layer_from_str(w_layer_name)
    pos_layer = get_layer_from_str(pos_layer_name)
    
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


def two_inputs_two_outputs(window: int,
              w_layer_name: Layer, w_input_dim: int, w_latent_dim: int, w_n_units: List[int], w_dropout: float, w_n_hidden: int,
              pos_layer_name: Layer, pos_input_dim: int, pos_latent_dim: int, pos_n_units: List[int], pos_dropout: float, pos_n_hidden: int)  -> Model:
    """
    Create a prediction model with two input branches, one for words and one for POS tags, and two outputs,
    one for the next word prediction and one for the associated POS tag prediction.

    Args:
        window (int): [description]
        w_layer_name (str): type of layer of the word prediction part
        w_input_dim (int): input dimensions, i.e. number of total unique training words
        w_latent_dim (int): number of embeddings in the word prediction part
        w_n_units (List[int]): number of units for each hidden layer of the word prediction part
        w_dropout (float): dropout amount (0 to disable) in the word prediction part
        w_n_hidden (int): number of hidden layers in the word prediction part
        pos_layer_name (str): type of layer of the POS prediction part
        pos_input_dim (int): input dimensions, i.e. number of total unique training POS tags
        pos_latent_dim (int): number of embeddings in the POS prediction part
        pos_n_units (List[int]): number of units for each hidden layer in the POS prediction part
        pos_dropout (float): dropout amount (0 to disable) in the POS prediction part
        pos_n_hidden (int): number of hidden layers in the POS prediction part

    Returns:
        Model: the full model
    """
    
    if type(w_n_units) == int:
        w_n_units = [w_n_units]

    if type(pos_n_units) == int:
        pos_n_units = [pos_n_units]

    w_layer = get_layer_from_str(w_layer_name)
    pos_layer = get_layer_from_str(pos_layer_name)
    
    w_input = Input(shape=(window,), name='w_input')
    pos_input = Input(shape=(window,), name='pos_input')
    w_branch = word_branch(w_input, w_layer, w_input_dim, w_latent_dim, window, w_dropout, w_n_hidden, w_n_units)
    p_branch = pos_branch(pos_input, pos_layer, pos_input_dim, pos_latent_dim, window, pos_dropout, pos_n_hidden, pos_n_units)

    x = concatenate([w_branch, p_branch])

    w_output = Dense(w_input_dim, activation='softmax', name='w_output')(x)
    pos_output = Dense(pos_input_dim, activation='softmax', name='pos_output')(x)

    model = Model(
        inputs=[w_input, pos_input],
        outputs=[w_output, pos_output])
    
    return model
