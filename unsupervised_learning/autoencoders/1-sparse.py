#!/usr/bin/env python3
""" Sparse autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ creates a sparse autoencoder

        - input_dims is an integer containing the dimensions of the model
          input
        - hidden_layers is a list containing the number of nodes for each
          hidden layer in the encoder, respectively
            - the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of the
          latent space representation
        - lambtha is the regularization parameter used for L1
          regularization on the encoded output
        Returns: encoder, decoder, auto
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the sparse autoencoder model
    """
    regularizer = keras.regularizers.l1(lambtha)
    # General input
    inputs = keras.Input(shape=(input_dims,))
    # Encoded
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(inputs)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    comp = keras.layers.Dense(latent_dims, activation='relu',
                              activity_regularizer=regularizer)(encoded)

    dec_input = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(dec_input)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    reconstructed = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(decoded)

    # Encoder and Decoder models
    encoder = keras.models.Model(inputs, comp)
    decoder = keras.models.Model(dec_input, reconstructed)

    # Fill the complete model and create it
    inp = encoder(inputs)
    outputs = decoder(inp)
    auto = keras.models.Model(inputs, outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
