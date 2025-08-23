
#!/usr/bin/env python3
""" Variational auto-encoder """
import tensorflow.keras as keras


def sampling(inputs):
    """ Sampling function """
    z_mean, z_log_var = inputs
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder

        - input_dims is an integer containing the dimensions of the
          model input
        - hidden_layers is a list containing the number of nodes for
          each hidden layer in the encoder, respectively
            - the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of the
          latent space representation
        Returns: encoder, decoder, auto
            - encoder is the encoder model, which should output the
              latent representation, the mean, and the log variance,
              respectively
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    encoder_inp = keras.Input(shape=(input_dims,))

    x = keras.layers.Dense(hidden_layers[0], activation='relu')(encoder_inp)
    for i in range(1, len(hidden_layers)):
        x = keras.layers.Dense(hidden_layers[i], activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_var])

    decoder_inp = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[-1],
                           activation='relu')(decoder_inp)
    for i in range(len(hidden_layers) - 2, -1, -1):
        x = keras.layers.Dense(hidden_layers[i], activation='relu')(x)
    decoder_out = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    encoder = keras.models.Model(inputs=encoder_inp,
                                 outputs=[z, z_mean, z_log_var])
    decoder = keras.models.Model(inputs=decoder_inp, outputs=decoder_out)

    comp = encoder(encoder_inp)[0]
    outputs = decoder(comp)
    vae = keras.models.Model(encoder_inp, outputs)

    def loss(y, y_pred):
        """ loss function """
        recons_loss = keras.backend.binary_crossentropy(y, y_pred)
        recons_loss = keras.backend.sum(recons_loss, axis=1)
        kl_loss = (1 + z_log_var - keras.backend.square(z_mean) -
                   keras.backend.exp(z_log_var))
        kl_loss = keras.backend.sum(kl_loss, axis=1)
        kl_loss *= -0.5
        return recons_loss + kl_loss

    vae.compile(optimizer='adam', loss=loss)
    return encoder, decoder, vae
