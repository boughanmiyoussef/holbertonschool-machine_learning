#!/usr/bin/env python3
"""
Wasserstein GANs with gradient penalty
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class WGAN_GP(keras.Model):
    """
    This class represents a Wasserstein GAN (WGAN) with weight clipping.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN model with a generator, discriminator,
        latent generator, and real examples.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3   # standard value, but can be changed if necessary
        self.beta_2 = .9   # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
    Generates an interpolated sample between real and fake samples.

    This is typically used in Wasserstein GANs with Gradient Penalty(WGAN-GP),
    where interpolation between real &fake datapoints is needed to compute the
    gradient penalty term.
    Args:
        real_sample (tf.Tensor): A batch of real data samples.
        fake_sample (tf.Tensor): A batch of generated (fake) data samples.
    Returns:
        tf.Tensor: A batch interpolated samples, where each sample is a linear
        interpolation between a real and a fake sample using random weights.
    The interpolation is computed as:
        interpolated = u * real_sample + (1 - u) * fake_sample
    where `u` is a random tensor from a uniform distribution between 0 and 1,
    with the same shape as the batch.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape)-u
        return u*real_sample+v*fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
    Computes the gradient penalty for a batch of interpolated samples.

    This is used in Wasserstein GANs with Gradient Penalty to enforce
    the Lipschitz constraint by penalizing the norm of the gradient of the
    discriminator's output with respect to its input.
    Args:
        interpolated_sample: A batch of interpolated samples, typically
            generated as a linear interpolation between real and fake samples.
    Returns:
        tf.Tensor: The mean squared difference between the L2 norm of the
        gradients and 1.0. A lower value indicates a better enforcement of
        the Lipschitz constraint.
    Formula:
        penalty = E[(||∇D(ẋ)||₂ - 1)²]
        where ẋ is the interpolated input and D is the discriminator.

    Notes:
        - The gradients are taken with respect to the interpolated samples.
        - The norm is computed across the feature dimensions.
        - `self.axis` should typically exclude the batch dimension
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        Performs one training step for the WGAN model using weight clipping.
        Args:
            - useless_argument: not used in this implementation.
        Returns:
            - dict: containing losses and gradient penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_sample = \
                    self.get_interpolated_sample(real_samples, fake_samples)

                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            discr_gradients = \
                disc_tape.gradient(new_discr_loss,
                                   self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients,
                    self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        gen_gradients = \
            gen_tape.gradient(gen_loss,
                              self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replace the weights of the generator and discriminator
        Args:
            gen_h5:generator have been stored in the .h5 file gen_h5
            disc_h5:discriminator have been stored in the .h5 file gen_h5
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
