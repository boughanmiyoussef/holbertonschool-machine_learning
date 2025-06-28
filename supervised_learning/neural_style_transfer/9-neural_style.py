#!/usr/bin/env python3
"""performs tasks for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """
    style_image - the preprocessed style image
    content_image - the preprocessed content image
    alpha - the weight for content cost
    beta - the weight for style cost
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        style_image - the image used as a style reference,
            stored as a numpy.ndarray
        content_image - the image used as a content reference,
            stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        model - the Keras model used to calculate cost
        gram_style_features - a list of gram matrices calculated from the
            style layer outputs of the style image
        content_feature - the content layer output of the content image
        """
        valid = (isinstance(style_image, np.ndarray)
                 and style_image.ndim == 3 and style_image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        valid_img = (isinstance(content_image, np.ndarray)
                     and content_image.ndim == 3
                     and content_image.shape[2] == 3)
        if not valid_img:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3)
        containing the image to be scaled
        Returns: the scaled image
        """
        valid = (isinstance(image, np.ndarray)
                 and image.ndim == 3 and image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_size = (int(h * scale), int(w * scale))
        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

        # Resize using bicubic interpolation (while in [0, 255] range)
        resized_image = tf.image.resize(
            image_tensor,
            new_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

        # rescale to [0, 1]
        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        creates the model used to calculate cost
        """
        # Load VGG19 without top layers and pretrained on ImageNet
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        # Replace MaxPooling2D layers with AveragePooling2D
        pooling_layers = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        vgg.save("base_vgg.h5")
        # Reload the VGG model with the pooling layers swapped
        vgg = tf.keras.models.load_model("base_vgg.h5",
                                         custom_objects=pooling_layers)
        # Make sure that the model is non-trainable
        vgg.trainable = False
        # Get the desired layer outputs
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        # Create a model that returns the outputs
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        # Get dimensions
        batch, height, width, channels = tf.shape(input_layer)
        # Unroll the input layer
        input_layer = tf.reshape(input_layer, [-1, channels])
        # Calculate the Gram matrix
        gram = tf.matmul(input_layer, input_layer, transpose_a=True)
        # Normalize by the number of spatial locations (height * width)
        gram = tf.expand_dims(gram, axis=0)  # Add batch dimension
        return gram / tf.cast(height * width, tf.float32)

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        # Get the style features and content features
        # For style image outputs
        style_outputs = self.model(style_image)
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]

        # For content image outputs
        self.content_feature = self.model(content_image)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        ch = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")
        if len(gram_target.shape) != 3 or gram_target.shape != [1, ch, ch]:
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")

        # Calculate the Gram matrix for the style output
        gram_style = self.gram_matrix(style_output)
        # Calculate the style cost
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for all layers"""
        L = len(self.style_layers)
        if not isinstance(style_outputs, list):
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        if len(style_outputs) != L:
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        # Calculate the style cost for each layer
        style_cost = 0
        for target, output in zip(self.gram_style_features, style_outputs):
            style_cost += self.layer_style_cost(output, target)
        return style_cost / L

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image"""
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        if len(content_output.shape) != len(self.content_feature.shape):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        # Calculate the content cost
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image"""
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        if generated_image.shape != s:
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        # Get the outputs for the generated image
        outputs = self.model(generated_image)
        # Calculate the style cost
        style_cost = self.style_cost(outputs[:-1])
        # Calculate the content cost
        content_cost = self.content_cost(outputs[-1])
        # Calculate the total cost
        total_cost = self.alpha * content_cost + self.beta * style_cost
        return total_cost, content_cost, style_cost

    def compute_grads(self, generated_image):
        """
        Computes the gradients of the total cost with
        respect to the generated image
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        if generated_image.shape != s:
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        with tf.GradientTape() as tape:
            # Calculate the total cost
            tape.watch(generated_image)
            total, content, style = self.total_cost(generated_image)
        # Calculate the gradients
        grads = tape.gradient(total, generated_image)
        return grads, total, content, style

    def generate_image(
            self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """
        iterations - the number of iterations to perform gradient descent over
        step - if not None, the step at which you should print information
        about the training, including the final iteration:
            print Cost at iteration {i}: {J_total}, content {J_content},
                style {J_style}
            i is the iteration
            J_total is the total cost
            J_content is the content cost
            J_style is the style cost
        lr - the learning rate for gradient descent
        beta1 - the beta1 parameter for gradient descent
        beta2 - the beta2 parameter for gradient descent
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        # Initialize the generated image as a copy of the content image
        optimizer = tf.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)
        generated_image = tf.Variable(self.content_image)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations):
            with tf.GradientTape() as tape:
                grads, total, content, style = self.compute_grads(
                    generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            # Clip pixel values to maintain valid image range
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

            if step is not None and i % step == 0:
                print(f"Cost at iteration {i}: "
                      f"{total.numpy()}, content {content.numpy()}, "
                      f"style {style.numpy()}")

            if total.numpy() < best_cost:
                best_cost = total.numpy()
                best_image = generated_image
        # Removes the extra dimension from the image
        best_image = best_image[0]
        best_image.assign(tf.clip_by_value(best_image, 0, 1))
        return best_image.numpy(), best_cost