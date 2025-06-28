#!/usr/bin/env python3
"""
Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    The NST class performs tasks for Neural Style Transfer.
    
    Public Class Attributes:
    - style_layers: A list of layers to use for style extraction,
      default ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'].
    - content_layer: The layer to use for content extraction,
      default 'block5_conv2'.
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes an instance of NST.
        
        Parameters:
        - style_image (numpy.ndarray): The image used as a style reference.
        - content_image (numpy.ndarray): The image used as a content reference.
        - alpha (float): Weight for content cost. Default 1e4.
        - beta (float): Weight for style cost. Default 1.

        Raises:
        - TypeError: If style_image is not a numpy.ndarray with shape (h, w, 3),
          raises error with message "style_image must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If content_image is not a numpy.ndarray with shape (h, w, 3),
          raises error with message "content_image must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If alpha is not a non-negative number,
          raises error with message "alpha must be a non-negative number".
        - TypeError: If beta is not a non-negative number,
          raises error with message "beta must be a non-negative number".
          
        Instance Attributes:
        - style_image: The preprocessed style image.
        - content_image: The preprocessed content image.
        - alpha: Weight for content cost.
        - beta: Weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (float, int)) or beta < 0:
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
        Resizes an image so that its pixel values are between 0 and 1,
        and its largest side is 512 pixels.
        
        Parameters:
        - image (numpy.ndarray): A numpy.ndarray of shape (h, w, 3) containing the image to resize.
        
        Raises:
        - TypeError: If image is not a numpy.ndarray with shape (h, w, 3),
          raises error with message "image must be a numpy.ndarray with shape (h, w, 3)".
          
        Returns:
        - tf.Tensor: The resized image as a tf.Tensor with shape (1, h_new, w_new, 3),
          where max(h_new, w_new) == 512 and min(h_new, w_new) is scaled proportionally.
          The image is resized using bicubic interpolation, and pixel values are normalized
          from [0, 255] to [0, 1].
        """
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Resize the image using bicubic interpolation
        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Normalize pixel values to [0, 1]
        image_normalized = image_resized / 255

        # Clip values to ensure they remain within [0, 1]
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        # Add batch dimension along axis 0 and return
        return tf.expand_dims(image_clipped, axis=0)

    def load_model(self):
        """
        Loads the VGG19 model with AveragePooling2D layers instead of MaxPooling2D.
        """
        # Get VGG19 from Keras applications
        model_vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        model_vgg19.trainable = False

        # Selected layers
        selected_layers = self.style_layers + [self.content_layer]
        outputs = [model_vgg19.get_layer(name).output for name in selected_layers]

        # Build the model
        model = tf.keras.Model([model_vgg19.input], outputs)

        # Replace MaxPooling layers with AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5', custom_objects=custom_objects)
        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Computes the Gram matrix of a given tensor.
        
        Args:
        - input_layer: A tf.Tensor or tf.Variable of shape (1, h, w, c).
        
        Returns:
        - A tf.Tensor of shape (1, c, c) containing the Gram matrix of input_layer.
        """
        # Validate rank and batch size of input_layer
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4 or input_layer.shape[0] != 1:
            raise TypeError("input_layer must be a tensor of rank 4")
        
        # Compute Gram matrix: (batch, height, width, channels)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        
        # Normalize by number of positions (h*w), then return Gram tensor
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations

    def generate_features(self):
        """
        Extracts features used to calculate neural style cost.
        
        Sets the following public instance attributes:
        - gram_style_features - a list of Gram matrices computed from the style layers' outputs of the style image.
        - content_feature - the output of the content layer from the content image.
        """
        # Ensure images are properly preprocessed
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)

        # Get model outputs
        style_outputs = self.model(preprocessed_style)[:-1]

        # Set content feature, no additional processing needed
        self.content_feature = self.model(preprocessed_content)[-1]

        # Compute and set Gram matrices for style layer outputs
        self.gram_style_features = [self.gram_matrix(output) for output in style_outputs]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a specific layer
        
        Args:
            style_output: tf.Tensor of shape (1, h, w, c) containing the style output of the layer
            gram_target: tf.Tensor of shape (1, c, c) containing the target Gram matrix for this layer
            
        Returns:
            The style cost for this layer
        """
        # Input validation
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        # Extract c from style_output, not from gram_target
        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c)):
            raise TypeError(f"gram_target must be a tensor of shape [1, {c}, {c}]")

        # Compute generated Gram matrix
        gram_generated = self.gram_matrix(style_output)

        # Ensure types match
        gram_generated = tf.cast(gram_generated, dtype=gram_target.dtype)

        # Return mean squared difference
        return tf.reduce_mean(tf.square(gram_generated - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the total style cost across all layers
        
        Returns:
            float: Total style cost
        """
        len_s = len(self.style_layers)

        # Input validation
        if not isinstance(style_outputs, list) or len(style_outputs) != len_s:
            raise TypeError(f"style_outputs must be a list with length {len_s}")

        total_cost = 0.0
        weight = 1.0 / len_s

        for style_out, gram in zip(style_outputs, self.gram_style_features):
            layer_cost = self.layer_style_cost(style_out, gram)
            total_cost += weight * layer_cost

        return total_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost with proper normalization
        """
        scf = self.content_feature.shape

        # Input validation
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or content_output.shape != scf):
            raise TypeError(f"content_output must be a tensor of shape {scf}")

        # Compute mean squared difference
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost combining content and style
        
        Args:
            generated_image: TF Tensor of shape (1, H, W, 3)
            
        Returns:
            tuple: (total_cost, content_cost, style_cost)
        """
        s = self.content_image.shape

        # Validate generated image shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != s):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Preprocess for VGG19
        generated_preprocessed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255.0)

        # Feature extraction
        outputs = self.model(generated_preprocessed)
        style_outputs = outputs[:-1]  # Style layer outputs
        content_output = outputs[-1]  # Content layer output

        # Compute costs
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_total = self.alpha * J_content + self.beta * J_style

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Computes gradients of the total cost with respect to the generated image
        
        Args:
            generated_image: tf.Tensor of shape (1, H, W, 3)
            
        Returns:
            tuple: (gradients, J_total, J_content, J_style)
        """
        s = self.content_image.shape

        # Validate generated image shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != s):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Compute gradients with GradientTape
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """
        Generates the style-transferred image
        
        Args:
            iterations: Number of optimization steps
            step: Interval for logging
            lr: Learning rate
            beta1: Adam optimizer parameter
            beta2: Adam optimizer parameter
            
        Returns:
            best_image: Optimized image
            best_cost: Lowest total cost achieved
        """
        # Input validation
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0 <= beta1 <= 1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0 <= beta2 <= 1):
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialize generated image from content image
        generated_image = tf.Variable(self.content_image, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        # Optimization loop
        for i in range(iterations):
            with tf.GradientTape() as tape:
                J_total, J_content, J_style = self.total_cost(generated_image)

            # Update image
            gradients = tape.gradient(J_total, generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])

            # Clip pixel values between [0, 1]
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

            # Save best result so far
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            # Periodic logging
            if step and (i + 1) % step == 0:
                print(f"Iteration {i+1}: Cost={J_total}, Content={J_content}, Style={J_style}")

        return best_image[0], best_cost