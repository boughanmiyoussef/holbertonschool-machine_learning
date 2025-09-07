
#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import GPyOpt
import os
import matplotlib.pyplot as plt

# Prepare data MNIST
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_val = x_val.reshape(-1, 28 * 28)

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

# create best points
os.makedirs("checkpoints", exist_ok=True)

# function optimization
def train_model(params):
    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2 = float(params[0][3])
    batch_size = int(params[0][4])

    model = keras.Sequential([
        layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2), input_shape=(784,)),
        layers.Dropout(dropout),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = f"checkpoints/best_lr{learning_rate}_u{units}_d{dropout}_l2{l2}_b{batch_size}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0
    )
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0,
        callbacks=[checkpoint, early_stop]
    )

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    return -val_accuracy

# Search data
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
    {'name': 'units', 'type': 'discrete', 'domain': tuple(range(32, 257, 32))},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=train_model,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=True
)

optimizer.run_optimization(max_iter=30)

# save bayes_opt.txt
with open('bayes_opt.txt', 'w') as f:
    f.write("Best parameters found:\n")
    f.write(str(optimizer.X[np.argmin(optimizer.Y)]) + "\n")
    f.write("Best validation accuracy: {:.4f}\n".format(-optimizer.fx_opt))

# plot
plt.plot(-optimizer.Y)
plt.xlabel("Iteration")
plt.ylabel("Validation Accuracy")
plt.title("Bayesian Optimization Convergence")
plt.grid()
plt.savefig("convergence_plot.png")
plt.show()
