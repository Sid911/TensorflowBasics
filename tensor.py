import tensorflow as tf
from tensorflow import keras
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def createModel():
    md = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    md.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print(md.summary())

    return md


# Loading Data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating model and checkpoint
model = createModel()

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

try:
    model.load_weights(checkpoint_path)
    print("Model Loaded ☺")
except (ImportError, ValueError):
    print("No PreTrained Model found... Training starting 😵")

    model.fit(
        train_images, train_labels,
        epochs=15,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback]
    )

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=3)
print(f"Loss = {test_loss}, accuracy = {test_acc}")

inp = input("\nWould you like to save the model? : ")
if inp == "yes" or "Yes":
    savePath = "model.h5"
    print(f"Saving mode to '/{savePath}")
    model.save(filepath=savePath, overwrite=True, save_format="h5")
else:
    pass


