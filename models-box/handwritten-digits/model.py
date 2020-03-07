"""
MNIST handwritten model.
- Main model
- CoreML converter
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.metrics import binary_accuracy, categorical_accuracy

import coremltools


CLASS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_dataset():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = np.expand_dims(train_x, axis=3).astype('float32')
    train_x /= 255
    test_x = np.expand_dims(test_x, axis=3).astype('float32')
    test_x /= 255
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return (train_x, train_y, test_x, test_y)

def build_model():
    model = Sequential(name='LeNet-5')
    model.add(Conv2D(6, (5,5), strides=(1,1), input_shape=(28,28,1), name='C1', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), name='S2'))
    model.add(Conv2D(16, (5,5), strides=(1,1), name='C3', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), name='S4'))
    model.add(Conv2D(120, (4,4), strides=(1,1), name='C5', activation='relu'))
    model.add(Flatten(name='C5-Flat'))
    model.add(Dense(120, name='C5-Dense', activation='relu'))
    model.add(Dense(84, name='F6'))
    model.add(Dense(10, activation='softmax', name='Output'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def load_model():
    train_x, train_y, test_x, test_y = load_dataset()
    model = build_model()
    print("Starting model training ...")
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=15, verbose=0)
    print(f"Finished training model.")
    print(f"Final training accuracy : {history.history['acc'][-1]}")
    print(f"Final training loss : {history.history['loss'][-1]}")
    print(f"Validation accuracy : {history.history['val_acc'][-1]}")
    print(f"Validation loss : {history.history['val_loss'][-1]}")
    return model

def generate_coreml_model():
    keras_model = load_model()
    coreml_model = coremltools.converters.keras.convert(
        keras_model,
        input_names="digits",
        image_input_names="digit_image",
        image_scale=1/255.0,
        class_labels=CLASS_LABELS
    )
    coreml_model.save("handwritten_digits.mlmodel")

if __name__ == '__main__' :
    generate_coreml_model()