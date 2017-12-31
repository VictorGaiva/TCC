"""
Where everything happends
"""
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from feature_extraction import open_atr_file

def supp_vec_machine():
    """
    Support vector machine implementation
    """
    return "damn"

def neural_network(epochs, x_train, y_train, x_test, y_test, neurons, batch_size, dropout):
    """
    Runs a training and testing session using given parameters
    """
    #create the model
    model = models.Sequential()

    #init the layers
    first_layer = neurons.pop(9)
    model.add(layers.Dense(first_layer, input_dim=len(x_train[0]), activation='relu'))
    model.add(layers.Dropout(dropout))

    for layer in neurons:
        model.add(layers.Dense(layer, activation='relu'))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(3, activation='softmax'))

    #opt
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    #compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )

    #
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    #
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    return score


def main():
    """
    Main
    """
    print("Nothing to be done")

if __name__ == '__main__':
    main()
