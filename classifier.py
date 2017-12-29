"""
Where everything happends
"""
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from feature_extraction import open_atr_file

def train(input_path, output, window_size):
    """
    Uses the batches in the input folder to train the NN
    """
    #get datasets
    x_train = open_atr_file(input_path + "/train_x.atr05")
    y_train = open_atr_file(input_path + "/train_y.atr03")
    x_test = open_atr_file(input_path + "/validate_x.atr05")
    y_test = open_atr_file(input_path + "/validate_y.atr03")

    #reshape
    x_train, y_train = reshape(x_train, y_train, window_size)
    x_test, y_test = reshape(x_test, y_test, window_size)

    #create the model
    model = models.Sequential()
    #init the layers
    model.add(layers.Dense(32, input_dim=len(x_train[0]), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
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
        epochs=50,
        batch_size=128
    )

    #
    score = model.evaluate(x_test, y_test, batch_size=128)

    print(score)

def reshape(datax, datay, window_size):
    """
    Reshapes the input
    """
    #fit the data
    while len(datax)%window_size != 0:
        datax = datax[:-1]
        datay = datay[:-1]

    #reshape datax
    new_datax = datax.flatten()
    new_datax = np.reshape(new_datax, (-1, len(datax[0])*window_size))

    #reshape datay
    new_datay = np.zeros(shape=(int(len(datay)/window_size), 3))
    for i in range(0, int(len(datay)/window_size)):
        new_datay[i] = datay[i*window_size]
    return [new_datax, new_datay]

def main():
    """
    Main
    """
    print("Nothing to be done")

if __name__ == '__main__':
    main()
