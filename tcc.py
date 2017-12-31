"""
Pipelines everything. From the segmentation, feature extration, preprocessing and training
"""
import segmentation as se
import batches_creation as bc
import classifier as cl
import feature_extraction as fe
import numpy as np

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
    Where the magic happens
    """
    labels_path = "../dataset/labels"
    audios_path = "../dataset/audios"
    features_path = "../dataset/features"
    batches_path = "../dataset/batches"

    #Extract the features into the folder
    print("Starting features extraction.")
    #se.features_from_folder(labels_path, audios_path, features_path)

    #make the batches
    #bc.make_batches(features_path, batches_path)

    #get datasets
    x_train = fe.open_atr_file(batches_path + "/train_x.atr05")
    y_train = fe.open_atr_file(batches_path + "/train_y.atr03")
    x_test = fe.open_atr_file(batches_path + "/validate_x.atr05")
    y_test = fe.open_atr_file(batches_path + "/validate_y.atr03")

    scores = []
    for wind in range(2, 20):
        #reshape
        x_train2, y_train2 = reshape(x_train, y_train, wind)
        x_test2, y_test2 = reshape(x_test, y_test, wind)

        #start network
        scores.append(cl.neural_network(
            100,                #epochs
            x_train2, y_train2, #train data
            x_test2, y_test2,   #test data
            [64, 64, 64, 32],   #layers specification
            128,                #batch size
            0.5                 #dropout
        ))
    myf = open("64_64_64_32.txt", "w")
    myf.write(str(scores))


if __name__ == '__main__':
    main()
