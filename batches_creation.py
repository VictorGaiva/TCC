"""
Routines for pre processing the data right before giving it to the neural network
"""
import json
import sys
import os
import random
import numpy as np

import feature_extraction as fe

def get_config():
    """
    Opens and validate the config.json file in this directory and returns it
    """
    with open("config.json") as jsonfile:
        try:
            configuration = json.load(jsonfile)
        except OSError:
            return None

    return configuration

def get_data(data_path=""):
    """
    Returns the train, validation a test datasets
    """
    if data_path == "":
        make_batches()
    else:
        print("Just open and return it.")
    return [0, 0, 0]

def make_batches(input_folder="./dataset", output_folder="./batches", config=None):
    """
    Creates each batch of the dataset
    """

    #try to get the configurations
    if not config:
        config = get_config()
        if not config:
            print("Couldn't load configuration file.")
            return None

    #check the existence of each folder
    if not os.path.isdir(input_folder):
        print("Folder", input_folder, "not found.")
        return -1
    if not os.path.isdir(output_folder):
        print("Folder", output_folder, "not found. Creating it.")
        try:
            os.makedirs(output_folder)
        except OSError:
            print("Error in creating output folder.")
            return -1
    #Config
    train = 0.7
    validate = 0.2
    test = 0.1
    tags = {
        "vinheta":   [1, 0, 0],
        "propaganda":[0, 1, 0],
        "talk":      [0, 0, 1]
    }
    my_seed = 42

    #list the files inside the input folder
    filenames = os.listdir(input_folder)

    #shuffle stuff around
    random.seed(my_seed)
    random.shuffle(filenames)

    #split the files in batches
    tr_batch = filenames[:int(len(filenames)*train)]
    vl_batch = filenames[int(len(filenames)*train):int(len(filenames)*(train+validate))]
    te_batch = filenames[int(len(filenames)*(train+validate)):]

    #make the batches
    train_x, train_y = make_batch(input_folder, tr_batch, tags)
    validate_x, validate_y = make_batch(input_folder, vl_batch, tags)
    test_x, test_y = make_batch(input_folder, te_batch, tags)

    #save them
    fe.write_as_bin(os.path.join(output_folder, "train_x"), train_x)
    fe.write_as_bin(os.path.join(output_folder, "train_y"), train_y)
    fe.write_as_bin(os.path.join(output_folder, "validate_x"), validate_x)
    fe.write_as_bin(os.path.join(output_folder, "validate_y"), validate_y)
    fe.write_as_bin(os.path.join(output_folder, "test_x"), test_x)
    fe.write_as_bin(os.path.join(output_folder, "test_y"), test_y)

def make_batch(folder_path, filenames, tags, config=None):
    """
    Given a list of files, makes a single batch with its data
    and returns it with its labels, set by the tags argument
    """
    #init
    labels_shape = (0, len(next(iter(tags.values()))))
    x_train = np.empty(shape=(0, 5))
    y_train = np.empty(shape=labels_shape)

    #for each file
    for filename in filenames:
        #get the features from it
        features = fe.open_atr_file(os.path.join(folder_path, filename))
        if features is None:
            print(filename, "not found.")
            continue

        #concatenate the new values on the batch
        x_train = np.vstack([x_train, features])

        #make an array with its label
        label = fe.label_from_filename(filename)
        labels = np.full(
            shape=(len(features), len(next(iter(tags.values())))),
            fill_value=tags[label]
        )
        y_train = np.vstack([y_train, labels])
    return [x_train, y_train]

def main():
    """
    Where we can test each function
    """
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(
            "Usage:\n\t$ python3",
            sys.argv[0],
            "<input_folder>",
            "<output_path>",
            "[configuration]"
        )
        return
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    make_batches(input_path, output_path)

if __name__ == '__main__':
    main()