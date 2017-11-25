"""Routines for extraction features from audio files and saving them to disk"""
import os
import csv
import sys

import time

import librosa
import librosa.display

import numpy as np

def extract_from_folder(input_directory, output_directory):
    """
    Extracts the features from all the .wav files in a dir and
    saves them in correspoding .csv files in the output folder
    """
    #Compute the names for the input and output
    files_list = os.listdir(input_directory)

    #create list of arguments
    args_list = []
    for filename in files_list:
        #if filename.split('.')[-1] != '.wav':
        #    print(filename, "is not a .wav file. Ignoring it.")
        #    continue

        #get the input and output paths in the right format
        input_file = os.path.join(input_directory, filename)
        new_file = "".join(filename.split('.')[:-1])
        output_file = os.path.join(output_directory, new_file)

        #extract features
        args_list.append([input_file, output_file])

    now = time.time()
    #start processing
    for argument in args_list:
        print("Processing data from", argument[0])
        ret = extract_and_save(argument[0], argument[1])
        print("Saved atributes into", argument[1] + ".atr" + format(ret, '02d'))

    print(int(time.time() - now), 'seconds.')

def extract_and_save(input_file, output_file):
    """
    Extracts the features of given file and saves it into given output path
    Returns the atributes depth for convinience
    """
    #open file
    audio_data, sample_rate = librosa.core.load(input_file)

    #get features
    feature_vector = get_features(audio_data, sample_rate, 2048, 512)

    #save to file
    write_as_bin(output_file, feature_vector)

    #return attribute count
    return len(feature_vector[0])

def get_features(data, sample_rate, frame_length=2048, hop_length=512):
    """
    Returns a ndarry with the atributes for each frame in the input file
    """

    #root-mean-square energy
    atr_rmse = librosa.feature.rmse(
        y=data,
        frame_length=frame_length,
        hop_length=hop_length
    ).flatten()#we are flattening so it's shape is (n,) and not (1, n)

    #zero crossing rate
    atr_zcr = librosa.feature.zero_crossing_rate(
        y=data,
        frame_length=frame_length,
        hop_length=hop_length
    ).flatten()

    #centroid
    atr_centroid = librosa.feature.spectral_centroid(
        y=data,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )#don't flatten yet

    #bandwidth
    atr_bandwidth = librosa.feature.spectral_bandwidth(
        y=data,
        n_fft=frame_length,
        hop_length=hop_length,
        centroid=atr_centroid
    ).flatten()
    #now you can
    atr_centroid = atr_centroid.flatten()

    #spectral rolloff
    atr_rolloff = librosa.feature.spectral_rolloff(
        y=data,
        n_fft=frame_length,
        hop_length=hop_length,
        roll_percent=0.85
    ).flatten()

    n_array = np.array([atr_zcr, atr_rmse, atr_centroid, atr_bandwidth, atr_rolloff]).transpose()
    return n_array

def write_as_csv(filepath, data):
    """
    Writes given data into the given file as a csv file
    """
    with open(filepath + '.csv', 'w') as csvfile:
        spam_writer = csv.writer(csvfile, delimiter=',', quotechar='\'')
        for line in data:
            spam_writer.writerow(line)

def write_as_bin(filepath, data):
    """
    Writes given data into the given file as a binary file
    The extension of the output file starts with '.atr' and ends with
    the number of atributes corresponding to onde frame
    """
    bin_data = data.flatten().tobytes()
    ext = '.atr' + format(len(data[0]), '02d')
    with open(filepath + ext, 'wb') as bin_file:
        bin_file.write(bin_data)

def main():
    """
    Main foo
    """
    print('Feature extraction')
    if len(sys.argv) < 2:
        print("Use: ", sys.argv[0], " <audio files directory> [option]")
        exit()
    #Diretorio de entrada
    extract_from_folder(sys.argv[1], './csv/')

if __name__ == "__main__":
    main()
