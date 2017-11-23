"""Extracts the features from the .wav files"""
import os
import csv
import sys

import multiprocessing
import time

import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

#
FOO_ARGUMENTS = []

def write_as_csv(filepath, data):
    """
    Writes given data into the given file as a csv file
    """
    with open(filepath + '.csv', 'w') as csvfile:
        spam_writer = csv.writer(csvfile, delimiter=',', quotechar='\'')
        for line in data:
            spam_writer.writerow(line[0])

def write_as_bin(filepath, data):
    """
    Writes given data into the given file as a binary file
    The extension of the output file starts with '.atr' and ends with
    the number of atributes corresponding to onde frame
    """
    bin_data = np.array(data).tobytes()
    ext = '.atr' + format(len(data[0][0]), '02d')
    with open(filepath + ext, 'wb') as bin_file:
        bin_file.write(bin_data)

def get_features(filepath):
    """
    Returns a vector with the relevant atributs given an audioframe
    """
    audio_file, sample_rate = librosa.core.load(filepath)

    #Temporal features
    #currEnergy = frame.energy()

    #frameMAX = currEnergy.max()

    #frameMEA = np.median(currEnergy)

    #chroma
    #atr_chroma = librosa.feature.chroma_stft(
    #    y=audio_file,
    #    frame_length=2048,
    #    hop_length=512
    #)

    #root-mean-square energy
    atr_rmse = librosa.feature.rmse(
        y=audio_file,
        frame_length=2048,
        hop_length=512
    )

    #zero crossing rate
    atr_zcr = librosa.feature.zero_crossing_rate(
        y=audio_file,
        frame_length=2048,
        hop_length=512
    )

    #centroid
    atr_centroid = librosa.feature.spectral_centroid(
        y=audio_file,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512
    )

    #bandwidth
    atr_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_file,
        n_fft=2048,
        hop_length=512,
        centroid=atr_centroid
    )

    #currFLT = currSpectr.flatness() #mean() got an unexpected keyword argument 'axis'

    #SpectMEA = currSpectr.mean()

    #SpectRFF = currSpectr.rolloff()
    atr_rolloff = librosa.feature.spectral_rolloff(
        y=audio_file,
        n_fft=2048,
        hop_length=512,
        roll_percent=0.85
    )

    #SpectKUR = currSpectr.kurtosis()
    n_array = np.array([atr_zcr, atr_rmse, atr_centroid, atr_bandwidth, atr_rolloff]).transpose()
    return n_array

def extract_and_save(input_file, output_file):
    """
    Extracts the features of given file and saves it into given output path
    """
    feature_vector = get_features(input_file)
    write_as_bin(output_file, feature_vector)


def extract_from_folder(input_directory, output_directory):
    """
    Extracts the features from all the .wav files in a dir and
    saves them in correspoding .csv files in the output folder
    """
    #Compute the names for the input and output
    files_list = os.listdir(input_directory)

    #create list of arguments
    for filename in files_list:
        #if filename.split('.')[-1] != '.wav':
        #    print(filename, "is not a .wav file. Ignoring it.")
        #    continue
        #get the input and output paths in the right format
        input_file = os.path.join(input_directory, filename)
        new_file = "".join(filename.split('.')[:-1])
        output_file = os.path.join(output_directory, new_file)

        #extract features
        FOO_ARGUMENTS.append([input_file, output_file])
    #"""
    #start processing
    now = time.time()
    for argument in FOO_ARGUMENTS:
        print("Processing data from", argument[0])
        extract_and_save(argument[0], argument[1])
        print("Saved atributes from", argument[0], "into", argument[1])
    print(int(time.time() - now), 'seconds.')

    """
    #multiprocess (it has not speedup)
    workers = []
    for x in range(4):
        workers.append(multiprocessing.Process(target=worker, args=(x, 0)))

    now = time.time()
    #
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print(int(time.time() - now), 'seconds.')
    """


def worker(start_i, end_i):
    print(os.getpid(), ":Processing data from", FOO_ARGUMENTS[start_i][0])
    extract_and_save(FOO_ARGUMENTS[start_i][0], FOO_ARGUMENTS[start_i][1])
    print(os.getpid(), "Finished", FOO_ARGUMENTS[start_i][0],
          ". Saved to", FOO_ARGUMENTS[start_i][1])

if __name__ == "__main__":
    print('Feature extraction')
    if len(sys.argv) < 2:
        print("Use: ", sys.argv[0], " <audio files directory> [option]")
        exit()
    #Diretorio de entrada
    extract_from_folder(sys.argv[1], './csv/')
