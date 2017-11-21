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
    with open(filepath, 'w') as csvfile:
        spam_writer = csv.writer(csvfile, delimiter=',', quotechar='\'')
        #todos atributos
        for line in data:
            newlist = []
            for val in line:
                newlist.append(val[0])
            spam_writer.writerow(line)


def get_features(filepath):
    """
    Returns a vector with the relevant atributs given an audioframe
    """
    features = []
    audio_file, sample_rate = librosa.core.load(filepath)
    #frames = librosa.util.frame(audio_file, 2048, 1536)
    features.append(librosa.feature.chroma_stft(y=audio_file, n_fft=2048, hop_length=512))
    #for frame in frames:
        #
        #features.append(librosa.feature.chroma_stft(y=frame, sr=sample_rate))
        #currEnergy = frame.energy()

        #Temporal features
        #frameMAX = currEnergy.max()
        #frameMEA = np.median(currEnergy)
        #frameRMS = frame.rms()
        #frameZCR = frame.zcr()

        #Spectral features
        #currCTD = currSpectr.centroid() #division by zero
        #currCHR = currSpectr.chroma()   #finda a good use to this
        #currFLT = currSpectr.flatness() #mean() got an unexpected keyword argument 'axis'
        #SpectMEA = currSpectr.mean()
        #SpectRFF = currSpectr.rolloff()
        #SpectKUR = currSpectr.kurtosis()

        #features.append([frameMAX, frameMEA, frameRMS, frameZCR, SpectMEA, SpectRFF, SpectKUR])
    return features

def extract_and_save(input_file, output_file):
    """
    Extracts the features of given file and saves it into given output path
    """
    feature_vector = get_features(input_file)
    write_as_csv(output_file, feature_vector)


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
        new_file = "".join(filename.split('.')[:-1]) + '.csv'
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
