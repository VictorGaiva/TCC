"""Extracts the features from the .wav files"""
import os
import csv
import sys
import time
from queue import Queue
from threading import Thread

import librosa


#import numpy as np
#import matplotlib.pyplot as plt

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
            spam_writer.writerow(newlist)

def get_features(filepath):
    """
    Returns a vector with the relevant atributs given an audioframe
    """
    features = []
    audio_file, sample_rate = librosa.core.load(filepath)
    frames = librosa.util.frame(audio_file, 2048, 1536)
    i = 0
    total2 = len(frames)
    for frame in frames:
        #print stuff
        i = i + 1
        #sys.stdout.write('\r' + str(i) + '/' + str(total2))
        #sys.stdout.flush()
        #default calc
        #Calc some stuff
        features.append(librosa.feature.chroma_stft(y=frame, sr=sample_rate))
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
    #post print stuff
    sys.stdout.write('\n')
    sys.stdout.flush()

    # Compute the spectral flux
    return features

def extract_and_save(input_path, output_path):
    """
    Extracts the features of given file and saves it into given output path
    """
    feature_vector = get_features(input_path)
    write_as_csv(output_path, feature_vector)

def worker():
    """
    Worker for processing the data
    """
    while True:
        item = filenames.get()

        if item is None:
            break

        file_size = os.path.getsize(item[0])
        file_size = file_size/1024
        if file_size > 1024:
            file_size = str(int(file_size/1024)) + 'MB'
        else:
            file_size = str(int(file_size)) + 'KB'

        print('Processing file', item[0], 'with', file_size, '\b.')

        now = time.time()

        extract_and_save(item[0], item[1])

        now = time.time() - now

        print('Finished file', item[0], 'in', int(now), 'seconds.\n')

        filenames.task_done()

filenames = Queue()

if __name__ == "__main__":
    print('Feature extraction')
    if len(sys.argv) < 2:
        print("Use: ", sys.argv[0], " <audio files directory> [option]")
        exit()
    #Diretorio de entrada
    input_directory = sys.argv[1]

    #Diretorio de saida
    output_directory = './csv/'

    files_list = os.listdir(input_directory)
    workers = []

    #Compute the names for the input and output
    for filename in files_list:
        newFile = "".join(filename.split('.')[:-1]) + '.csv'
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, newFile)
        filenames.put([input_file, output_file])

    workersCount = 2

    #Create thread objects
    for x in range(0, workersCount):
        w = Thread(target=worker)
        w.start()
        workers.append(w)

    #Wait for list to empty
    now = time.time()
    filenames.join()
    now = time.time() - now
    print(int(now), 'seconds.')

    #Create thread objects
    for x in range(0, workersCount):
        filenames.put(None)
