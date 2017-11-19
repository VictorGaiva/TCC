"""Extracts the features from the .wav files"""
import os
import csv
import sys
import librosa
from threading import Thread
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
    print(' ├-┐Opening file \'', filepath, '\'')
    audio_file, sample_rate = librosa.core.load(filepath)

    print(' | |Sample rate =', sample_rate)

    print(' | |Decomposing into overlaping frames')
    frames = librosa.util.frame(audio_file, 2048, 1536)

    i = 0
    total2 = len(frames)

    print(' | |Computing features')
    for frame in frames:
        #print stuff
        i = i + 1
        sys.stdout.write('\r | └-┐' + str(i) + '/' + str(len(frame)))
        sys.stdout.flush()
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
    print(' |Extracting features')
    feature_vector = get_features(input_path)

    print(' |Writing data to file : \"' + output_path + '\"')
    write_as_csv(output_path, feature_vector)
    print(' |Done.')

if __name__ == "__main__":
    print('Feature extractor')
    if len(sys.argv) < 2:
        print("Use: ", sys.argv[0], " <audio files directory> [option]")
        exit()
    #Diretorio de entrada
    input_directory = sys.argv[1]

    #Diretorio de saida
    output_directory = './csv/'

    Files = os.listdir(input_directory)
    total = len(Files)
    aux = 1
    jobs = []
    
    for filename in Files:
        print('Now processing file ' + str(aux) + '/' + str(total) + ': '+filename)
        newFile = "".join(filename.split('.')[:-1]) + '.csv'
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, newFile)
        aux = aux + 1
        #Computa atributos
        jobs.append(Thread(target=extract_and_save, args=(input_file, output_file)))
        jobs[0].start()
        jobs[0].join()
