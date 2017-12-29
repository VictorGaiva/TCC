"""Getting some statistics"""
import wave
import os
import numpy as np
import feature_extraction as fe

#plotting
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


def get_audio_metadata(vinhetas_path, propagandas_path, talks_path):
    """
    Gets the statiscts about the audio files
    """
    statistics = {}
    statistics['vinheta'] = []
    statistics['propaganda'] = []
    statistics['talk'] = []

    #vinhetas
    for filename in os.listdir(vinhetas_path):
        with wave.open(vinhetas_path+filename, 'r') as audiofile:
            frames = audiofile.getnframes()
            rate = audiofile.getframerate()
            duration = frames / float(rate)
            statistics['vinheta'].append(duration)
    #propagandas
    for filename in os.listdir(propagandas_path):
        with wave.open(propagandas_path+filename, 'r') as audiofile:
            frames = audiofile.getnframes()
            rate = audiofile.getframerate()
            duration = frames / float(rate)
            statistics['propaganda'].append(duration)
    #talks
    for filename in os.listdir(talks_path):
        with wave.open(talks_path+filename, 'r') as audiofile:
            frames = audiofile.getnframes()
            rate = audiofile.getframerate()
            duration = frames / float(rate)
            statistics['talk'].append(duration)
    for key, val in statistics.items():
        print(key)
        val = np.array(val)
        print('\tSize:', len(val), 'files')
        print('\tTotal:', val.sum(), 'seconds')
        print('\tMin:', val.min(), 'seconds')
        print('\tMax:', val.max(), 'seconds')
        print('\tSTD:', val.std())
def get_features_metadata(features_path):
    """
    Produces some metadata about the features
    """
    feat = fe.open_atr_file(features_path+"/train_x.atr05")

    feat = np.transpose(feat)

    for val in feat:
        print('\tSize:', len(val), 'files')
        print('\tTotal:', val.sum(), 'seconds')
        print('\tMin:', val.min(), 'seconds')
        print('\tMax:', val.max(), 'seconds')
        print('\tSTD:', val.std())
        print("\n")

def get_batches_distribution(features_path, batch_name):
    """
    Plots a distribution of the values of each atribute in the batch files
    """
    feat = fe.open_atr_file(features_path)

    feat = np.transpose(feat)

    sns.set(color_codes=True)
    maxxx = 0
    i = 0
    for val in feat:
        i += 1
        print("Ploting", i)
        myplot = sns.distplot(val, hist=False)
        if max(val) > maxxx:
            maxxx = max(val)

        #setting axes
    myplot.set_xlim(left=0, right=maxxx)
    myplot.set_ylim(bottom=0, top=15)

        #making figure
    fig = myplot.get_figure()
    #plotting
    fig.savefig(batch_name + ".png")
    #cleaning
    fig.clf()


def main():
    """Main foo"""
    vinhetas_path = '../capturas/por classe/vinhetas/'
    propagandas_path = '../capturas/por classe/propagandas/'
    talks_path = '../capturas/por classe/talks/'
    #get_audio_metadata(vinhetas_path, propagandas_path, talks_path)
    #get_features_metadata("./output")
    get_batches_distribution("../dataset/batches/test_x.atr05", "test4")
    get_batches_distribution("../dataset/batches/validate_x.atr05", "validate4")
    get_batches_distribution("../dataset/batches/train_x.atr05", "train4")

if __name__ == '__main__':
    main()
