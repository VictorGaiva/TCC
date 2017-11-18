import wave
import os
import numpy as np

def main():
    vinhetas_path = '../capturas/por classe/vinhetas/'
    propagandas_path = '../capturas/por classe/propagandas/'
    talks_path = '../capturas/por classe/talks/'
    statistics = {}
    statistics['vinheta'] = []
    statistics['propaganda'] = []
    statistics['talk'] = []

    #vinhetas
    for filename in os.listdir(vinhetas_path):
        with wave.open(vinhetas_path+filename, 'r') as audiofile:
            frame = audiofile.readframes(1)
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
if __name__ == '__main__':
    main()
