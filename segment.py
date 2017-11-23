"""Extracts the features from the .wav files"""
import os
import csv
import sys

#import multiprocessing
import time

import librosa
import librosa.display

import numpy as np
import feature_extraction as fe

"""
152,149410	158,539943	vinheta
337,893384	342,110814	vinheta
490,718871	494,791427	vinheta
702,695017	725,232913	talk
725,232913	726,810425	vinheta
945,058387	947,859276	vinheta
1122,262871	1124,846449	vinheta
1350,583569	1355,364396	vinheta
1663,884670	1670,174596	vinheta
1896,720593	1928,451922	talk
2125,536275	2129,938821	vinheta
2401,359397	2404,329304	vinheta
2596,133822	2600,061504	vinheta
2883,410001	2886,126380	vinheta
3059,946457	3086,740014	talk
3087,589133	3088,913117	vinheta
3254,728931	3257,594207	vinheta
3498,269399	3504,088486	vinheta
"""
def get_labels(input_path):
    """
    Returns a dict with the start, end the class of each label from given file
    """
    with open(input_path, 'r') as labels_file:
        labels = []
        for line in labels_file:
            words = line.split('\t')
            lb_dict = {}
            lb_dict['start'] = float(words[0].replace(',', '.'))
            lb_dict['end'] = float(words[1].replace(',', '.'))
            lb_dict['class'] = words[2][:-1]
            labels.append(lb_dict)
    return labels


#Compute the names for the input and output
files_list = os.listdir("../capturas/testelabel")

#create list of arguments
for filename in files_list:
    for ln in get_labels(os.path.join("../capturas/testelabel", filename)):
        print(ln['start'])
