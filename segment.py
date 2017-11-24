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

def get_segments(input_path):
    """
    Returns a dict with the start, end the class of each label from given file.
    """
    with open(input_path, 'r') as segments_file:
        segments = []
        for line in segments_file:
            words = line.split('\t')
            sg_dict = {}
            sg_dict['start'] = float(words[0].replace(',', '.'))
            sg_dict['end'] = float(words[1].replace(',', '.'))
            sg_dict['class'] = words[2][:-1]
            segments.append(sg_dict)
    return segments

def features_from_labels(audio_file, segments):
    """
    For each label, extract the features from its segment and returns the list
    with the features from all of them.
    """
    segments_features = []
    #for each segment
    for segment in segments:
        features = features_from_label(audio_file, segment)
        #and append it to the list
        segments_features.append(features)
    return segments_features

def features_from_label(audio_file, segment):
    """
    Using the label, extract the features from the segment defined
    by the label.
    """
    duration = segment['end'] - segment['start']
    audio, sample_rate = librosa.core.load(
        audio_file,
        duration=duration,
        offset=segment['start']
    )
    features = fe.get_features(audio, sample_rate)
    return features


def main(input_path):
    """
    main foo
    """
    #Compute the names for the input and output
    files_list = os.listdir(input_path)

    #create list of .txt files
    label_files = []
    audio_files = []
    for filename in files_list:
        #get its extension
        file_extension = filename.split('.')[-1]
        #save in the right list

        if file_extension == 'txt':
            label_files.append(filename)
        elif file_extension == 'wav':
            audio_files.append(filename)
        elif not file_extension.startswith('atr'):
            print('Ignoring', filename, 'because of unrecognized extension.')

    #for each label file
    for label_file in label_files:
        #find matching .wav file
        filename = label_file.split('.')[-2]
        match = ''
        for audio_file in audio_files:
            if audio_file.split('.')[-2] == filename:
                match = audio_file
                break
        #found a match
        if match != '':
            #extract segments
            segments = get_segments(os.path.join(input_path, label_file))
            #get the names
            audio_file = os.path.join(input_path, match)
            bin_folder = os.path.join(input_path, 'features')
            total = len(segments)
            i = 1
            print('Extracting features from each segment in', match + '.')
            for segment in segments:
                print(str(i) + '/' + str(total))
                #get the features from the segment
                features = features_from_label(audio_file, segment)
                bin_filename = filename + '-' + segment['class'] + '-' + format(i, '02d')
                bin_path = os.path.join(bin_folder, bin_filename)
                #save them to file
                fe.write_as_bin(bin_path, features)
                i += 1
            print('Done with', match + '.')
        else:
            print('Couldn\'t find a match with file', label_file + '.')
    print('All done. Bye')

if __name__ == '__main__':
    main("../capturas/testelabel")
