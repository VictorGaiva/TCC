"""Extracts the features from the .wav files"""
import os
import sys

import librosa
import librosa.display

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

def features_from_folder(label_folder, audio_folder, output_folder):
    """
    This function reads all the label files from the label_folder, extract the features from the
    audio with matching name in audio_folder and saves all the features into the output_folder.
    """
    print('Listing label files from folder.')
    #scan labels folder
    labels_list = os.listdir(label_folder)
    label_files = []
    for filename in labels_list:
        #get its extension
        file_extension = filename.split('.')[-1]
        if file_extension != 'txt':
            continue
        #save to without its extension
        label_files.append(filename[:-4])

    print('Listing audio files from folder.')
    #scan audio folder
    audios_list = os.listdir(audio_folder)
    audio_files = []
    for filename in audios_list:
        #get its extension
        file_extension = filename.split('.')[-1]
        if file_extension != 'wav':
            continue
        #save to without its extension
        audio_files.append(filename[:-4])

    print('Removing files without matches')
    #use only the files with matching audio/label
    files_to_process = []
    for label_file in label_files:
        if label_file in audio_files:
            files_to_process.append(label_file)

    print('Processing each file...')
    i = 1
    class_count = {}
    total_f = len(files_to_process)
    #for each file
    for processing in files_to_process:
        print('File', str(i) + '/' + str(total_f))
        i += 1

        #
        label_file = os.path.join(label_folder, processing + ".txt")
        audio_file = os.path.join(audio_folder, processing + ".wav")

        #get the segments from the corresponding label file
        segments = get_segments(label_file)

        #
        total_s = len(segments)
        j = 1
        #for each segment
        for segment in segments:
            print('\tSegment', str(j) + '/' + str(total_s), segment['class'])
            j += 1

            if class_count.get(segment['class']) is None:
                class_count[segment['class']] = 1
            else:
                class_count[segment['class']] += 1
            output_filename = segment['class']
            output_filename += '-' + format(class_count[segment['class']], '04d')
            output_filename = os.path.join(output_folder, output_filename)

            #get its features
            segment_features = features_from_label(audio_file, segment)

            #save it to a file
            fe.write_as_bin(output_filename, segment_features)

def main():
    """
    main foo
    """
    if len(sys.argv) != 4:
        print(
            'Usage\n',
            'python3',
            sys.argv[0],
            '<label_folder>',
            '<audio_folder>',
            '<output_folder>'
        )
    else:
        features_from_folder(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()
