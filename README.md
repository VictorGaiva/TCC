# TCC

The source files for each scrpit of my TCC done so far.

## feature_extraction.py

It has a set of routines for extracting features from audio files, saving them to disk as binary or csv files.
The most important functions so far are:
*_get\_features_
    which extracts 5 relevant features for each frame of given audio file.
*_write\_as\_bin_
    which saves the features to a '.atr$N' where $N is the number of features in the data given to the function.

## segment.py

It has a set of routines for getting labels from .txt file(using audacity format) and extracting features from these segments. It uses _feature\_extraction.py_ for extracting the features.
The most important functions so far are:
..*_get\_segments_
    which reads a .txt file containing the labels and timestamps for the segments, and returning them in a list of dicts
..*_features\_from\_label_
    which extracts the features from an audio file only on the segment given on the label

## stream_capture.py

It is an inline script to capture audio streams from the internet and saving them to a specified folder.
*It opens a thread for each link on the list
*The thread receives the stream and save it to file with the timestamp of the begining of the capture
*After 3600 seconds, it closes the file and opens a new one with updated timestamp
*The script stops itself after the defined time is over

## statistics_extraction.py

One time use script for extracting statistics from the data
