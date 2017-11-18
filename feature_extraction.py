"""Extracts the features from the .wav files"""
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def GetFeaturesPymir(audioData, feature):
    """
    Returns a vector with the relevant atributs given an audioframe
    """
    import pymir
    windowSize = 512
    attrList = ['frameMAX', 'frameMEA', 'frameRMS', 'frameZCR', 'SpectMEA', 'SpectRFF', 'SpectKUR']

    features = []
    print('    Decoposing into frames')
    fixedFrames = wavData.frames(windowSize, np.hamming)

    i = 0
    total = len(fixedFrames)
    if feature == -1:
        print('    Calculating features for each frame')
    else:
        print('    Calculating \''+attrList[feature]+'\' for each frame')
    for frame in fixedFrames:
        #print stuff
        i = i + 1
        sys.stdout.write('\r      ' + str(i) + '/' + str(total))
        sys.stdout.flush()
        #default calc
        if feature == -1:
            #Calc some stuff
            currSpectr = frame.spectrum()
            currEnergy = frame.energy()

            #Temporal features
            frameMAX = currEnergy.max()
            frameMEA = np.median(currEnergy)
            frameRMS = frame.rms()
            frameZCR = frame.zcr()

            #Spectral features
            #currCTD = currSpectr.centroid() #division by zero
            #currCHR = currSpectr.chroma()   #finda a good use to this
            #currFLT = currSpectr.flatness() #mean() got an unexpected keyword argument 'axis'
            SpectMEA = currSpectr.mean()
            SpectRFF = currSpectr.rolloff()
            SpectKUR = currSpectr.kurtosis()

            features.append([frameMAX, frameMEA, frameRMS, frameZCR, SpectMEA, SpectRFF, SpectKUR])
        #specific features
        else:
            #0 e 1
            if feature == 0 or feature == 1:
                currEnergy = frame.energy()
                if feature == 0:
                    features.append(currEnergy.max())
                if feature == 1:
                    features.append(np.median(currEnergy))
            #2 e 3
            elif feature == 2:
                features.append(frame.rms())
            elif feature == 3:
                features.append(frame.zcr())
            #4, 5 e 6
            else:
                currSpectr = frame.spectrum()
                if feature == 4:
                    features.append(currSpectr.mean())
                elif feature == 5:
                    features.append(currSpectr.rolloff())
                elif feature == 6:
                    features.append(currSpectr.kurtosis())
    #post print stuff
    sys.stdout.write('\n')
    sys.stdout.flush()

    # Compute the spectral flux
    return features

if __name__ == "__main__":
    print('Feature extractor')
    if len(sys.argv) < 2:
        print("Use: ", sys.argv[0], " <audio files directory> [option]")
        exit()
    #Diretorio de entrada
    input_directory = sys.argv[1]
    #Diretorio de saida
    output_directory = './csv/' + "".join(input_directory.split('/')[1:])

    feature = -1
    if len(sys.argv) > 2:
        #Sobreescreve arquivos de saida ja existentes
        if sys.argv[2] == '-w':
            option = 1
        #Faz um append de atributo em cada linha da saida
        elif sys.argv[2] == '-a':
            if len(sys.argv) != 4:
                print("ERROR:Must specify unique feature when\
                       appending new data to an existing file.")
                exit()
            option = 2
        #atributo especifico ou todos?
        if len(sys.argv) == 4:
            feature = int(sys.argv[3])

        #Escreve todos atributos caso arquivo nao exista
        else:
            option = 0
    #Escreve todos atributos caso arquivo nao exista
    else:
        option = 0

    Files = os.listdir(input_directory)
    total = len(Files)
    i = 1
    for filename in Files:
        print('Now processing file ' + str(i) + '\\' + str(total) + ': '+filename)
        i = i + 1
        newFile = "".join(filename.split('.')[:-1]) + '.csv'
        #Se a opcao for '0' e o arquivo de saida ja exista, pule este arquivo
        if option == 0 and os.path.isfile(os.path.join(output_directory, newFile)):
            print('File already processed. Jumping to next file.\n')
            continue

        #Abertura do arquivo de audio
        print('  Opening file')
        wavData = pymir.AudioFile.open(os.path.join(input_directory, filename))

        #Escrita normal
        if option != 2:
            #Computa atributos
            print('  Extracting features')
            featureVector = GetFeaturesPymir(wavData, feature)
            print('  Writing data to file : \"' + newFile + '\"')

            with open(os.path.join(output_directory, newFile), 'w') as csvfile:
                spamWriter = csv.writer(csvfile, delimiter=',', quotechar='\"')
                #todos atributos
                if feature == -1:
                    spamWriter.writerow(attrList)
                    spamWriter.writerows(featureVector)
                #apenas 1 atributo
                else:
                    #escreve cabecalho
                    spamWriter.writerow(attrList[feature])
                    #escreve dados
                    spamWriter.writerows(featureVector)

        #append
        else:
            #Computa atributos
            print('  Extracting features')
            featureVector = GetFeaturesPymir(wavData, feature)

            print('  Appending new feature to file : \"' + newFile + '.tmp\"')
            #open existing file
            with open(os.path.join(output_directory, newFile), 'r') as csvfile1:
                #open temporary file
                with open(os.path.join(output_directory, newFile+'.tmp'), 'w') as csvfile2:
                    spamWriter = csv.writer(csvfile2, delimiter=',', quotechar='\"')
                    firstRow = True
                    #para cada linha do arquivo original
                    for row in csv.reader(csvfile1):
                        #se for o header, adicione o nome da nova coluna
                        if firstRow:
                            firstRow = False
                            rowCp = row[:]
                            rowCp.append(attrList[feature])
                            spamWriter.writerow(rowCp)
                        #se for o resto, adicione o atributo
                        else:
                            #remove o primeiro elemento e coloca ele no final da row atual
                            rowCp = row[:]
                            rowCp.append(featureVector.pop(0))
                            spamWriter.writerow(rowCp)
                    #renomeia os arquivos
