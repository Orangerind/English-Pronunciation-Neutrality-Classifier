#Programmed by: Rey Benjamin M. Baquirin
#This program extracts full 26 MFCCs for all .WAV files contained in the directories. 
#The parent folder consists of an arbitrary number of folders named after the word they contain. 
#Each word folder then contains the .WAV files of the utterances whose MFCCs are extracted.

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc

if __name__=='__main__':

	input_folder = "./set1_unlabeled/"

	# Access the input directory
	for dirname in os.listdir(input_folder):

	    # Get the name of the subfolder
	    subfolder = os.path.join(input_folder, dirname)

	    # Initialize numpy array
	    X = np.array([])

	    # Iterate through all audio files and extract features 
	    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
		    # Read the input file
		    filepath = os.path.join(subfolder, filename)
		    sampling_freq, audio = wavfile.read(filepath)

		    # Extract MFCC features. As per python_speech_features
		    print("Extracting MFCC for " + filename + "...")

		    mfcc_features = mfcc(audio, preemph=0.97,winlen=0.025, winstep=0.01, nfft=402, nfilt=26, appendEnergy=True, numcep=26, winfunc=np.hamming)

		    # Save MFCC features  
		    X = mfcc_features

		    # Output MFCC features for each audio file
		    np.savetxt(filename + "_mfcc.csv", X, delimiter=",")

		    print("Completed.")
		    #print(filename, X.shape)

