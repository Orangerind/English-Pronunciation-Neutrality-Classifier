#Programmed by: Rey Benjamin M. Baquirin
#This program takes in an input folder that contains 2 inner directories. 1st subdirectory is the word, 
#and the innermost subdirectory is the classification of that word. 
#Each audio file is read through scipy and a the corresponding mfcc features is extracted using python_speech_features with default kwargs
#each csv file contains a matrix given by frames x mfccs

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc

if __name__=='__main__':

	input_folder = "./Generated Dataset (fin)/"

	# Access the input directory
	for dirname in os.listdir(input_folder):

	    # Get the name of the subfolder
	    subfolder = os.path.join(input_folder, dirname)

	    for subname in os.listdir(subfolder):

	    	# Get the name of the subsubfolder
	    	s_subfolder = os.path.join(subfolder + "/", subname)

	    	# Initialize numpy array
	    	X = np.array([])

	    	# Iterate through all audio files and extract features 
	    	for filename in [x for x in os.listdir(s_subfolder) if x.endswith('.wav')]:
		        # Read the input file
		        filepath = os.path.join(s_subfolder, filename)
		        sampling_freq, audio = wavfile.read(filepath)

		        # Extract MFCC features. As per python_speech_features, features come in frames x numcep shape
		        print("Extracting MFCC for " + filename + "...")

		        mfcc_features = mfcc(audio, preemph=0.97,winlen=0.025, winstep=0.01, nfft=402, nfilt=26, appendEnergy=True, numcep=26, winfunc=np.hamming)

		        # Save MFCC features  
		        X = mfcc_features

		        # Output MFCC features for each audio file
		        np.savetxt(filename + "_mfcc_" + subname + ".csv", X, delimiter=",")

		        print("Completed.")
		        #print(filename, X.shape)

