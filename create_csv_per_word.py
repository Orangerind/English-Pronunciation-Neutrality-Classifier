#Programmed by: Rey Benjamin M. Baquirin
#This program takes in the whole dataset .csv file and creates separate .csv files corresponding to each word
#This will be used as the word's corresponding features to be inputted into a model. 
#The program also appends the ground truth for each audio file.
#This program's output are .csv files containing features and labels per word

import pandas as pd
import numpy as np

if __name__=='__main__':

	#Initialize variables
	num_per_word = 11
	holder = np.array([])
	dataset = pd.read_csv('dataset.csv', header=None)
	start = 0
	end = num_per_word - 1
	wordlist = ['actually','basically','broadband','computer','genie','internet','mobile','mobility','unfortunately','wireless']		

	for word in wordlist:	
		# Clear array after each file
		holder = np.empty((0,13))

		print("Starting parse for ", word)

		holder = np.append(holder, dataset.ix[start:end,:])
		# Reshape array to be array of arrays separated per file x 13 mfccs + ground truth
		holder = holder.reshape(num_per_word, 14)

		print("Saving dataset for", word, ".")
	
		np.savetxt(word + ".csv", holder, delimiter=",")

		print("Done.")
		
		#update index range
		start = end + 1
		end = start + 10
