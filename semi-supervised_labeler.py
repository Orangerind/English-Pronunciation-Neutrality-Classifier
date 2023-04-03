
#Programmed by: Rey Benjamin M. Baquirin
#This program classifies input utterances based on previously trained standard ANN models to label them as part of the semi-supervised learning
#It takes the output of flatten_mfcc_mean program and labels the files using a standard ANN .h5 model

# Silence recommendation for faster Tensorflow computations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import dependencies
import keras
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import metrics

# Reading the dataset
dataset = pd.read_csv('set1_unlabeled_mfcc_flat.csv')

# Save 26-dimensional MFCC as input vector for all instances
X = dataset.iloc[:, 1:27].values 
# Save ground truth for all instances 1 (Neutral) 0 (Not Neutral)
y = dataset.iloc[:, 27:].values 
# Reshape ground truth vectors to avoid error when split for y
c, r = y.shape
y = y.reshape(c,)

count = 0

# Save filenames to a list for printing later
filenames = dataset['File'].tolist()

# Loead the h5 model to be used for classification/labelling
loaded_model = load_model("baseline_model.h5")

print("\nClassifying...")
print("File Name \t\t\t Ground Truth")

# Predict/Classify the unlabeled files
prediction = loaded_model.predict(X)

for pred in prediction:
	if pred < 0.5:
		print(filenames[count],"\t | \t", 0)
	else:
		print(filenames[count],"\t | \t", 1)
	count += 1

print("Done.")