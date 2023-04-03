# English-Pronunciation-Neutrality-Classifier
This is a repository for the code of this published paper on the use of ANN to classify English pronunciation of Filipino call center agents as 'Neutral' or 'Not Neutral' for the company's standards.

## DATASET PREPARATION

110 recorded audio files (11 utterences of the 10 most commonly used words in the call center)

## FEATURE EXTRACTION

MFCCs extracted from the audio files, cleaned, and transformed into csv using the following Python libraries

* scipy
* python_speech_features
* regex
* pandas
* numpy

## NEURAL NETWORK TRAINING

ANN was trained and evaluated using the following Python libraries

* keras
* scikit-learn
* matplotlib

### Results of this experiment was published in an AI Journal: https://doi.org/10.4114/intartif.vol21iss62pp134-144
