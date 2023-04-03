
#Programmed by: Rey Benjamin M. Baquirin
#This program trains a standard Artificial Neural Network for binary classification
#The model has an input layer of 26 nodes to accept the dataset created by flatten_mfcc_mean.py
#The input layer has 26 nodes, the  first hidden layer has 300 nodes, and the second hidden layer has 150 and an output layer with 1 node
#Training is implemented to stop at the minimum training loss and Cross validation comes in Stratified 10-fold Cross Validation
#Metrics captured were training accuracy, training loss, validation accuracy, validation loss, precision, recall, and the F1 Score

# Silence recommendation for faster Tensorflow computations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import dependencies
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils.vis_utils import plot_model

# Reading the dataset
dataset = pd.read_csv('baseline_mfcc_flat.csv')

# Save 26-dimensional MFCC as input vector for all instances
X = dataset.iloc[:, 1:27].values 
# Save ground truth for all instances 1 (Neutral) 0 (Not Neutral)
y = dataset.iloc[:, 27:].values 
# Reshape ground truth vectors to avoid error when split for validation
c, r = y.shape
y = y.reshape(c,)

# Initialize the ANN
ann_model = Sequential()
Input_layer = Dense(activation="relu", input_dim=26, units=300, kernel_initializer="random_normal",use_bias=True)
Hidden_layer = Dense(activation="relu", units=150, kernel_initializer="random_normal",use_bias=True)
Output_layer = Dense(activation="sigmoid", units=1, kernel_initializer="random_normal",use_bias=True)


# Add input layer and first hidden layer 
ann_model.add(Input_layer)

# Add second hidden layer
ann_model.add(Hidden_layer)

# Add the output layer
ann_model.add(Output_layer)

# Compile the ANN with SGD optimizer and binary cross entropy loss function, measuring accuracy
ann_model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ["accuracy"])

# Declare variables
# Variables for metrics
mod_loss = []
mod_vloss = []
mod_vacc= []
fold_f1 = []
fold_prec = []
fold_rec = []

# Variables for confusion matrix
conf_mat_shape = (2,2)
conf_mat = np.zeros(conf_mat_shape)
conf_mat_all = np.zeros(conf_mat_shape)

# Variables for identification of misclassified files
file_names = dataset.iloc[:, 0:1].values
misclass = []
misclass_files = []
i = 0;
probab = []

print("\nApplying Stratified 10-Fold Validation. Early stopping implemented based on minimum training loss.")

#Define F1 Score class
class ComputeF1(Callback):

	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		val_targ = self.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict)
		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		#print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
		return self.val_f1s, self.val_recalls, self.val_precisions

# Create callback to compute f1score
compute_f1score = ComputeF1()

# Create a stopping condition callback to stop epochs when training loss is minimized for that fold
stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='min')

# Create a random seed for use in validation initializing
seed = 7
np.random.seed(seed)

# Define stratified 10-fold cross validation with random seed and shuffle
tenfold_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Create loop for Stratified 10-fold cross validation and start training
for train, validation in tenfold_strat.split(X, y):

	i = i + 1
	print("-------------------------------------------------------------------------------------------------")
	print("Fold " , i)

	# Fit the model with 1000 epochs and batch size 4, save history to measure statistics, compute f1score using callbacks
	history = ann_model.fit(X[train], y[train], epochs=1000, batch_size=4, verbose=0,validation_data=(X[validation],y[validation]),callbacks=[stop,compute_f1score])

	# Validation score per fold
	y_true = y[validation]
	y_pred = ann_model.predict(X[validation])
	confmat = metrics.confusion_matrix(y_true,(y_pred>0.5).astype(int))

	# Compute average loss, accuracy, precision, and recall per fold by taking the mean across all epochs
	train_accuracy = np.asarray(history.history['acc']).mean()
	train_loss = np.asarray(history.history['loss']).mean()
	val_accuracy = np.asarray(history.history['val_acc']).mean()
	val_loss = np.asarray(history.history['val_loss']).mean()
	f1 = np.asarray(compute_f1score.val_f1s).mean()
	precision = np.asarray(compute_f1score.val_precisions).mean()
	recall = np.asarray(compute_f1score.val_recalls).mean()

	# Append all metrics to arrays for visualization later
	mod_loss.append(train_loss)
	mod_vloss.append(val_loss)
	mod_vacc.append(val_accuracy)
	fold_f1.append(f1)
	fold_rec.append(recall)
	fold_prec.append(precision)


	# Trace misclassified files based on the confusion matrix
	for j in range(len(validation)):
		# Append probabilities of each instance
		probab = np.append(probab, y_pred[j])

		if(y_pred[j] < 0.5):
			y_pred[j] = 0
		else:
			y_pred[j] = 1

		if (y_pred[j] != y_true[j]):
			misclass = np.append(misclass, validation[j])
		else:
			missclass = None

	misclass_files = np.append(misclass_files, misclass)

	misclass = np.empty(((confmat[0,1] + confmat[1,0]),0))
	conf_mat_all = np.add(conf_mat_all,confmat)

	# Print results of fold
	print("\nTraining Accuracy ==> ", train_accuracy , " | Training Loss ==> ", train_loss,
		  "\nValidation Accuracy ==> ", val_accuracy, " | Validation Loss ==> ", val_loss,
		  "\nRecall ==> ", recall , " | Precision ==> ", precision, "\nF1 Score ==> ", f1)


# Print out mean of 10 fold stratified validation and the standard deviation
print("-------------------------------------------------------------------------------------------------")
print("\nModel accuracy: ", np.mean(mod_vacc)*100, "(+/- ", np.std(mod_vacc)*100, ")")

# Print confusion matrix for all folds
print("\nConfusion Matrix after 10 folds: ")
print(conf_mat_all)

#Print list of misclassified files
print("\nMisclassified Files: ")

for indices in misclass_files:
	print(file_names[int(indices)])

#Print list of probabilities for each file
print("\nInstance Probabilities: ")
for ind in range(0, len(probab)):
	print(file_names[ind], probab[ind])	

# Save the model weights for future use
ann_model.save('modeltry.h5') 

# Print model architecture 
print(ann_model.summary())
plot_model(ann_model, to_file='model_plot_26.png', show_shapes=True, show_layer_names=False)

# Summarize history for loss
plt.plot(mod_loss, 'b', linewidth=0.6)
plt.plot(mod_vloss,'g', linewidth=0.6)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('10 Cross Fold')
plt.xticks(np.arange(10))
plt.legend(['Training Error', 'Validation Error'], loc='upper right')
plt.show()

# Summarize history for f1score
plt.plot(fold_f1, 'black', linewidth=1)
plt.plot(fold_prec,'g', linewidth=0.6)
plt.plot(fold_rec, 'b', linewidth=0.6)
plt.title('F1 Score, Precision, and Recall')
plt.ylabel('Score')
plt.xlabel('10 Cross Fold')
plt.xticks(np.arange(10))
plt.legend(['F1 Score', 'Precision', 'Recall'], loc='lower right')
plt.show()