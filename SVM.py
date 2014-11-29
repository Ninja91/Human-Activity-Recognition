# Author "Keerthi Raj Nagaraja"

# Importing libraries
from math import *
from numpy import *
#from sympy import Symbol,cos,sin
from operator import *
from numpy.linalg import *
import time
import ctypes
from sklearn import *
from collections import defaultdict
import common
from matplotlib import pyplot as plt
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

filename='../UCI HAR Dataset/' 							# Dataset Used (Change as per your computer's path)
#--------------------------------------------------------------------------------------------#

X_train=common.parseFile(filename+'train/X_train.txt')				# Read X Train 
Y_train=(common.parseFile(filename+'train/y_train.txt')).flatten()		# Read Y Train and flatten it to 1D array
X_test=common.parseFile(filename+'test/X_test.txt')				# Read X Test
Y_test=(common.parseFile(filename+'test/y_test.txt')).flatten()			# Read Y test and flatten it to 1D array

print len(X_train), len(Y_train)						# Printing Lengths of Train and Test Data
print len(X_test), len(Y_test)

X_dynamic_train, Y_dynamic_train=common.getDataSubset(X_train, Y_train, [1,2,3])	# Get Train sub data for [1,2,3]
X_nondynamic_train, Y_nondynamic_train=common.getDataSubset(X_train, Y_train, [4,5,6])  # Get Train sub data for [4,5,6]

X_dynamic_test, Y_dynamic_test=common.getDataSubset(X_test, Y_test, [1,2,3])		# Get Test sub data for [1,2,3]
X_nondynamic_test, Y_nondynamic_test=common.getDataSubset(X_test, Y_test, [4,5,6])	# Get Test sub data for [4,5,6]

X_nondynamic_train=common.getPowerK(X_nondynamic_train, [1,2])				# Convert X Train to X+X^2
X_nondynamic_test=common.getPowerK(X_nondynamic_test, [1,2])				# Convert X Test to X+X^2
#X_nondynamic_train_6, Y_nondynamic_train_6=common.getDataSubset(X_train, Y_train, [6]) # Used earlier to get Sub data for just 6th label
#X_nondynamic_test_6, Y_nondynamic_test_6=common.getDataSubset(X_test, Y_test, [6])
#Y_nondynamic_train_sublabels=common.convertLabel(Y_nondynamic_train, [4,5], [6])	# Used earlier to convert [4,5] to [1] and [6] to [0]
#Y_nondynamic_test_sublabels=common.convertLabel(Y_nondynamic_test, [4,5], [6])

print len(X_dynamic_train), len(Y_dynamic_train), Y_dynamic_train		# Printing lenghts and Labels extracted for verification
print len(X_nondynamic_train), len(Y_nondynamic_train), Y_nondynamic_train

sample_weights=common.getSampleWeights( X_nondynamic_train, Y_nondynamic_train , [4,5,6])	# Get sample weights for non-dynamic Data
#print sample_weights

################################################################################################
#Code used for Dynamic Data - Commented for now

'''clf = svm.LinearSVC(multi_class='crammer_singer')
clf.fit(X_dynamic_train, Y_dynamic_train)
Y_predict_dynamic=clf.predict(X_dynamic_test)
print type(Y_predict_dynamic), size(Y_predict_dynamic), Y_predict_dynamic
prec, rec, f_score=common.checkAccuracy(Y_dynamic_test, Y_predict_dynamic, [1,2,3])
print prec
print rec
print f_score
print common.createConfusionMatrix(Y_predict_dynamic, Y_dynamic_test, [1,2,3])
#print clf.n_support_'''
################################################################################################

# SVM Code for Linear Kernel with sample weights for non-dynamic classes [4,5,6]
clf = svm.SVC(kernel='linear')
clf.fit(X_nondynamic_train, Y_nondynamic_train, sample_weight=sample_weights) 		# Fit SVM using sample weights
Y_predict_nondynamic=clf.predict(X_nondynamic_test)					# Predict Labels for test data
print type(Y_predict_nondynamic), size(Y_predict_nondynamic), Y_predict_nondynamic	# Print Lenghts and predicted labels for verification
prec, rec, f_score=common.checkAccuracy(Y_nondynamic_test, Y_predict_nondynamic, [4,5,6]) # Check accuracy
print prec										# Print Precision, Recall and f-score
print rec
print f_score
print common.createConfusionMatrix(Y_predict_nondynamic, Y_nondynamic_test, [4,5,6])	# Print Confusion Matrix for the same
