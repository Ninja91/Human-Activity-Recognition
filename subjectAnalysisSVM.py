# Author - Hariharan Seshadri #

import common
import math
import numpy as np
from sklearn import *
import scipy
from collections import Counter
import copy

print "\n"

#######################
# Parse the files #

X_train=common.parseFile('X_train.txt')				 
Y_train=common.parseFile('Y_train.txt')	
Y_train = Y_train.flatten()
subject_train = common.parseFile('subject_train.txt')
subject_train = subject_train.flatten()

X_test=common.parseFile('X_test.txt')			
Y_test=common.parseFile('Y_test.txt')
Y_test = Y_test.flatten()	

######################
# Hyper-parameters #

#top_N = 5 

#######################
# Pre-processing of data #

print "Computing means and covariances"

trainSubjects = [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]
requiredLabels = [4,5,6]

X_train = common.getPowerK( X_train, [1,2])

mean_array = []
cov_array = []
label_array = []

for i in trainSubjects:
	for j in requiredLabels:

		# Get subject info
		X_train_new , Y_train_new , subjectInfo= common.getSubjectData(X_train,Y_train,[i])

		# Get Data Subset #
		X_train_new , Y_train_new = common.getDataSubset(X_train, Y_train, requiredLabels)

		mean,cov = common.getDistribution(X_train_new,Y_train_new,j)
		mean_array.append(mean)
		cov_array.append(cov)
		label_array.append( j )

print "Done"


print "Pre_processing Training Data"

X_train , Y_train = common.getDataSubset(X_train, Y_train, requiredLabels)

featureArray = []  
trainSubjects = [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]

for i in xrange(len(X_train)):
	new_feature_1 = list(X_train[i])
	new_feature_2 =  [0]*len(trainSubjects)*len(requiredLabels)

	for j in xrange(len(mean_array)):
		distance = np.sqrt(np.sum((mean_array[j]-new_feature_1)**2))

		new_feature_2[j] = distance 

	#new_feature_2[ trainSubjects.index( int(subject_train[i]) )*3 + requiredLabels.index( int(Y_train[i]) ) ] = 1
	new_feature = list(new_feature_1) + list(new_feature_2)
	featureArray.append( new_feature )

X_train_expanded = np.asarray(featureArray)

print "Done"


print "Pre_processing Test Data"

X_test  = common.getPowerK( X_test, [1,2])  
X_test , Y_test = common.getDataSubset(X_test, Y_test, requiredLabels)

featureArray = []  

for i in xrange(len(X_test)):
	new_feature_1 = list(X_test[i])
	new_feature_2 =  [0]*len(trainSubjects)*len(requiredLabels)

	for j in xrange(len(mean_array)):
		distance = np.sqrt(np.sum((mean_array[j]-new_feature_1)**2))
		new_feature_2[j] = distance 

	#distance_copy = copy.deepcopy(new_feature_2)
	#distance_copy = np.sort( distance_copy )

	#normalizingSum = np.sum( np.array(distance_copy)[0:top_N] )

	# Encoding probabilities

	#for k in xrange(len(new_feature_2)):
	#	if new_feature_2[k] in list( np.array(distance_copy)[0:top_N] ):
	#		new_feature_2[k] = 1 - float(new_feature_2[k])/normalizingSum
	#	else:
	#		new_feature_2[k] = 0

	# Appending new features

	new_feature = list(new_feature_1) + list(new_feature_2)

	featureArray.append( new_feature )

X_test_expanded = np.asarray(featureArray)

print "Done"

print len(featureArray[0])

#######################
# Training an SVM#

print "Training an SVM"

#sample_weights = common.getSampleWeights(X_train,Y_train, requiredLabels)

clf = svm.SVC(kernel='linear')
clf.fit(X_train_expanded, Y_train) #,sample_weight = sample_weights                             
Y_predict=clf.predict(X_test_expanded)

print "Done"

#######################
# Check Accuracy

print "Checking accuracy"

precision,recall, f_score = common.checkAccuracy( Y_test , Y_predict , requiredLabels )
print f_score
		
confusionMatrix = common.createConfusionMatrix(Y_predict ,Y_test,requiredLabels)
print confusionMatrix

print "Done"

######################################################################

