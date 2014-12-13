# Author - Hariharan Seshadri #

import common
import math
import numpy as np
from sklearn import *
import scipy
from collections import Counter


print "\n"

#######################
# Parse the files

X_train=common.parseFile('X_train.txt')				 
Y_train=(common.parseFile('Y_train.txt'))	
Y_train = Y_train.flatten()

X_test=common.parseFile('X_test.txt')			
Y_test=(common.parseFile('Y_test.txt'))
Y_test = Y_test.flatten()	

#######################
# Pre-processing of data

print "Pre_processing"

X_test , Y_test = common.getDataSubset(X_test, Y_test, [4,5])

X_train = common.getPowerK( X_train, [1,2])  
X_test  = common.getPowerK( X_test, [1,2])  

print "Done"

#######################

# These hold the information about the clusters for a SPECIFIC ACTIVITY of a SPECIFIC PERSON #
# TRAINING HERE#

print "Training"

trainSubjects = [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]
requiredLabels = [4,5]
mean_array = []
cov_array = []
label_array = []

for i in trainSubjects:
	for j in requiredLabels:
		X_train_new , Y_train_new , subjectInfo= common.getSubjectData(X_train,Y_train,[i])
		mean,cov = common.getDistribution(X_train_new,Y_train_new,j)
		mean_array.append(mean)
		cov_array.append(cov)
		label_array.append( j )

print "Done"
	
######################################################################

# TEST HERE #

print "Testing"

predicted = []
	
for i in xrange(len(X_test)):

	feature = X_test[i]

	distance_array = []
	for i in xrange(len(mean_array)):

		# Find Euclidean distance

		distance = np.sqrt(np.sum((mean_array[i]-feature)**2))
		distance_array.append(distance)

	distance_sorted , labels_sorted = (list(x) for x in zip(*sorted(zip(distance_array , label_array))))

	
	top_labels = list( np.array(labels_sorted)[0:5] )

	top_labels = Counter(top_labels)
	predicted.append( top_labels.most_common(1)[0][0] )
	
print "Done"

######################################################################
# Check Accuracy

print "Checking accuracy"

precision,recall, f_score = common.checkAccuracy( Y_test , predicted , requiredLabels )
print f_score
		
confusionMatrix = common.createConfusionMatrix(predicted ,Y_test,requiredLabels)
print confusionMatrix

print "Done"

######################################################################

