# Author - Hariharan Seshadri

import common
import math
import numpy as np
from sklearn import *

print "\n"

# Parse the files
X_train=common.parseFile('X_train.txt')				 
Y_train=(common.parseFile('Y_train.txt'))	
Y_train = Y_train.flatten()
X_test=common.parseFile('X_test.txt')			
Y_test=(common.parseFile('Y_test.txt'))
Y_test = Y_test.flatten()	


# Appropriate Subset of data is got here
X_train, Y_train=common.getDataSubset(X_train, Y_train, [4,5,6])	
X_test, Y_test=common.getDataSubset(X_test, Y_test, [4,5,6])	


# Accelerometer/Gyroscope Data is got here. COMMENT IF NOT NEEDED
X_train = common.getGyroFeatures( X_train, 'features.txt')  # X_train = common.getAccFeatures( X_train, 'features.txt')
X_test = common.getGyroFeatures( X_test, 'features.txt')    # X_test = common.getAccFeatures( X_test, 'features.txt')


# Blowing up the feature space. COMMENT IF NOT NEEDED 
X_train=common.getPowerK(X_train, [1,2])		
X_test=common.getPowerK(X_test, [1,2])			


# Weight samples are obtained here. COMMENT IF NOT NEEDED
sample_weights=common.getSampleWeights( X_train, Y_train , [4,5,6])	# Get sample weights for non-dynamic Data


# SVM Training, Prediction.
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)                             
Y_predict=clf.predict(X_test)					


# Check and print accuracy
prec, rec, f_score = common.checkAccuracy(Y_test, Y_predict, [4,5,6]) 
print f_score