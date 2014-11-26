## Author: Hariharan Seshadri ##
## This program uses the KNN classifer to classify human activities ##


import numpy as np
from sklearn import neighbors, datasets
import common

###########################################################

print "Parsing"

X_train = common.parseFile( 'X_train.txt')
Y_train = common.parseFile( 'Y_train.txt')
Y_train = Y_train.flatten()

X_test = common.parseFile( 'X_test.txt')
Y_test = common.parseFile( 'Y_test.txt')
Y_test= Y_test.flatten()

print "Done"

print "Fitting Data"

ne = []
mean = []

for i in range(5,60,5):

	n_neighbors = i     ## Hyper - Parameter  ## 
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
	clf.fit(X_train, Y_train)

	print"Done"

	print "Predicting"

	predicted = []

	for x_test in X_test:
		predicted.append( clf.predict(x_test)[0] )

	print "Done"

	print "Checking accuracy"

	precision,recall,f_score = common.checkAccuracy( Y_test , predicted , [1,2,3,4,5,6]) # Must provide list of relavent labels #

	ne.append(i)
	mean.append( np.mean(f_score))
	

print mean
print "Done"
	
###########################################################


