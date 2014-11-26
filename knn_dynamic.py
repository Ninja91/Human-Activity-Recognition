## Author: Hariharan Seshadri ##
## This program uses the KNN classifer to distinguish between dynamic/non-dynamic activities ##

import numpy as np
from sklearn import neighbors, datasets
import common

###########################################################

print "Parsing"

X_train = common.parseFile( 'X_train.txt')
Y_train = common.parseFile( 'Y_train.txt')
Y_train = Y_train.flatten()
Y_train = common.convertLabel( Y_train )

X_test = common.parseFile( 'X_test.txt')
Y_test = common.parseFile( 'Y_test.txt')
Y_test= Y_test.flatten()
Y_test = common.convertLabel( Y_test )


print "Done"

print "Fitting Data"

ne = []
mean = []


for i in range(5,55,5):

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

	precision,recall,f_score = common.checkAccuracy( Y_test , predicted , [1,0]) # Must provide list of relavent labels #

	ne.append(i)
	mean.append( np.mean(f_score))
	

print mean
print "Done"
	
###########################################################


