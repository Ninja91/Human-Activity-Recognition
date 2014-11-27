## Author: Hariharan Seshadri ##
## This script tries to distinguish between SITTING,STANDING, and LAYING labels using Decision Trees ##

import numpy as np
from sklearn import tree
import common


print "Parsing"

X_train = common.parseFile( 'X_train.txt')
Y_train = common.parseFile( 'Y_train.txt')
Y_train = Y_train.flatten()
X_train,Y_train = common.getDataSubset(X_train, Y_train, [4,5,6])

X_test = common.parseFile( 'X_test.txt')
Y_test = common.parseFile( 'Y_test.txt')
Y_test= Y_test.flatten()
X_test,Y_test = common.getDataSubset(X_test, Y_test, [4,5,6])

print "Done"

print "Fitting Data"

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train) 

print "Done"



print "Predicting"

predicted = []

for x_test in X_test:
	predicted.append( clf.predict(x_test)[0] )


print "Done"

print "Checking accuracy"

precision,recall,f_score = common.checkAccuracy( Y_test , predicted , [4,5,6]) # Must provide list of relavent labels #

print f_score

print "Done"


