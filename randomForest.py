# AUthor : Hariharan Seshadri #

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import common

print "Parsing"

X_train = common.parseFile( 'X_train.txt')
Y_train = common.parseFile( 'Y_train.txt')
Y_train = Y_train.flatten()
X_train,Y_train = common.getDataSubset(X_train, Y_train, [4,5,6])

X_test = common.parseFile( 'X_test.txt')
Y_test = common.parseFile( 'Y_test.txt')
Y_test= Y_test.flatten()
X_test,Y_Test = common.getDataSubset(X_test, Y_test, [4,5,6])


print "Done"

clf = RandomForestClassifier(n_estimators=50)
clf = clf.fit(X_train, Y_train)


print "Predicting"

predicted = []

for x_test in X_test:
	predicted.append( clf.predict(x_test)[0] )

print "Done"

print "Checking accuracy"


precision,recall,f_score = common.checkAccuracy( predicted , Y_test , [4,5,6] )

print f_score
