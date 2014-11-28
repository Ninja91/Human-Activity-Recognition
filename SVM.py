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

filename='../UCI HAR Dataset/' # Dataset Used
#--------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#

X_train=common.parseFile(filename+'train/X_train.txt')
Y_train=(common.parseFile(filename+'train/y_train.txt')).flatten()
X_test=common.parseFile(filename+'test/X_test.txt')
Y_test=(common.parseFile(filename+'test/y_test.txt')).flatten()
#Y_train=common.convertLabel(Y_train)
#Y_test=common.convertLabel(Y_test)
print len(X_train), len(Y_train)
print len(X_test), len(Y_test), 
#print X_test.flags

X_dynamic_train, Y_dynamic_train=common.getDataSubset(X_train, Y_train, [1,2,3])
X_nondynamic_train, Y_nondynamic_train=common.getDataSubset(X_train, Y_train, [4,5])

X_dynamic_test, Y_dynamic_test=common.getDataSubset(X_test, Y_test, [1,2,3])
X_nondynamic_test, Y_nondynamic_test=common.getDataSubset(X_test, Y_test, [4,5])

X_nondynamic_train_6, Y_nondynamic_train_6=common.getDataSubset(X_train, Y_train, [6])
X_nondynamic_test_6, Y_nondynamic_test_6=common.getDataSubset(X_test, Y_test, [6])

print len(X_dynamic_train), len(Y_dynamic_train), Y_dynamic_train
print len(X_nondynamic_train), len(Y_nondynamic_train), Y_nondynamic_train

#class_weights=dict()
#class_weights[1]=1.0
#class_weights[2]=1.0
#class_weights[3]=1.0
'''clf = svm.LinearSVC(multi_class='crammer_singer')
clf.fit(X_dynamic_train, Y_dynamic_train)

#clf = neighbors.KNeighborsClassifier(50, weights='distance')
#clf.fit(X_dynamic_train, Y_dynamic_train)
Y_predict_dynamic=clf.predict(X_dynamic_test)
print type(Y_predict_dynamic), size(Y_predict_dynamic), Y_predict_dynamic
prec, rec, f_score=common.checkAccuracy(Y_dynamic_test, Y_predict_dynamic, [1,2,3])
print prec
print rec
print f_score
print common.createConfusionMatrix(Y_predict_dynamic, Y_dynamic_test, [1,2,3])
#print clf.n_support_'''

#clf = neighbors.KNeighborsClassifier(1, weights='distance')
#clf.fit(X_nondynamic_train, Y_nondynamic_train)
class_weights=dict()
class_weights[4]=2.5
class_weights[5]=1.0
clf = svm.LinearSVC(multi_class='crammer_singer', class_weight=class_weights)
clf.fit(X_nondynamic_train, Y_nondynamic_train)
Y_predict_nondynamic=clf.predict(X_nondynamic_test)
print type(Y_predict_nondynamic), size(Y_predict_nondynamic), Y_predict_nondynamic
prec, rec, f_score=common.checkAccuracy(Y_nondynamic_test, Y_predict_nondynamic, [4,5])
print prec
print rec
print f_score
print common.createConfusionMatrix(Y_predict_nondynamic, Y_nondynamic_test, [4,5])
#print clf.n_support_


#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#shrinking=True, tol=0.001, verbose=False)
#clf.predict([[2., 2.]])
