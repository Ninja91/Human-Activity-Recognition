# Author "Keerthi Raj Nagaraja"

# Importing libraries
from math import *
from numpy import *
#from sympy import Symbol,cos,sin
from operator import *
from numpy.linalg import *
import time
import ctypes
from sklearn import svm
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
Y_train=common.convertLabel(Y_train)
Y_test=common.convertLabel(Y_test)
print len(X_train), len(Y_train)
print len(X_test), len(Y_test), 
print X_test.flags
#X_dynamic, Y_dynamic=common.getDataSubset(X_train, Y_train, [1,2,3])
#X_nondynamic, Y_

#print len(X_sub_dynamic), len(Y_sub_dynamic)

clf = svm.SVC(cache_size=1000)
clf.fit(X_train, Y_train)
Y_predict=clf.predict(X_test)
print type(Y_predict), size(Y_predict), Y_predict
prec, rec, f_score=common.checkAccuracy(Y_test, Y_predict, [0,1])
print prec
print rec
print f_score
print clf.n_support_
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#shrinking=True, tol=0.001, verbose=False)
#clf.predict([[2., 2.]])
