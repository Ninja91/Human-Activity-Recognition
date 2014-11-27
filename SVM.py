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
#----------------------------------------------------------------------------------------------#
#This function reads the manual correspondences saved in a text file.
def readmatches(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\n'):
        rows.append(line.split(' '))
    rows.pop()
    #for loopVar1 in range(0, len(rows)):
    #	for loopVar2 in range(0, len(rows[loopVar1])):
    #		rows[loopVar1][loopVar2]=float(rows[loopVar1][loopVar2])
    return rows 

#--------------------------------------------------------------------------------------------#
#This function saves the homographies to a text file 
def save_matrix(filename, H):
	fo = open(filename, 'w', 0)
	for loopVar1 in range(H.shape[0]):
		for loopVar2 in range(H.shape[1]):
			fo.write(str(H[loopVar1, loopVar2]))
			if loopVar2!=H.shape[1]-1:		
				fo.write('\t')
		if loopVar1!=H.shape[0]-1:
			fo.write('\n')
	fo.close()	
		
#---------------------------------------------------------------------------------------------#

X_train=common.parseFile( filename+'train/X_train.txt')
Y_train=common.parseFile( filename+'train/y_train.txt')
Y_train = Y_train.flatten()
print len(X_train), len(Y_train)
X_sub_dynamic, Y_sub_dynamic=common.getDataSubset(X_train, Y_train, [1,2,3,4,5,6])
print len(X_sub_dynamic), len(Y_sub_dynamic)
#X = [[0, 0], [1, 1]]
#y = [0, 1]
#clf = svm.SVC()
#clf.fit(X, y)
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#shrinking=True, tol=0.001, verbose=False)
#clf.predict([[2., 2.]])
