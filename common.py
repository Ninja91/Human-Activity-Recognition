## This file contains some data handling routines ##
## Authors: Hariharan Seshadri, Keerthi Nagaraj, Nitin Jain ##

import numpy as np
from sklearn import neighbors, datasets
from collections import defaultdict

###################################################################################
## Given a file name, returns the parsed file in the form of an array ##

def parseFile( file_name ):
	f = open(file_name)
	featureArray = []
	lines = f.readlines()
	for line in lines:
		feature_length = len(line.split(" "))

		raw_feature = line.split(" ")

		feature = []

		for index in xrange( feature_length ):
			try:
				feature.append( float( raw_feature[index] ))
			except:
				continue
		
		featureArray.append( feature )

	return np.asarray( featureArray )

###################################################################################
## Given two LISTS- original and predicted , returns the precision, accuracy, fscore

def checkAccuracy( original , predicted , labels ):
	TP = defaultdict(list)
	FP = defaultdict(list)
	FN = defaultdict(list)

	precision = []
	recall = []
	f_score = []
	
	for i in xrange(len(original)):
		
		if original[i] == predicted[i]:
			TP[str(int(original[i]))].append(1)			 

		elif original[i] != predicted[i]:
			FP[str(int(predicted[i]))].append(1)			 
			FN[str(int(original[i]))].append(1)			 

	
	for label in labels:
		p = float( len(TP[str(label)]) ) / ( len(TP[str(label)]) + len(FP[str(label)]))
		precision.append( p )

		r = float( len(TP[str(label)]) ) / ( len(TP[str(label)]) + len(FN[str(label)]))
		recall.append( r  )

		fs = float( 2*p*r ) / (p+r)				
		f_score.append( fs)

	return precision , recall , f_score


###################################################################################

## Distinguishes labels as Dynamic[1]/Non-Dynamic[0] ## 

def convertLabel( labels ):
	dynamic = []

	for label in labels:

		if label == 1 or label == 2 or label == 3:
			dynamic.append( 1 )
			 
		elif label == 4 or label == 5 or label == 6:
			dynamic.append( 0 )

	return np.asarray(dynamic)

###################################################################################
# This function takes in input 2D array of inputData(X) and 1D array of inputLabels(Y) and returns subset of those data which belong to requiredLabels(Ex: [1,4,5]). Required Labels is a 1D list. Returns 2D array of subData (X'), 1D array of subLabels(Y')
def getDataSubset(inputData, inputLabels, RequiredLabels): 
	subData=[]
	subLabels=[]
	for loopVar in range(len(inputLabels)):
		if inputLabels[loopVar] in RequiredLabels:
			subData.append(inputData[loopVar])
			subLabels.append(inputLabels[loopVar])
	return np.asarray(subData), np.asarray(subLabels)
###################################################################################
