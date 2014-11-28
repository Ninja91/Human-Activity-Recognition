## This file contains some data handling routines ##
## Authors: Hariharan Seshadri, Keerthi Nagaraj, Nitin Jain ##

import numpy as np
from sklearn import neighbors, datasets
from collections import defaultdict
import scipy

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

def convertLabel(labels, posLabels, Neglabels):
	dynamic = []

	for label in labels:

		if label in posLabels:
			dynamic.append( 1 )
			 
		elif label in Neglabels:
			dynamic.append( 0 )
		else:
			print "Unknown Label: Good Gawd :)"
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

#This function creates the Confusion Matrix for the predicted model
def createConfusionMatrix(predictedYLabels,originalYLabels,labelList):
    confusionMatrix = np.zeros((len(labelList),len(labelList)))
    #print len(predictedYLabels)

    if len(originalYLabels) != len(predictedYLabels):
        print 'Error'

    for i in xrange(len(originalYLabels)):
        if (predictedYLabels[i] not in labelList) or (originalYLabels[i] not in labelList):
            print 'Error'
        else:
            confusionMatrix[labelList.index(originalYLabels[i]),labelList.index(predictedYLabels[i])] = confusionMatrix[labelList.index(originalYLabels[i]),labelList.index(predictedYLabels[i])] + 1
    return confusionMatrix


#############################################################################

#This function returns the Mahalanobis distance between two given class 1 TO class 2 with respect to class 1's variance (i.e.) Mahalanobis Distance). NOTE: labels is a LIST containing only TWO labels.

def getMahalanobisDistance( X_train, Y_train, labels ):

	labelA = labels[0]
	labelB = labels[1]

	X_A , Y_A = getDataSubset(X_train, Y_train, [labelA])
	mean_A = np.mean(X_A,axis = 0)
	cov_A = np.cov(X_A,rowvar = 0)

	X_B , Y_B = getDataSubset(X_train, Y_train, [labelB])
	mean_B = np.mean(X_B,axis = 0)
	cov_B = np.cov(X_B,rowvar = 0)

	return scipy.spatial.distance.mahalanobis(mean_A, mean_B, cov_A) 
	
#############################################################################

#This function returns the Mean and Covariance of tha data of a particular label. 'label' requires a single number

def getDistribution( X_train, Y_train, label ):

	X_A , Y_A = getDataSubset(X_train, Y_train, [label])
	mean_A = np.mean(X_A,axis = 0)
	cov_A = np.cov(X_A,rowvar = 0)

	return mean_A,cov_A 

#############################################################################

#This function returns the sample weights based on HOW CLOSE THE SAMPLE IS TO THE MEAN OF IT'S CLASS , "labels" is a LIST that specifies the labels in Y_train

def getSampleWeights( X_train, Y_train , labels):
	
	sample_weights = []
	mean = []
	cov = []

	for i in labels:
		
		mean_A , cov_A = getDistribution( X_train , Y_train , i )

		mean.append( mean_A )
		
		cov.append( cov_A )	

		
	for i in xrange( len(X_train) ):

		index = labels.index( int(Y_train[i]) )
		this_mean = mean[ index ]
		this_cov = cov[ index ]

		weight = 	scipy.spatial.distance.mahalanobis(X_train[i], this_mean , this_cov)	

		weight = float(1) / weight

		sample_weights.append( weight )		
				
	return np.asarray(sample_weights)
	
#############################################################################

#This function returns the the input array whose entries are raised by power list 'k'. 

def getPowerK( X_features, k):

	X_features_new = []

	for x_feature in X_features:
		x_feature_new = []

		for power in k:
			for x_feature_dimension in x_feature:

				x_feature_new.append( np.power(x_feature_dimension,power) )

		X_features_new.append( np.asarray(x_feature_new) )				


	return np.asarray(X_features_new)
	
#############################################################################
