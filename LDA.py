#Author: Nitin A Jain

import numpy as np
from sklearn.lda import LDA
import common

def LDA_onFullDataset():
    #Parsing Full training dataset
    XFull = common.parseFile('../UCI HAR Dataset/train/X_train.txt')
    YFull = common.parseFile('../UCI HAR Dataset/train/y_train.txt')

    #Parsing Full testing dataset
    XFullTest = common.parseFile('../UCI HAR Dataset/test/X_test.txt')
    YFullTest = common.parseFile('../UCI HAR Dataset/test/y_test.txt')

    #Fitting data using LDA classifier
    clf = LDA()
    clf.fit(XFull, YFull.flatten())

    #Testing the results
    precision,recall,fscore = common.checkAccuracy(clf.predict(XFullTest),YFullTest,[1,2,3,4,5,6])
    print fscore

def LDA_onNonDynamicData():
    #Parsing Full training dataset
    XFull = common.parseFile('../UCI HAR Dataset/train/X_train.txt')
    YFull = common.parseFile('../UCI HAR Dataset/train/y_train.txt')

    #Parsing Full testing dataset
    XFullTest = common.parseFile('../UCI HAR Dataset/test/X_test.txt')
    YFullTest = common.parseFile('../UCI HAR Dataset/test/y_test.txt')

    #Getting the dataset associated with Non-Dynamic Activities on training 
    X_NonDynamic,Y_NonDynamic = common.getDataSubset(XFull,YFull.flatten(),[4,5,6])
    #Getting the dataset associated with Non-Dynamic Activities on testing
    X_NonDynamicTest,Y_NonDynamicTest = common.getDataSubset(XFullTest,YFullTest.flatten(),[4,5,6])

    #Fitting data using LDA classifier
    clf = LDA()
    clf.fit(X_NonDynamic, Y_NonDynamic.flatten())

    precision,recall,fscore = common.checkAccuracy(clf.predict(X_NonDynamicTest),Y_NonDynamicTest,[4,5,6])
    common.createConfusionMatrix(clf.predict(X_NonDynamicTest).flatten(),Y_NonDynamicTest.flatten(),[4,5,6])
    print fscore

    #Getting the dataset associated with Dynamic Activities on training 
    X_Dynamic,Y_Dynamic = common.getDataSubset(XFull,YFull.flatten(),[1,2,3])
    #Getting the dataset associated with Dynamic Activities on testing
    X_DynamicTest,Y_DynamicTest = common.getDataSubset(XFullTest,YFullTest.flatten(),[1,2,3])
    print len(X_DynamicTest),len(Y_DynamicTest)

    #Fitting data using LDA classifier
    clf = LDA()
    clf.fit(X_Dynamic, Y_Dynamic.flatten())

    precision,recall,fscore = common.checkAccuracy(clf.predict(X_DynamicTest),Y_DynamicTest,[1,2,3])
    common.createConfusionMatrix(clf.predict(X_DynamicTest).flatten(),Y_DynamicTest.flatten(),[1,2,3])

    print fscore

if __name__=='__main__':
    LDA_onFullDataset()
    LDA_onNonDynamicData()