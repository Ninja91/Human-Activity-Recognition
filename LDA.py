#Author: Nitin A Jain

import numpy as np
from sklearn.lda import LDA
import common

XFull = common.parseFile('./UCI HAR Dataset/train/X_train.txt')
YFull = common.parseFile('./UCI HAR Dataset/train/y_train.txt')

XFullTest = common.parseFile('./UCI HAR Dataset/test/X_test.txt')
YFullTest = common.parseFile('./UCI HAR Dataset/test/y_test.txt')

clf = LDA()
clf.fit(XFull, YFull.flatten())

precision,recall,fscore = common.checkAccuracy(clf.predict(XFullTest),YFullTest,[1,2,3,4,5,6])
print fscore