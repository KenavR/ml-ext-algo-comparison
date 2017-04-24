from sklearn import svm

from constants import NUMBER_OF_FOLDS
from splits.folds import foldX

def run(data, target):
    classifier = svm.SVC()
    return foldX(data, target, classifier, NUMBER_OF_FOLDS)


def runLinear(data, target):
    classifier = svm.LinearSVC()
    return foldX(data, target, classifier, NUMBER_OF_FOLDS)