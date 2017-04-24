from sklearn.naive_bayes import GaussianNB

from constants import NUMBER_OF_FOLDS
from splits.folds import foldX


def run(data, target):
    classifier = GaussianNB()
    return foldX(data, target, classifier, NUMBER_OF_FOLDS)