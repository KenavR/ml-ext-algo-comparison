from sklearn import neighbors

from constants import NUMBER_OF_FOLDS
from splits.folds import foldX


def run(data, target, weights, n):
    for w in weights:
        classifier = neighbors.KNeighborsClassifier(n, weights=w)
        return foldX(data, target, classifier, NUMBER_OF_FOLDS)