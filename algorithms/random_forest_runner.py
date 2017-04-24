from sklearn import ensemble

from constants import NUMBER_OF_FOLDS
from splits.folds import foldX

def run(data, target, estimator, max_features):
    classifier = ensemble.RandomForestClassifier(n_estimators=estimator, max_features=max_features)
    return foldX(data, target, classifier, NUMBER_OF_FOLDS)