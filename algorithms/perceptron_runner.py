from sklearn.neural_network import MLPClassifier

from constants import SEED, NUMBER_OF_FOLDS
from splits.folds import foldX


def run(data, target):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = SEED)
    return foldX(data, target, classifier, NUMBER_OF_FOLDS)