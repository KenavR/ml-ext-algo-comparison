from sklearn import tree

from splits.folds import foldX


def runWithGini(data, target):
    return run(data, target)

def runWithEntropy(data, target):
    return run(data, target, "entropy")

def runWithPrePruning(data, target, minWeigthFractionLeaf, minSamplesLeaf, maxDepth):
    return run(data, target, "gini", minWeigthFractionLeaf, minSamplesLeaf, maxDepth)

# parameter zum durchreichen an sklearn implementierung
# standard values minimum
def run(data, target, criterion="gini", minWeightFractionLeaf=0, minSamplesLeaf=1, maxDepth=None):
    classifier = tree.DecisionTreeClassifier(criterion=criterion, min_weight_fraction_leaf=minWeightFractionLeaf,
                                             min_samples_leaf=minSamplesLeaf, max_depth=maxDepth)
    return foldX(data, target, classifier, 5)
