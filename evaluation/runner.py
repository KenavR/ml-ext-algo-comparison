from algorithms.nearest_neighbors_runner import run as runKNN
from algorithms.random_forest_runner import run as runRandomForest
from algorithms.svc_runner import run as runSVC, runLinear as runLinearSVC
from algorithms.naive_bayes_runner import run as runNaiveBayes
from algorithms.perceptron_runner import run as runPerceptron
from algorithms.decision_tree_runner import run as runDecisionTree, runWithEntropy, runWithGini, runWithPrePruning
from printer.evaluation import printEvaluationRow


def runAndPrintAllClassifier(data, target):
    printEvaluationRow("KNN-3", *runKNN(data, target, ["uniform", "distance"], 3))
    printEvaluationRow("KNN-15", *runKNN(data, target, ["uniform", "distance"], 15))
    printEvaluationRow("KNN-30", *runKNN(data, target, ["uniform", "distance"], 30))

    estimators = [15, 40]
    features = ["sqrt", "log2", 3]

    for e in estimators:
        for f in features:
            printEvaluationRow('Random Forest ['+str(e)+'|'+str(f)+']', *runRandomForest(data, target, e, f))

    printEvaluationRow("Decision Tree Gini", *runWithGini(data, target))
    printEvaluationRow("Decision Tree Entropy", *runWithEntropy(data, target))
    printEvaluationRow("DT PP (0.15, 20, 5)", *runWithPrePruning(data, target, 0.15, 20, 5))
    printEvaluationRow("DT PP (0, 10, 3)", *runWithPrePruning(data, target, 0, 10, 3))
    printEvaluationRow("DT PP (0.3, 1, 10)", *runWithPrePruning(data, target, 0.3, 1, 10))

    printEvaluationRow("SVC", *runSVC(data, target))
    printEvaluationRow("LinearSVC", *runLinearSVC(data, target))
    printEvaluationRow("Naive Bayes", *runNaiveBayes(data, target))
    printEvaluationRow("Perceptron", *runPerceptron(data, target))
    return