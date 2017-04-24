import numpy as np

from images.dataset import load as loadImagesDataset
from loaders.digits import load as loadDigits
from loaders.iris import load as loadIris
from loaders.breast_cancer import load as loadBreastCancer
from evaluation.runner import runAndPrintAllClassifier
from printer.evaluation import printEvaluationHeader

[data, target] = loadDigits()
printEvaluationHeader("Digits", "digits|5Folds")
runAndPrintAllClassifier(data, target)
print('')
print('')
print('')

[data, target] = loadIris()
printEvaluationHeader("Iris", "iris|5Folds")
runAndPrintAllClassifier(data, target)
print('')
print('')
print('')

[data, target] = loadBreastCancer()
printEvaluationHeader("Breast Cancer", "cancer|5Folds")
runAndPrintAllClassifier(data, target)
print('')
print('')
print('')

[data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target] = loadImagesDataset()
printEvaluationHeader("Images Dataset", "images|5Folds")
runAndPrintAllClassifier(np.asarray(data), target)

printEvaluationHeader("Images Dataset 1D", "1D|5Folds")
runAndPrintAllClassifier(np.asarray(dataOpenCV_1D), target)

printEvaluationHeader("Images Dataset 2D", "2D|5Folds")
runAndPrintAllClassifier(np.asarray(dataOpenCV_2D), target)

printEvaluationHeader("Images Dataset 3D", "3d|5Folds")
runAndPrintAllClassifier(np.asarray(dataOpenCV_3D), target)