from sklearn import datasets
from sklearn.utils import shuffle

from constants import SEED


def load():
    dataSet = datasets.load_iris()
    data, target = shuffle(dataSet.data, dataSet.target, random_state=SEED)
    return [data, target]