import os
import pickle

from constants import DATASET_PATH
from images.features import extract

def load():
    extract()
    with open(DATASET_PATH + '/data.pickle', 'rb') as fr:
        data = pickle.load(fr)

    with open(DATASET_PATH + '/dataOpenCV_1D.pickle', 'rb') as fr:
        dataOpenCV_1D = pickle.load(fr)

    with open(DATASET_PATH + '/dataOpenCV_2D.pickle', 'rb') as fr:
        dataOpenCV_2D = pickle.load(fr)

    with open(DATASET_PATH + '/dataOpenCV_3D.pickle', 'rb') as fr:
        dataOpenCV_3D = pickle.load(fr)

    with open(DATASET_PATH + '/target.pickle', 'rb') as fr:
        target = pickle.load(fr)

    return [data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target]