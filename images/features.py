import pickle
import cv2
import glob, os
import numpy as np

from PIL import Image
from sklearn import preprocessing

from constants import IMAGES_PATH, DATASET_PATH, IMAGE_GLOB


def extract():
    def _filesExists():
        fileNames = ['/data.pickle', '/dataOpenCV_1D.pickle', '/dataOpenCV_2D.pickle', '/dataOpenCV_3D.pickle',
                     '/target.pickle']
        for fn in fileNames:
            if not os.path.exists(DATASET_PATH + fn):
                return False
        return True

    def _getFiles(path):
        os.chdir(path)
        fileNames = glob.glob(IMAGE_GLOB)
        return [fileNames, []]

    def _labelData(fileNames):
        for fileName in fileNames:
            targetLabels.append(fileName[:fileName.index("/")])
        return targetLabels

    def _convertLabelsToIntegers(targetLabels):
        le = preprocessing.LabelEncoder()
        le.fit(targetLabels)
        return le.transform(targetLabels)

    def _extractPILFeatures(fileNames):
        data = []
        for index, fileName in enumerate(fileNames):
            imagePIL = Image.open(imagePath + "/" + fileName)
            imagePIL = imagePIL.convert('RGB')
            featureVector = imagePIL.histogram()

            data.append((featureVector))
        return data

    def _extractOpenCVFeatures(fileNames):
        dataOpenCV_1D = []
        dataOpenCV_2D = []
        dataOpenCV_3D = []

        flatten = lambda l: [item for sublist in l for item in sublist]

        for fileName in fileNames:
            imagePIL = Image.open(imagePath + "/" + fileName)
            imagePIL = imagePIL.convert('RGB')
            imageOpenCV = np.array(imagePIL)
            imageOpenCV = imageOpenCV[:, :, ::-1].copy()
            chans = cv2.split(imageOpenCV)
            colors = ("b", "g", "r")

            featuresOpenCV_1D = []
            bins_1D = 64
            for (chan, color) in zip(chans, colors):
                histOpenCV = cv2.calcHist([chan], [0], None, [bins_1D], [0, 256])
                featuresOpenCV_1D.extend(histOpenCV)
            featureVectorOpenCV_1D = flatten(featuresOpenCV_1D)

            dataOpenCV_1D.append(featureVectorOpenCV_1D)

            if (len(featureVectorOpenCV_1D) != bins_1D * 3):
                print("Feature vector has an incorrect length: " + str(len(featureVectorOpenCV_1D)) + " in file: " + fileName)

            featuresOpenCV_2D = []
            bins2D = 16
            featuresOpenCV_2D.extend(
                cv2.calcHist([chans[1], chans[0]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
            featuresOpenCV_2D.extend(
                cv2.calcHist([chans[1], chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
            featuresOpenCV_2D.extend(
                cv2.calcHist([chans[0], chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
            featureVectorOpenCV_2D = flatten(featuresOpenCV_2D)
            dataOpenCV_2D.append(featureVectorOpenCV_2D)

            featuresOpenCV_3D = cv2.calcHist([imageOpenCV], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            featureVectorOpenCV_3D = featuresOpenCV_3D.flatten()
            dataOpenCV_3D.append(featureVectorOpenCV_3D)
        return [dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D]

    def _writeDatasetToFile(data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target):
        with open(DATASET_PATH + '/data.pickle', 'wb') as fw:
            pickle.dump(data, fw)

        with open(DATASET_PATH + '/dataOpenCV_1D.pickle', 'wb') as fw:
            pickle.dump(dataOpenCV_1D, fw)

        with open(DATASET_PATH + '/dataOpenCV_2D.pickle', 'wb') as fw:
            pickle.dump(dataOpenCV_2D, fw)

        with open(DATASET_PATH + '/dataOpenCV_3D.pickle', 'wb') as fw:
            pickle.dump(dataOpenCV_3D, fw)

        with open(DATASET_PATH + '/target.pickle', 'wb') as fw:
            pickle.dump(target, fw)

    if _filesExists():
        return

    # Helper functions

    # Process
    # Download data from http://data.vicos.si/datasets/FIDS30/FIDS30.zip - save and extract into data/FIDS30 directory
    print("Start feature extraction")
    imagePath=IMAGES_PATH
    print("Start loading Files")
    fileNames, targetLabels = _getFiles(imagePath)
    print("Start Label Data")
    targetLabels = _labelData(fileNames)
    print("Start Converting Labels to Integer")
    target = _convertLabelsToIntegers(targetLabels)
    print("Start Extracting PIL Features")
    data = _extractPILFeatures(fileNames)
    print("Start Extracting OpenCV Features")
    dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D = _extractOpenCVFeatures(fileNames)
    print("Start Writing Dataset to Files")
    _writeDatasetToFile(data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target)

    return [data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target]