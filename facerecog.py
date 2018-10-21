# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:15:49 2018

@author: asuss
"""
# !/usr/bin/python
import os
import cv2
import re
import numpy
import timeit
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from skimage.feature import hog



def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def loadDataImages(path):
    images_data = []
    target_name = []
    for targetname in os.listdir(path):
        target_path = path + "/" + targetname

        for imagename in os.listdir(target_path):
            img = cv2.imread(os.path.join(target_path, imagename))
            #img = read_pgm(os.path.join(target_path, imagename), byteorder='<')
            if img is not None:
                images_data.append(img)
                target_name.append(targetname)
    return images_data, target_name


def preProcessing(image):
    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Haar Feature & Crop
    face_cascade = cv2.CascadeClassifier('Haar Classifier XML/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    for (x, y, w, h) in faces:
        global roi_gray
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

    # Resize
    new_w, new_h = 64,128
    resize_image = cv2.resize(roi_gray, (new_w, new_h))
    return resize_image


def HOGpython(image) :
    hog_desc, hog_image = hog(image, orientations=16, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), block_norm='L2',visualise=True)
    return hog_desc,hog_image


def HOG(image):
    winSize = (64,128)
    blockSize = (16,16)
    cellSize = (8,8)
    blockStride = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    h = hog.compute(image)
    return h


def MatrixCombine(imgvector, matrix):
    try:
        matrix = numpy.vstack((matrix, imgvector))
    except:
        matrix = imgvector
    return matrix

dataset_path_train = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\skripsi-face-recognition\\PyCharm\\Dataset\\LFW_2\\Train"
dataset_path_test = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\skripsi-face-recognition\\PyCharm\\Dataset\\LFW_2\\Test"
#dataset_path = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\PyCharm\\Dataset\\LFW"
#dataset_path = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\PyCharm\\Dataset\\ORL"
#X, y = loadDataImages(dataset_path)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=42)

X_train,y_train = loadDataImages(dataset_path_train)
X_test,y_test = loadDataImages(dataset_path_test)
print("-------------------------")

try :
    matrix_feature = None
    for img in X_train:
        pp_image = preProcessing(img)
        hog_feature,hog_image = HOGpython(pp_image)
        matrix_feature = MatrixCombine(hog_feature, matrix_feature)

    print(matrix_feature.shape)
    pca = PCA(n_components=10, svd_solver='auto',
              whiten=True).fit(matrix_feature)
    #mean, eigenvectors = cv2.PCACompute(matrix_feature, numpy.mean(matrix_feature, axis=0)

    X_train_hog = matrix_feature
    X_train_pca = pca.transform(matrix_feature)
    print(pca.explained_variance_.shape)
    print(pca.mean_.shape)
    print(pca.components_.shape)


    print("HOG Shape : " , X_train_hog.shape)
    print("HOG-PCA Shape : ",X_train_pca.shape)

    bdt_real_hog = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        learning_rate=1)

    bdt_real_pca = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        learning_rate=1)

    tic=timeit.default_timer()
    bdt_real_hog.fit(X_train_hog, y_train)
    toc=timeit.default_timer()
    hog_train_time =toc-tic

    tic=timeit.default_timer()
    bdt_real_pca.fit(X_train_pca, y_train)
    toc=timeit.default_timer()
    hog_pca_train_time =toc-tic
    # model = SVC()
    # model.fit(X_train, y_train)

    print("-------------------------")

    matrix_feature = None
    for img in X_test:
        pp_image = preProcessing(img)
        hog_feature, hog_image = HOGpython(pp_image)
        matrix_feature = MatrixCombine(hog_feature, matrix_feature)

    pca = PCA(n_components=10, svd_solver='auto',
              whiten=True).fit(matrix_feature)
    X_test_hog = matrix_feature
    X_test_pca = pca.transform(matrix_feature)
    print("HOG Shape : " , X_test_hog.shape)
    print("HOG-PCA Shape : ",X_test_pca.shape)

    tic=timeit.default_timer()
    hog_score = bdt_real_hog.score(X_test_hog, y_test)
    toc=timeit.default_timer()
    hog_test_time = toc-tic

    tic=timeit.default_timer()
    pca_score = bdt_real_pca.score(X_test_pca, y_test)
    toc=timeit.default_timer()
    hog_pca_test_time = toc-tic

    print("HOG-ADABOOST : " ,hog_score)
    print("Hog Train Time : ", hog_train_time)
    print("Hog Test Time : ", hog_test_time)

    print("HOG-PCA-ADABOOST : " ,pca_score)
    print("Hog-PCA Train Time : ",hog_pca_train_time)
    print("Hog-PCA Train Time : ",hog_pca_test_time)

    #print(hog_score)
    #print(hog_train_time)
    #print(hog_test_time)
    #print(pca_score)
    #print(hog_pca_train_time)
    #print(hog_pca_test_time)
except Exception:
    pass


