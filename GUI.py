import sys
import os
import timeit
import traceback
import re

import cv2
import numpy

from Preprocessing import Preprocessing
from HOGMethod import HOGMethod
from PCAMethod import PCAMethod
from AdaBoostMethod import AdaBoostMethod
from MethodParam import MethodParam

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, qRgb


#param untuk training
#param = MethodParam(64,128,8,2,16,20,10)
preprocessing = Preprocessing(64,128)
hog_training_testing = HOGMethod(16,2,4)
pca_training_testing = PCAMethod(10)
adaboost_training_testing = AdaBoostMethod(10)

#prameter metode hog+Adaboost
#param_hog_adaboost = MethodParam(64,128,8,2,4,10,15)
hog_ha = HOGMethod(8,2,4)
pca_ha = PCAMethod(10)
adaboost_ha = AdaBoostMethod(15)

#parameter metode hog+pca+Adaboost
#param_hog_pca_adaboost = MethodParam(64,128,8,2,16,20,10)
hog_hpa = HOGMethod(8,2,16)
pca_hpa = PCAMethod(20)
adaboost_hpa = AdaBoostMethod(10)


ROOT_PATH = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\skripsi-face-recognition\\PyCharm"

#dataset ORL
dataset_path_train = ROOT_PATH+"\\Dataset\\ORL v2\\Train"
dataset_path_test = ROOT_PATH+"\\Dataset\\ORL v2\\Test"

#dataset caltech
#dataset_path_train = ROOT_PATH+"\\Dataset\\Caltech\\Train"
#dataset_path_test = ROOT_PATH+"\\Dataset\\Caltech\\Test_Classifier"
#dataset_path_test = ROOT_PATH+"\\Dataset\\Caltech\\Test_Contrast\\Biasa"

#test_path
test_path = ROOT_PATH+"\\Test"
test_path_hog_adaboost = ROOT_PATH+"\\Test\\HOG_AdaBoost"
test_path_hog_pca_adaboost = ROOT_PATH+"\\Test\\HOG_PCA_AdaBoost"

#model pas training dan testing
model_path = ROOT_PATH+"\\Model"

#model Caltech
model_path_hog_adaboost_caltech = ROOT_PATH+"\\Model\\Hasil Uji Coba\\Caltech\\Revisi sesudah pra-sidang\\Uji Coba Classifier\\Final\\HOG+AdaBoost (8,16,4,10,15)"
model_path_hog_pca_adaboost_caltech = ROOT_PATH+"\\Model\\Hasil Uji Coba\\Caltech\\Revisi sesudah pra-sidang\\Uji Coba Classifier\\Final\\HOG+PCA+AdaBoost(8,16,16,20,10)"

#model ORL
model_path_hog_adaboost_ORL = ROOT_PATH+"\\Model\\Hasil Uji Coba\\ORL\\20 v2\\HOG+ADABOOST"
model_path_hog_pca_adaboost_ORL = ROOT_PATH+"\\Model\\Hasil Uji Coba\\ORL\\20 v2\\HOG+PCA+ADABOOST"

class GUI(QDialog) :
    def __init__(self):

        super(GUI, self).__init__()
        loadUi('GUI.ui',self)
        self.setWindowTitle('Face Recognition System')
        self.trainButton.clicked.connect(self.on_trainButton_clicked)
        self.loadImageButton_hogAdaBoost.clicked.connect(self.on_loadImageButton_hogAdaBoost_clicked)
        self.preProcessingButton_hogAdaBoost.clicked.connect(self.on_preProcessingButton_hogAdaBoost_clicked)
        self.loadImageButton_hogpcaAdaBoost.clicked.connect(self.on_loadImageButton_hogpcaAdaBoost_clicked)
        self.preProcessingButton_hogpcaAdaBoost.clicked.connect(self.on_preProcessingButton_hogpcaAdaBoost_clicked)
        self.hogButton_hogAdaBoost.clicked.connect(self.on_hogButton_hogAdaBoost_clicked)
        self.hogButton_hogpcaAdaBoost.clicked.connect(self.on_hogButton_hogpcaAdaBoost_clicked)
        self.pcaButton_hogpcaAdaBoost.clicked.connect(self.on_pcaButton_hogpcaAdaBoost_clicked)
        self.adaboostButton_hogAdaBoost.clicked.connect(self.on_adaboostButton_hogAdaBoost_clicked)
        self.adaboostButton_hogpcaAdaBoost.clicked.connect(self.on_adaboostButton_hogpcaAdaBoost_clicked)
        self.datasetComboBox.addItem("Caltech")
        self.datasetComboBox.addItem("ORL")

    @pyqtSlot()
    def on_trainButton_clicked(self):
        self.datasetComboBox.setEnabled(False)
        self.trainButton.setEnabled(False)
        dataset = self.datasetComboBox.currentText()
        try :
            f = open(os.path.join(test_path, 'dataset.txt'), 'w')
            f.write(dataset)
            f.close()
        except Exception :
            print(traceback.format_exc())
            pass

        #print("--Load Dataset--")
        #X_train, y_train = loadDataImages(dataset_path_train)
        #X_test, y_test = loadDataImages(dataset_path_test)
        ##X, y = loadDataImages(dataset_path)
        ##X_train, X_test, y_train, y_test = train_test_split(
        ##   X, y, test_size=0.3, random_state=40)

        #trainingProcess(X_train,y_train)
        #testingProcess(X_test,y_test)

        if dataset == "Caltech" :
            result_hog_adaboost = numpy.loadtxt(os.path.join(model_path_hog_adaboost_caltech, 'Result_Model.txt'))
            result_hog_pca_adaboost = numpy.loadtxt(os.path.join(model_path_hog_pca_adaboost_caltech, 'Result_Model.txt'))
        else :
            result_hog_adaboost = numpy.loadtxt(os.path.join(model_path_hog_adaboost_ORL, 'Result_Model.txt'))
            result_hog_pca_adaboost = numpy.loadtxt(os.path.join(model_path_hog_pca_adaboost_ORL, 'Result_Model.txt'))
        f1_ha = result_hog_adaboost[0]
        ha_train_time = result_hog_adaboost[1]
        ha_test_time = result_hog_adaboost[2]

        f1_hpa = result_hog_pca_adaboost[3]
        hpa_train_time = result_hog_pca_adaboost[4]
        hpa_test_time = result_hog_pca_adaboost[5]
        self.hasil_hogAdaBoost.setText(
            'Accuracy Score : ' + str(f1_ha * 100) + '%, Training Time : ' + str(ha_train_time) + 's, Testing Time : ' + str(
                ha_test_time)+'s')
        self.hasil_hogpcaAdaBoost.setText(
            'Accuracy Score : ' + str(f1_hpa * 100) + '%, Training Time : ' + str(hpa_train_time) + 's, Testing Time : ' + str(
                hpa_test_time)+'s')
        self.param_hogAdaBoost.setText("Resize Image = (" + str(preprocessing.resize_width) + ","
                                       + str(preprocessing.resize_height) + "), Cell = (" + str(
           hog_ha.cell_size) + "," + str(hog_ha.cell_size)
                                       + "), Block = (" + str(
            hog_ha.block_size * hog_ha.cell_size) + "," + str(
            hog_ha.block_size * hog_ha.cell_size) + "), Bins = "
                                       + str(hog_ha.bins) + ", Eigen Vektor = " + str(
            pca_ha.k_dimensions) + ", Iterasi = " + str(adaboost_ha.iterations))
        self.param_hogpcaAdaBoost.setText("Resize Image = (" + str(preprocessing.resize_width) + ","
                                          + str(preprocessing.resize_height) + "), Cell = (" + str(
            hog_hpa.cell_size) + "," + str(hog_hpa.cell_size)
                                          + "), Block = (" + str(
            hog_hpa.block_size * hog_hpa.cell_size) + "," + str(
            hog_hpa.block_size * hog_hpa.cell_size) + "), Bins = "
                                          + str(hog_hpa.bins) + ", Eigen Vektor = " + str(
            pca_hpa.k_dimensions) + ", Iterasi = " + str(adaboost_hpa.iterations))
        self.loadImageButton_hogAdaBoost.setEnabled(True)
        self.loadImageButton_hogpcaAdaBoost.setEnabled(True)



    @pyqtSlot()
    def on_loadImageButton_hogAdaBoost_clicked(self) :
        try:
            fileName = QFileDialog().getOpenFileName()
            imagePath = str(fileName[0])
            image = cv2.imread(imagePath)
            image_test = cv2.resize(image,(259,200))
            image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
            image_test = numpy.require(image_test, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(image_test))
            cv2.imwrite(os.path.join(test_path_hog_adaboost, "test.jpg"), image)
            self.loadImageView_hogAdaBoost.setPixmap(pixmap)
            self.preProcessingButton_hogAdaBoost.setEnabled(True)
        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_loadImageButton_hogpcaAdaBoost_clicked(self):
        try:
            fileName = QFileDialog().getOpenFileName()
            imagePath = str(fileName[0])
            image = cv2.imread(imagePath)
            image_test = cv2.resize(image, (259, 200))
            image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
            image_test = numpy.require(image_test, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(image_test))
            cv2.imwrite(os.path.join(test_path_hog_pca_adaboost, "test.jpg"), image)
            self.loadImageView_hogpcaAdaBoost.setPixmap(pixmap)
            self.preProcessingButton_hogpcaAdaBoost.setEnabled(True)
        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_preProcessingButton_hogAdaBoost_clicked(self):
        try:
            image_test = cv2.imread(os.path.join(test_path_hog_adaboost, "test.jpg"))
            dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
            if(dataset=="Caltech") :
                greyscale = preprocessing.greyScale(image_test)
            else :
                greyscale = image_test

            haarFeature_image = preprocessing.haarFeatureClassifier(greyscale)
            resize_image = preprocessing.resizeImage(haarFeature_image)
            resize_image = numpy.require(resize_image, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(resize_image))

            cv2.imwrite(os.path.join(test_path_hog_adaboost, "preprocess_test.jpg"), resize_image)
            self.preProcessingView_hogAdaBoost.setPixmap(pixmap)
            self.hogButton_hogAdaBoost.setEnabled(True)
        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_preProcessingButton_hogpcaAdaBoost_clicked(self):
        try:
            image_test = cv2.imread(os.path.join(test_path_hog_pca_adaboost, "test.jpg"))
            dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
            if (dataset == "Caltech"):
                greyscale = preprocessing.greyScale(image_test)
            else:
                greyscale = image_test
            haarFeature_image = preprocessing.haarFeatureClassifier(greyscale)
            resize_image = preprocessing.resizeImage(haarFeature_image)
            resize_image = numpy.require(resize_image, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(resize_image))

            cv2.imwrite(os.path.join(test_path_hog_pca_adaboost, "preprocess_test.jpg"), resize_image)
            self.preProcessingView_hogpcaAdaBoost.setPixmap(pixmap)
            self.hogButton_hogpcaAdaBoost.setEnabled(True)
        except Exception:
            print(traceback.format_exc())
            pass


    @pyqtSlot()
    def on_hogButton_hogAdaBoost_clicked(self):
        try :
            image_test = cv2.imread(os.path.join(test_path_hog_adaboost, "preprocess_test.jpg"),0)
            hog_feature,hog_image = hog_ha.hogImage(image_test)

            numpy.savetxt(os.path.join(test_path_hog_adaboost, 'hog_feature_test.txt'), hog_feature, fmt='%s')
            hog_image = hog_ha.visualizeHog(hog_image)
            hog_image = numpy.require(hog_image, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(hog_image))

            cv2.imwrite(os.path.join(test_path_hog_adaboost, "hog.jpg"), hog_image)
            self.hogView_hogAdaBoost.setPixmap(pixmap)
            self.adaboostButton_hogAdaBoost.setEnabled(True)

        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_hogButton_hogpcaAdaBoost_clicked(self):
        try:
            image_test = cv2.imread(os.path.join(test_path_hog_pca_adaboost, "preprocess_test.jpg"), 0)
            hog_feature, hog_image = hog_hpa.hogImage(image_test)

            numpy.savetxt(os.path.join(test_path_hog_pca_adaboost, 'hog_feature_test.txt'), hog_feature, fmt='%s')
            hog_image = hog_hpa.visualizeHog(hog_image)
            hog_image = numpy.require(hog_image, numpy.uint8, 'C')
            pixmap = QPixmap.fromImage(toQImage(hog_image))

            cv2.imwrite(os.path.join(test_path_hog_pca_adaboost, "hog.jpg"), hog_image)
            self.hogView_hogpcaAdaBoost.setPixmap(pixmap)
            self.pcaButton_hogpcaAdaBoost.setEnabled(True)

        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_pcaButton_hogpcaAdaBoost_clicked(self):
        try :
            self.pcaViewList_hogpcaAdaBoost.clear()
            hog_feature = numpy.loadtxt(os.path.join(test_path_hog_pca_adaboost, 'hog_feature_test.txt'))
            dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
            if (dataset == "Caltech") :
                pca = joblib.load(os.path.join(model_path_hog_pca_adaboost_caltech, "Model_PCA.sav"))
            else :
                pca = joblib.load(os.path.join(model_path_hog_pca_adaboost_ORL, "Model_PCA.sav"))

            pca_feature = pca.transform(hog_feature.reshape(1,-1)) #reshape karena datanya cuma 1
            numpy.savetxt(os.path.join(test_path_hog_pca_adaboost, "pca_feature_test.txt"),pca_feature, fmt='%s')
            for i in range(pca_feature.shape[1]) :
                self.pcaViewList_hogpcaAdaBoost.addItem(str(pca_feature[0,i]))

            self.adaboostButton_hogpcaAdaBoost.setEnabled(True)
        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_adaboostButton_hogAdaBoost_clicked(self):
        try :
            self.faceClassList_hogAdaBoost.clear()
            feature = numpy.loadtxt(os.path.join(test_path_hog_adaboost, 'hog_feature_test.txt'))
            dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
            if (dataset == "Caltech"):
                classifier = numpy.load(os.path.join(model_path_hog_adaboost_caltech, 'HOG_AdaBoost_Model.npy'))
            else :
                classifier = numpy.load(os.path.join(model_path_hog_adaboost_ORL, 'HOG_AdaBoost_Model.npy'))
            names = adaboost_ha.testAdaBoost(feature,classifier)
            for i in range(len(names)) :
                self.faceClassList_hogAdaBoost.addItem(str(names[i,0])+" - " +str(names[i,1]))

        except Exception:
            print(traceback.format_exc())
            pass

    @pyqtSlot()
    def on_adaboostButton_hogpcaAdaBoost_clicked(self):
        try:
            self.faceClassList_hogpcaAdaBoost.clear()
            feature = numpy.loadtxt(os.path.join(test_path_hog_pca_adaboost, 'pca_feature_test.txt'))
            dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
            if (dataset == "Caltech"):
                classifier = numpy.load(os.path.join(model_path_hog_pca_adaboost_caltech, 'HOG_PCA_AdaBoost_Model.npy'))
            else :
                classifier = numpy.load(os.path.join(model_path_hog_pca_adaboost_ORL, 'HOG_PCA_AdaBoost_Model.npy'))

            names = adaboost_hpa.testAdaBoost(feature, classifier)
            for i in range(len(names)):
                self.faceClassList_hogpcaAdaBoost.addItem(str(names[i, 0]) + " - " + str(names[i, 1]))

        except Exception:
            print(traceback.format_exc())
            pass

def toQImage(im, copy=False):
    if im is None:
        return QImage()

    if im.dtype == numpy.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable([qRgb(i, i, i) for i in range(256)])
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim


def loadDataImages(path):
    images_data = []
    target_name = []
    dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
    for targetname in os.listdir(path):
        target_path = path + "/" + targetname
        for imagename in os.listdir(target_path):
            if dataset == "Caltech" :
                img = cv2.imread(os.path.join(target_path, imagename))
            else :
                img = read_pgm(os.path.join(target_path, imagename), byteorder='<')
            if img is not None:
                images_data.append(img)
                target_name.append(targetname)
    return images_data, target_name

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

def trainingProcess(img_data,label) :
    print("--Training--")
    result = numpy.zeros(shape=(6,1),dtype=numpy.float32)
    hog_matrix = mainProcess(img_data)
    #train classifier pca dengan vektor fitur dari hog
    pca = pca_training_testing.pcaImage(hog_matrix)
    print(pca.explained_variance_.shape)
    print(pca.mean_.shape)
    print(pca.components_.shape)
    #simpen ke file
    joblib.dump(pca, os.path.join(model_path, "Model_PCA.sav"))
    #ubah vektor fitur awal dengan classifier pca sebelumnya
    pca_matrix = pca.transform(hog_matrix)

    X_train_HA = hog_matrix
    X_train_HPA = pca_matrix

    print("HOG data train shape : ", X_train_HA.shape)
    print("HOG-PCA data train shape : ", X_train_HPA.shape)

    tic = timeit.default_timer()
    HA_model, HA_list = adaboost_training_testing.trainAdaBoost(X_train_HA, label)
    toc = timeit.default_timer()
    _ha_train_time_ = toc - tic
    result[1] = _ha_train_time_
    numpy.savetxt(os.path.join(model_path, 'HOG_AdaBoost_Model.txt'), HA_model, fmt='%s')
    numpy.save(os.path.join(model_path, 'HOG_AdaBoost_Model.npy'), HA_list)

    tic = timeit.default_timer()
    HPA_model, HPA_list = adaboost_training_testing.trainAdaBoost(X_train_HPA, label)
    toc = timeit.default_timer()
    _hpa_train_time_ = toc - tic
    result[4] = _hpa_train_time_
    numpy.savetxt(os.path.join(model_path, 'HOG_PCA_AdaBoost_Model.txt'), HPA_model, fmt='%s')
    numpy.save(os.path.join(model_path, 'HOG_PCA_AdaBoost_Model.npy'), HPA_list)

    numpy.savetxt(os.path.join(model_path, 'Result_Model.txt'), result, fmt='%0.3f')


def testingProcess(img_data,label) :
    try :
        print("--Testing--")

        result = numpy.loadtxt(os.path.join(model_path, 'Result_Model.txt'))

        hog_matrix = mainProcess(img_data)
        #load classsifier pca dr proses training
        pca = joblib.load(os.path.join(model_path, "Model_PCA.sav"))
        #transform vektor fitur testing menggunakan classifier pca
        pca_matrix = pca.transform(hog_matrix)

        X_test_HA = hog_matrix
        X_test_HPA = pca_matrix

        print("HOG data test shape : ", X_test_HA.shape)
        print("HOG-PCA data test shape : ", X_test_HPA.shape)

        hog_adaboost_model_list = numpy.load(os.path.join(model_path, 'HOG_AdaBoost_Model.npy'))
        tic = timeit.default_timer()
        f1_ha = adaboost_training_testing.scoreAdaBoost(X_test_HA, label, hog_adaboost_model_list)
        toc = timeit.default_timer()
        _ha_test_time_ = toc - tic
        result[0] = f1_ha
        result[2] = _ha_test_time_

        hog_pca_adaboost_model_list = numpy.load(os.path.join(model_path, 'HOG_PCA_AdaBoost_Model.npy'))
        tic = timeit.default_timer()
        f1_hpa = adaboost_training_testing.scoreAdaBoost(X_test_HPA, label, hog_pca_adaboost_model_list)
        toc = timeit.default_timer()
        _hpa_test_time_ = toc - tic
        result[3] = f1_hpa
        result[5] = _hpa_test_time_

        numpy.savetxt(os.path.join(model_path, 'Result_Model.txt'), result, fmt='%0.3f')

    except Exception:
        print(traceback.format_exc())
        pass

def mainProcess(img_data) :
    hog_matrix = None
    dataset = numpy.loadtxt(os.path.join(test_path, 'dataset.txt'), dtype=numpy.object, delimiter="%s")
    for img in img_data:
        if (dataset == "Caltech"):
            greyscale_image = preprocessing.greyScale(img)
        else:
            greyscale_image = img
        haarFeature_images = preprocessing.haarFeatureClassifier(greyscale_image)
        resize_images = preprocessing.resizeImage(haarFeature_images)
        hog_feature, hog_image = hog_training_testing.hogImage(resize_images)
        hog_matrix = pca_training_testing.matrixCombine(hog_feature, hog_matrix)
    return hog_matrix

def main() :

    app = QApplication(sys.argv)
    widget = GUI()
    widget.show()
    sys.exit(app.exec())

if __name__ == '__main__':
   main()
