import sys
import os
import timeit
import traceback

import cv2
import numpy

model_path = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\skripsi-face-recognition\\PyCharm\\Model\\Hasil Uji Coba\\Caltech\\Revisi sesudah pra-sidang\\Uji Coba Classifier\\Final\\HOG+PCA+AdaBoost(8,16,16,20,10)"


result = numpy.loadtxt(os.path.join(model_path, 'HOG_PCA_AdaBoost_Model.txt'), dtype=numpy.object, delimiter="%s")
print(result)

#numpy.savetxt(os.path.join(model_path, 'HOG_PCA_AdaBoost_Model2.txt'), result, fmt='%s')
