import numpy
from sklearn.decomposition import PCA

class PCAMethod :

    def __init__(self, k_dimensions):
        self.k_dimensions = k_dimensions


    def matrixCombine(self,imgvector, matrix):
        try:
            matrix = numpy.vstack((matrix, imgvector))
        except:
            matrix = imgvector
        return matrix

    def pcaImage(self,matrix) :
        pca = PCA(n_components=self.k_dimensions).fit(matrix)
        return pca