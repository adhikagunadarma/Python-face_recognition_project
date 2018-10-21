import math
import numpy
from HumanFace import HumanFace
from sklearn.metrics import accuracy_score
import traceback

class AdaBoostMethod :

    def __init__(self, iterations):
        self.iterations = iterations

    def weakClassify(threshold,data_matrix) :
        if (data_matrix > threshold) :
            value = -1
        else :
            value = 1

        return value


    def trainAdaBoost(self,data,label) :
        try :
            m, n = data.shape
            label = numpy.mat(label).T
            # print(label)
            classLabel = numpy.unique(label, axis=0)
            humanFaceModel = numpy.zeros(shape=(classLabel.shape[0], 2), dtype=object)
            humanFaceList = []

            for c in range(classLabel.shape[0]):  # jenis kelas muka (0-18)

                yLabel = numpy.ones(shape=(m, 1), dtype=numpy.int8)
                for x in range(label.shape[0]):  # jumlah data training, inisialisasi +1 / -1
                    if (label[x] != classLabel[c]):
                        yLabel[x] = -1

                threshold = numpy.sum(data, axis=0)
                threshold = numpy.divide(threshold, m)
                # gini_index,conditions = AdaBoostMethod.giniIndex(self,data,yLabel,threshold)
                # print(m ,' + ', n)
                print("classifier ke-", c)

                # for a in range (m) :
                #    print (label[a], "-", yLabel[a] , " - " , classLabel[cL])
                D = numpy.zeros(shape=(m, 1))
                D.fill(1 / m)
                T = self.iterations
                classArr = numpy.zeros(shape=(T, 3))
                for t in range(T):
                    weakArray = numpy.zeros(shape=(m, n), dtype=numpy.int8)
                    minError = math.inf

                    for i in range(n):  # untuk setiap dimensi fitur
                        # print("Threshold ke - ", i, " = ", threshold[i])
                        # for inequal in [0,1] : #0 = lt, 1 = gt
                        errorArray = numpy.ones(shape=(m, 1), dtype=numpy.float32)
                        for j in range(m):  # untuk setiap jumlah datanya

                            weakArray[j, i] = AdaBoostMethod.weakClassify(threshold[i], data[j, i])
                            if weakArray[j, i] == yLabel[j]:
                                errorArray[j] = 0
                            errorArray[j] = errorArray[j] * D[j]
                        totalError = numpy.sum(errorArray)
                        # print(inequal, " Weighted Error ke - ", i, " = ", totalError)
                        if (totalError < minError):
                            minError = totalError
                            dimension = i
                            # sign = inequal
                    # print("dimension ke - " , t , " = " , dimension, "signnya  = ",sign )
                    # print("min error ke-", t , " = " ,minError)
                    if (minError > 0.0):
                        alpha = float(0.5 * math.log((1 - minError) / minError))
                    else:
                        alpha = 0.0
                    # print("alpha ke-", t , " = " , alpha)
                    # print(weakArray)
                    for j in range(m):  # update weight baru
                        if weakArray[j, dimension] == yLabel[j]:
                            D[j] = D[j] * math.exp(alpha * -1)
                        else:
                            D[j] = D[j] * math.exp(alpha)
                    D = numpy.divide(D, D.sum())
                    # for j in range(m) :
                    #    print ("Iterasi ke - ",t+1," Weight baru ke - ",j , " = ", D[j])
                    classArr[t, 0] = dimension
                    classArr[t, 1] = threshold[dimension]
                    # classArr[t,2] = sign
                    classArr[t, 2] = alpha
                humanFaceList.append(HumanFace(classLabel[c], classArr))
                humanFaceModel[c, 0] = classLabel[c]
                humanFaceModel[c, 1] = classArr
            return humanFaceModel, humanFaceList

        except Exception:
            print(traceback.format_exc())

    def scoreAdaBoost(self,data,label, classifier) :
        m, n = data.shape
        y_label = numpy.mat(label).T
        #print (m , " - ", n)
        T = self.iterations
        y_pred = numpy.zeros(shape=(m, 1), dtype=numpy.object)
        y_pred.fill("Empty")
        for i in range (m) :
            thresh = 0
            for c in range(len(classifier)):
                total = 0
                #print(fC, " - ", classifier[c].name)
                class_classifier = classifier[c].classifier

                for t in range(T):
                    weakValue = AdaBoostMethod.weakClassify(float(class_classifier[t, 1]), data[i, int(class_classifier[t, 0])])
                    total += float(class_classifier[t, 2]) * weakValue

                #print(i, "-", classifier[c].name , " = " ,total, "-",y_label[i])
                if (total > thresh):
                    #thresh = total
                    if classifier[c].name == y_label[i]:
                        y_pred[i] = classifier[c].name

            #print(y_pred[i], " - ", y_label[i] ,"-",thresh)
        acc = accuracy_score(y_label ,y_pred)
        return acc

    def testAdaBoost(self,data,classifier) :
        T = self.iterations
        thresh = 0
        names = numpy.zeros(shape=(len(classifier),2),dtype="object")
        index = 0
        for c in range(len(classifier)):
            total = 0

            # print(fC, " - ", classifier[c].name)
            class_classifier = classifier[c].classifier
            for t in range(T):
                weakValue = AdaBoostMethod.weakClassify(float(class_classifier[t, 1]),
                                                            data[int(class_classifier[t, 0])])
                total += float(class_classifier[t, 2]) * weakValue

            # print(i, "-", faceClassifier[fC].name , " = " ,total)
            if (total > thresh):
                #thresh = total
                #name = classifier[c].name
                #names.append(classifier[c].name)
                names[index,0] = classifier[c].name
                names[index,1] = total
                index +=1
        return names



