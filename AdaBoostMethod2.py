
import math
import numpy
from HumanFace import HumanFace
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class AdaBoostMethod2 :

    def giniIndex(self,data,label,threshold):
        depth = 1
        m,n = data.shape

        gini = numpy.empty(shape=(n, 1))
        conditions = numpy.empty(shape=(n, 1),dtype=object)
        gini_index = numpy.empty(shape=(depth, 1))

        gini_lt_pos = 0
        gini_lt_neg = 0
        gini_gt_pos = 0
        gini_gt_neg = 0
        for i in range(n) :
            for j in range(m) :
                if data[j,i] < threshold[i] and label[j] == 1 :
                    gini_lt_pos += 1
                elif data[j,i] < threshold[i] and label[j] == -1 :
                    gini_lt_neg += 1
                elif data[j,i] >= threshold[i] and label[j] == 1:
                    gini_gt_pos += 1
                elif data[j,i] >= threshold[i] and label[j] == -1:
                    gini_gt_neg += 1
            gini_lt = 1 - (math.pow((gini_lt_pos/(gini_lt_neg+gini_lt_pos)),2)+math.pow((gini_lt_neg/(gini_lt_neg+gini_lt_pos)),2))
            gini_gt = 1 - (math.pow((gini_gt_pos/(gini_gt_neg+gini_gt_pos)),2)+math.pow((gini_gt_neg/(gini_gt_neg+gini_gt_pos)),2))
            gini[i] = ((gini_lt_pos+gini_lt_neg)/m)* gini_lt + ((gini_gt_pos+gini_gt_neg)/m)* gini_gt
            if (gini_gt) >= (gini_lt) :
                conditions[i] = "gt"
            else :
                conditions[i] = "lt"


        index = numpy.argsort(gini, axis=0)
        for x in range(depth) :
            gini_index[x] = index[x]

        return gini_index,conditions

    def treeClassifier(self,data,threshold,conditions,gini_index,i,j):

        if(conditions[int(gini_index[i])] == "gt") :
             if (data[j,int(gini_index[i])] >= (threshold[int(gini_index[i])])) :
                 if (i+1 <len(gini_index)) :
                     value = AdaBoostMethod2.treeClassifier(self, data, threshold,conditions, gini_index,i + 1,j)
                 else :
                     value = 1
             else :
                value = -1

        elif (conditions[int(gini_index[i])]  == "lt"):
            if (data[j,int(gini_index[i])] < (threshold[int(gini_index[i])])):
                if (i+1 < len(gini_index)):
                    value = AdaBoostMethod2.treeClassifier(self, data, threshold, conditions, gini_index,i + 1,j)
                else:
                    value = 1
            else:
                value = -1

        return value


    def trainAdaBoost(self,data,label,param) :
        humanFaceList = []
        m, n = data.shape
        label = numpy.mat(label).T
        #print(label)
        classLabel = numpy.unique(label, axis=0)
        # print (classLabel)
        for cL in range(classLabel.shape[0]):  # jenis kelas muka (0-10)

            yLabel = numpy.ones(shape=(m, 1), dtype=numpy.float32)
            for x in range(label.shape[0]):  # jumlah data training, inisialisasi +1 / -1
                if (label[x] != classLabel[cL]):
                    yLabel[x] = -1

            threshold = numpy.sum(data, axis=0)
            threshold = numpy.divide(threshold,m)
            gini_index,conditions = AdaBoostMethod2.giniIndex(self,data,yLabel,threshold)
            # print(m ,' + ', n)
            # print("classifier ke-", cL)
            # print(yLabel)
            D = numpy.empty(shape=(m, 1))
            D.fill(1 / m)
            T = param.T
            classArr = numpy.empty(shape =(T,1))
            aggClassEst = numpy.empty(shape=(m,1))
            for t in range(T):
                weakArray = numpy.zeros(shape=(m, 1), dtype=numpy.float32)
                minError = math.inf
                # print("Threshold ke - ", i, " = ", threshold[i])
                errorArray = numpy.ones(shape=(m, 1), dtype=numpy.float32)
                weightedError = numpy.zeros(shape=(m, 1), dtype=numpy.float32)
                for j in range(m):  # untuk setiap jumlah datanya
                    weakArray[j] = AdaBoostMethod2.treeClassifier(self,data,threshold,conditions,gini_index,0,j)
                    #print(j, " ", weakArray[j])
                    if weakArray[j] == yLabel[j]:
                        errorArray[j] = 0
                    weightedError[j] = errorArray[j] * D[j]
                    #print(weightedError[j])
                weightedError = numpy.sum(weightedError)
                #print("Weighted Error = ", weightedError)
                alpha = float(0.5 * math.log((1 - weightedError) / max(weightedError, 10 ** -16)))
                #print("alpha ke-", cL)
                #print(alpha)
                #print(weakArray)
                for j in range(m):  # update weight baru
                    D[j] = D[j] * math.exp(-1 * alpha * weakArray[j] * yLabel[j])
                D = numpy.divide(D, D.sum())
                #for j in range(m) :
                #    print ("Iterasi ke - ",t," Weight baru ke - ",j , " = ", D[j])
                classArr[t] = alpha

            humanFace = HumanFace(classLabel[cL], classArr, gini_index,threshold,conditions)
            humanFaceList.append(humanFace)

        return humanFaceList

    def scoreAdaBoost(self,data,label,faceClassifier,param) :
        m, n = data.shape
        y_label = numpy.mat(label).T
        # print (m , " - ", n)
        T = param.T

        y_pred = numpy.empty(shape=(m,1),dtype= numpy.object)
        y_pred.fill("Empty")
        for i in range (m) :
            #thresh = 0
            for fC in range(len(faceClassifier)):
                total = 0

                classifier = faceClassifier[fC].classifier
                for t in range(T):

                    weakValue = AdaBoostMethod2.treeClassifier(self, data, faceClassifier[fC].threshold,
                                                               faceClassifier[fC].conditions, faceClassifier[fC].gini_index,
                                                               0, i)
                    total += classifier[t] * weakValue

                print(i, "-", faceClassifier[fC].name, " = ", total)
                if (total > 0):
                    # thresh = total
                    if faceClassifier[fC].name == y_label[i]:
                        y_pred[i] = y_label[i]

            print(y_pred[i], " - ", y_label[i])
        f1score = f1_score(y_label, y_pred, average='micro')
        accscore = accuracy_score(y_label, y_pred)
        return accscore, f1score

    def testAdaBoost(self,data,faceClassifier,param) :
        m,n = data.shape
        #print (m , " - ", n)
        T = param.T
        value = numpy.zeros(shape=(len(faceClassifier),1))
        name = "null"
        for fC in range (len(faceClassifier)) :
            print (fC," - ",faceClassifier[fC].name)
            classifier = faceClassifier[fC].classifier
            total = numpy.zeros(shape=(T,1))
            for t in range(T) :
                weakValue = AdaBoostMethod2.weakClassify(self,classifier[t,1], data[0,classifier[t,0]])
                total[t] = classifier[t,2] * weakValue
            value[fC] = numpy.sign(numpy.sum(total))
            if value[fC] == 1 :
                name = faceClassifier[fC].name
        print(value)
        return name



