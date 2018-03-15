import json
import numpy as np
import os
from math import ceil


class DataLoader:

    def __init__(self, path, numClasses, batchSize, train_amount, val_amount):
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.allData = [[] for i in range(numClasses)]
        self.trainData = [[] for i in range(numClasses)]
        self.valData = [[] for i in range(numClasses)]
        self.testData = [[] for i in range(numClasses)]
        self.trainIterator = []
        self.valIterator = []
        self.testIterator = []
        for i in range(numClasses):
            self.trainIterator.append(0)
            self.valIterator.append(0)
            self.testIterator.append(0)

        print("Number of classes: " + str(self.allData.__len__()))
        files = os.listdir(path)
        for name in files:
            file = open(path + name, "r")
            if type(self.allData[int(name[:1])]) is list:
                self.allData[int(name[:1])] = self.allData[int(name[:1])] + json.loads(file.readline())
            else:
                self.allData[int(name[:1])] = json.loads(file.readline())
        print("Class 0: " + str(self.allData[0].__len__()))
        print("Class 1: " + str(self.allData[1].__len__()))
        print("Class 2: " + str(self.allData[2].__len__()))
        print("Class 3: " + str(self.allData[3].__len__()))

        self.size = 0
        for i in range(self.numClasses):
            self.size += len(self.allData[i])

        for i in range(numClasses):
            train_amount_abs = ceil(train_amount * self.allData[i].__len__())
            self.trainData[i] = self.allData[i][0:train_amount_abs]
            val_amount_abs = ceil(val_amount * self.allData[i].__len__())
            self.valData[i] = self.allData[i][train_amount_abs+1:train_amount_abs+val_amount_abs+1]
            self.testData[i] = self.allData[i][train_amount_abs+val_amount_abs+2:]

        self.trainSize = 0
        self.valSize = 0
        self.testSize = 0
        for i in range(numClasses):
            self.trainSize += self.trainData[i].__len__()
            self.valSize += self.valData[i].__len__()
            self.testSize += self.testData[i].__len__()

        print("Train data: "+str(self.trainSize))
        print("Val data: "+str(self.valSize))
        print("Test data: "+str(self.testSize))

        #print(self.trainData[2][1])
        #print(self.valData[2][1])
        #print(self.testData[2][1])


    def nextTrain_batch(self):
        batch_x = []
        batch_y = []
        for i in range(self.batchSize):
            index = i % self.numClasses
            if self.trainData[index].__len__() > 0:
                batch_x.append(self.trainData[index][self.trainIterator[index]])
                batch_y.append(index)
                if self.trainIterator[index] < self.trainData[index].__len__() - 1:
                    self.trainIterator[index] += 1
                else:
                    self.trainIterator[index] = 0
        return np.asarray(batch_x, np.float32), np.asarray(batch_y, np.int)

    def trainLength(self):
        return self.trainSize

    def nextVal_batch(self):
        batch_x = []
        batch_y = []
        for i in range(self.batchSize):
            index = i % self.numClasses
            if self.valData[index].__len__() > 0:
                batch_x.append(self.valData[index][self.valIterator[index]])
                batch_y.append(index)
                if self.valIterator[index] < self.valData[index].__len__() - 1:
                    self.valIterator[index] += 1
                else:
                    self.valIterator[index] = 0
        return np.asarray(batch_x, np.float32), np.asarray(batch_y, np.int)

    def valLength(self):
        return self.valSize

    def nextTest_batch(self):
        batch_x = []
        batch_y = []
        for i in range(self.batchSize):
            index = i % self.numClasses
            if self.testData[index].__len__() > 0:
                batch_x.append(self.testData[index][self.testIterator[index]])
                batch_y.append(index)
                if self.testIterator[index] < self.testData[index].__len__() - 1:
                    self.testIterator[index] += 1
                else:
                    self.testIterator[index] = 0
        return np.asarray(batch_x, np.float32), np.asarray(batch_y, np.int)

    def testLength(self):
        return self.testSize
