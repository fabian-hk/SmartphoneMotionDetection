import json
import numpy as np
import os
from math import ceil
import random

# Note:
# Problem files:
#       1_class_time_gravity_000005.txt
#       2_class_time_gravity_000002.txt

class DataLoader:
    # constants
    TRAIN = 0
    VAL = 1
    TEST = 2

    def __init__(self, path, num_classes, batch_size, amount=[]):
        # initialize variables
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.allData = [[] for i in range(num_classes)]
        self.batchData = [[[] for i in range(num_classes)] for i in range(3)]
        self.iterator = [[0 for j in range(num_classes)] for i in range(3)]
        self.dataSize = []

        print("Number of classes: " + str(self.allData.__len__()))
        files = os.listdir(path)
        # load all trainings data from disk
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

        # split data in train, val and test data
        amount.append(1.0 - amount[1] - amount[0])
        for i in range(self.num_classes):
            start = 0
            for j in range(2):
                amount_abs = ceil(amount[j] * self.allData[i].__len__())
                end = start + amount_abs
                self.batchData[j][i] = self.allData[i][start:end]
                start = end + 1
            self.batchData[2][i] = self.allData[i][start:]

        # calculate the size of each data set
        for j in range(3):
            size = 0
            for i in range(self.num_classes):
                size += self.batchData[j][i].__len__()
            self.dataSize.append(size)

        print("Train data: " + str(self.dataSize[0]))
        print("Val data: " + str(self.dataSize[1]))
        print("Test data: " + str(self.dataSize[2]))

        # print(self.batchData[0][0][1])
        # print(self.batchData[1][0][1])
        # print(self.batchData[2][0][1])

    def next_batch(self, j):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            index = random.randint(0, self.num_classes - 1)
            if self.batchData[j][index].__len__() > 0:
                batch_x.append(self.batchData[j][index][self.iterator[j][index]])
                batch_y.append(index)
                if self.iterator[j][index] < self.batchData[j][index].__len__() - 1:
                    self.iterator[j][index] += 1
                else:
                    self.iterator[j][index] = 0
        return np.asarray(batch_x, np.float32), np.asarray(batch_y, np.int)

    def length(self, i):
        return self.dataSize[i]
