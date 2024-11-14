import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = 'data.csv'

df = pd.read_csv(data, header=None, sep=',')


class GNaiveBayesClassifier:
    def priorProb(self, X, y):

        self.prior = (X.groupby(y).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior

    def statParamteres(self, X, y):

        self.mean = X.groupby(y).apply(np.mean).to_numpy()
        self.var = X.groupby(y).apply(np.var).to_numpy()
        return self.mean, self.var

    def densGauss(self, class_i, x):

        mean = self.mean[class_i]
        var = self.var[class_i]
        pmf = np.exp((-1/2)*((x-mean)**2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        return pmf

    def postProb(self, x):

        posteriors = []

        for i in range(self.count):

            prior = np.log(self.prior[i])
            conditional = np.sum(np.log(self.densGauss(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def train(self, X, y):
        self.classes = np.unique(y)
        self.count = len(self.classes)
        self.num_feature = X.shape[1]
        self.rows = X.shape[0]

        self.statParamteres(X, y)
        self.priorProb(X, y)

    def predict(self, X):
        preds = [self.postProb(f) for f in X.to_numpy()]
        return preds

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

    def confusionMatrix(self, predicted, classes): #confusion matrix
        mat = np.zeros((2, 2), dtype=np.int32)

        predictedNP = np.array(predicted)
        classesNP = np.array(classes)

        for i in range(len(predictedNP)):
            if predictedNP[i] == 1:
                if classesNP[i] == 1:#TP
                    mat[0][0] += 1
                elif classesNP[i] == 0:#FP
                    mat[0][1] += 1
            elif predictedNP[i] == 0:
                if classesNP[i] == 1:#TN
                    mat[1][0] += 1
                elif classesNP[i] == 0:#FN
                    mat[1][1] += 1

        print(mat)
