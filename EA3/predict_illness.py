import random
import pandas as pd
import time
from decision_tree import trainTestSplit, buildDecisionTree, decisionTreePredictions, calculateAccuracy

df = pd.read_csv('data.csv')
print("Decision Tree - Illness Dataset")
df['Blood_Pressure'] = df['Blood_Pressure'].map({'Low': -1, 'Normal': 0, 'High': 1})
df['Symptoms'] = df['Symptoms'].map({'Yes': 1, 'No': 0})
df['Family_History'] = df['Family_History'].map({'Yes': 1, 'No': 0})
df['Test_Levels'] = df['Test_Levels'].map({'Low': -1, 'Normal': 0, 'High': 1})
dataFrameTrain, dataFrameTest = trainTestSplit(df, testSize=0.25)

i = 1
accuracyTrain = 0
while accuracyTrain < 100:
    startTime = time.time()
    decisionTree = buildDecisionTree(dataFrameTrain, maxDepth=i)
    buildingTime = time.time() - startTime
    decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
    accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
    accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("maxDepth = {}: ".format(i), end="")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end="")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end="")
    print("buildTime = {0:.2f}s".format(buildingTime), end="\n")
    i += 1


def printDecisionTree(decisionTree, indent="  ", prefix=""):
    if not isinstance(decisionTree, dict):
        print(f"{prefix}Predict: {decisionTree}")
        return

    question = list(decisionTree.keys())[0]
    print(f"{prefix}Question: {question}")

    if not decisionTree[question]:
        return

    print(f"{prefix}{indent}If yes:")
    printDecisionTree(decisionTree[question][0], indent, prefix + indent)

    print(f"{prefix}{indent}If no:")
    printDecisionTree(decisionTree[question][1], indent, prefix + indent)


tree = buildDecisionTree(dataFrameTrain, maxDepth=i)
printDecisionTree(tree)

dataFrameGiven = pd.DataFrame([[25, 1, 1, 0, 1], [48, -1, 1, 0, 0], [28, 0, 0, 0, 0]], columns=['Age', 'Blood_Pressure', 'Symptoms', 'Family_History', 'Test_Levels'])

prediction = decisionTreePredictions(dataFrameGiven, tree)
print(f'\n{prediction}')
