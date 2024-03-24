# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from collections import Counter
from IPython.core.display import display, HTML
sns.set_style('darkgrid')
pd.set_option('display.max_columns', 50)

# Read Data

dataPath = "../../Data/NYC"

X_Train = pd.read_csv(f'{dataPath}/X_Train.csv')
X_Train.set_index('index', inplace = True)
y_Train = pd.read_csv(f'{dataPath}/y_Train.csv')
y_Train.set_index('index', inplace = True)

X_Test = pd.read_csv(f'{dataPath}/X_Test.csv')
X_Test.set_index('index', inplace = True)
y_Test = pd.read_csv(f'{dataPath}/y_Test.csv')
y_Test.set_index('index', inplace = True)

X_Train = X_Train.drop(columns=["start_datetime", "pickup_datetime", "dropoff_datetime", "statusDate"])
X_Test = X_Test.drop(columns=["start_datetime", "pickup_datetime", "dropoff_datetime", "statusDate"])

X = pd.concat([X_Train, X_Test], axis=0)
y = pd.concat([y_Train, y_Test], axis=0)


seedValue = 42
np.random.seed(seedValue)
runs = 30

for i in range(runs):
    print(f"Run: {i} Seed: {seedValue}")
    
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.3, random_state = seedValue)
    
    # Hyperparameter Tuning

    import xgboost as xgb

    # Train Model

    # Get the best parameters from the GridSearch

    model = xgb.XGBClassifier(
                                n_estimators = 10,
                                max_depth = 10,
                                learning_rate = 0.1,
                                tree_method = 'hist'
                            )


    print("Training has started. ", time.strftime("%Y-%m-%d %H:%M:%S"))
    trainStart = time.time()

    # main function
    model = model.fit(X_Train, y_Train)
    # main function

    trainEnd = time.time()
    trainingTime = trainEnd - trainStart
    print("Training has ended. ", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Test Model
    #bestModel = model.best_estimator_
    bestModel = model

    #X_Test_scaled = scaler.fit_transform(X_Test)
    #X_Test_scaled = X_Test

    # Predicting Accuracy the Train set results
    testTrainStart = time.time()
    #y_pred_train = bestModel.predict(X_Train)
    y_pred_train = bestModel.predict(X_Train)
    testTrainEnd = time.time()
    testingTimeTrain = testTrainEnd - testTrainStart

    accuracy_train = accuracy_score(y_Train, y_pred_train)


    # Predicting Accuracy the Test set results
    testTestStart = time.time()
    #y_pred_test = bestModel.predict(X_Test)
    y_pred_test = bestModel.predict(X_Test)
    testTestEnd = time.time()
    testingTimeTest = testTestEnd - testTestStart

    accuracy_test = accuracy_score(y_Test, y_pred_test)

    # Predicting Precision the Test set results
    precision = precision_score(y_Test, y_pred_test, average='weighted')

    # Predicting Recall the Test set results
    recall = recall_score(y_Test, y_pred_test, average='weighted')

    # Predicting F1 Score the Test set results
    f1Score = f1_score(y_Test, y_pred_test, average='weighted')

    print('Accuracy (train): ', accuracy_train)
    print('Accuracy (test): ', accuracy_test)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1Score)
    print('Train Time): ', trainingTime)
    print('Test Time (Train): ', testingTimeTrain)
    print('Test Time: ', testingTimeTest)

    # Save Results
    modelResult = [('XGB', precision, recall, f1Score, accuracy_train, accuracy_test, trainingTime, testingTimeTrain, testingTimeTest),]
    result = pd.DataFrame(data = modelResult, columns=['Model', 'Precision', 'Recall', 'f1 Score', 'Accuracy(Training)', 'Accuracy(Test)', 'Training Time', 'Testing Time (Train)', 'TestingTime'])
    result.to_csv(f'../../Result/NYC/result_XGB_{seedValue}.csv')
    result
    
    seedValue = seedValue + 1