import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import datetime as dt
import math

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from collections import Counter
from IPython.core.display import display, HTML
sns.set_style('darkgrid')
pd.set_option('display.max_columns', 50)

# Set the random number generator to a fixed sequence.
np.random.seed(42)

# Load Data Set
X_Train = pd.read_csv('../Data/data_X_Train.csv')
X_Train.set_index('index', inplace = True)

y_Train = pd.read_csv('../Data/data_y_Train.csv')
y_Train.set_index('index', inplace = True)

X_Test = pd.read_csv('../Data/data_X_Test.csv')
X_Test.set_index('index', inplace = True)

y_Test = pd.read_csv('../Data/data_y_Test.csv')
y_Test.set_index('index', inplace = True)


# Load the XGB prediction model from disk
import pickle
filename = 'modelXGB.sav'
predictor = pickle.load(open(filename, 'rb'))


# Optimization Process on Train Set
import STPMPA

a = 0;

target = -1
originalRow = pd.DataFrame({'A' : []})
datasetRow = pd.DataFrame({'A' : []})
index = 0

totalTD = 0
totalPT = 0
totalOP = 0

result_Train = copy.deepcopy(y_Train)

for i, rowX in X_Train.iterrows():
        
    # get the target
    target = y_Train.at[i, 'TARGET']
    
    if (a == 25000):
        break
    
    originalRow = rowX.to_frame().T
    datasetRow = dataset.iloc[i].to_frame().T
    index = i
    
    shipmentCreatedDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentCreatedDate'], "%Y-%m-%d %H:%M:%S")
    shipmentStatusDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentStatusDate'], "%Y-%m-%d %H:%M:%S")
        
    year = ((int(shipmentCreatedDate.year)) - 2020) / 3
    month = (int(shipmentCreatedDate.month))
    day = (int(shipmentCreatedDate.day))
    
    month_decoded = (month /11)
    day_decoded = (day / 30)
    
    month_lb = month_decoded
    day_lb = day_decoded
    month_ub = month_lb
    day_ub = day_lb + (5/30)
    
    if (month == 2):
        if ((day + 5) > 28):
            month_ub = month_lb + (1/11)
     
    elif (month == 4 or month == 6 or month == 9 or month == 11):
        if ((day + 5) > 30):
            month_ub = month_lb + (1/11)
    else:
        if ((day + 5) > 31):
            month_ub = month_lb + (1/11)

    problem = {
        "lb": [year + 0.00001,month_lb,day_lb,0.00001,0.00001],
        "ub": [1,month_ub,day_ub,1,1],
        "minmax": "min",
        "fit_func": objective_function,
        "name": "Shipment Status Time Prediction Problem",
        "log_to": None,
    }
    
    np.random.seed(42)

    optimizer = STPMPA.STPMPA(epoch=20, pop_size=5)
            
    optimizationStart = time.time()

    # main function
    optimizer.solve(problem)
    # main function

    optimizationEnd = time.time()
    optimizationTime = optimizationEnd - optimizationStart
    
    bestPrediction = optimizer.solution[0]
    
    year_encoded = 2019 + math.ceil(bestPrediction[0] * 3)
    month_encoded = math.ceil(bestPrediction[1] * 11) % 12
    day_encoded = math.ceil(bestPrediction[2] * 30) % 31
    hour_encoded = math.ceil(bestPrediction[3] * 23)
    minute_encoded = math.ceil(bestPrediction[4]* 59)

    if (month_encoded == 0):
        month_encoded = 12
    if (day_encoded == 0):
        day_encoded = 31     
        
    if (month_encoded == 2 and (day_encoded > 28)):
        day_encoded = 28
    if ((month_encoded == 4 or month_encoded == 6 or month_encoded == 9 or month_encoded == 11) and (day_encoded > 30)):
        day_encoded = 30
    
    predictionDate = dt.datetime(year_encoded, month_encoded, day_encoded, hour_encoded, minute_encoded, 0)
    timeDifference = int((abs(predictionDate - shipmentStatusDate).total_seconds()) / 60)
    
    bestSolutionSet = preprocess(predictionDate, originalRow, datasetRow, index)
    
    predictionStart = time.time()

    # main function
    prediction = predictor.predict(bestSolutionSet)
    # main function

    predictionEnd = time.time()
    predictionTime = predictionEnd - predictionStart
        
    
    result_Train.at[i, "prediction"] = prediction
    result_Train.at[i, "actualDate"] = shipmentStatusDate
    result_Train.at[i, "predictionDate"] = predictionDate
    result_Train.at[i, "timeDifference"] = timeDifference
    result_Train.at[i, "predictionTime"] = predictionTime
    result_Train.at[i, "optimizationTime"] = optimizationTime
    result_Train.at[i, "status"] = target
    

    print(f"{a}, TD: {timeDifference}, Pre.T: {predictionTime}, Opt.T: {optimizationTime}")
    
    totalTD += timeDifference
    totalPT += predictionTime
    totalOP += optimizationTime
    
    a += 1

# Optimization Process on Test Set
a = 0;

#global parameters for objective function
target = -1
originalRow = pd.DataFrame({'A' : []})
datasetRow = pd.DataFrame({'A' : []})
index = 0

totalTD = 0
totalPT = 0
totalOP = 0

result_Test = copy.deepcopy(y_Test)

for i, rowX in X_Test.iterrows():
        
    # get the target
    target = y_Test.at[i, 'TARGET']
    
    if (a == 25000):
        break
    
    originalRow = rowX.to_frame().T
    datasetRow = dataset.iloc[i].to_frame().T
    index = i
    
    shipmentCreatedDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentCreatedDate'], "%Y-%m-%d %H:%M:%S")
    shipmentStatusDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentStatusDate'], "%Y-%m-%d %H:%M:%S")
        
    year = ((int(shipmentCreatedDate.year)) - 2020) / 3
    month = (int(shipmentCreatedDate.month))
    day = (int(shipmentCreatedDate.day))
    
    month_decoded = (month /11)
    day_decoded = (day / 30)
    
    month_lb = month_decoded
    day_lb = day_decoded
    month_ub = month_lb
    day_ub = day_lb + (5/30)
    
    if (month == 2):
        if ((day + 5) > 28):
            month_ub = month_lb + (1/11)
     
    elif (month == 4 or month == 6 or month == 9 or month == 11):
        if ((day + 5) > 30):
            month_ub = month_lb + (1/11)
    else:
        if ((day + 5) > 31):
            month_ub = month_lb + (1/11)

    problem = {
        "lb": [year + 0.00001,month_lb,day_lb,0.00001,0.00001],
        "ub": [1,month_ub,day_ub,1,1],
        "minmax": "min",
        "fit_func": objective_function,
        "name": "Shipment Status Time Prediction Problem",
        "log_to": None,
    }
    
    np.random.seed(42)

    optimizer = MPA.STPMPA(epoch=20, pop_size=5)
            
    optimizationStart = time.time()

    # main function
    optimizer.solve(problem)
    # main function

    optimizationEnd = time.time()
    optimizationTime = optimizationEnd - optimizationStart
    
    bestPrediction = optimizer.solution[0]
    
    year_encoded = 2019 + math.ceil(bestPrediction[0] * 3)
    month_encoded = math.ceil(bestPrediction[1] * 11) % 12
    day_encoded = math.ceil(bestPrediction[2] * 30) % 31
    hour_encoded = math.ceil(bestPrediction[3] * 23)
    minute_encoded = math.ceil(bestPrediction[4]* 59)

    if (month_encoded == 0):
        month_encoded = 12
    if (day_encoded == 0):
        day_encoded = 31     
        
    if (month_encoded == 2 and (day_encoded > 28)):
        day_encoded = 28
    if ((month_encoded == 4 or month_encoded == 6 or month_encoded == 9 or month_encoded == 11) and (day_encoded > 30)):
        day_encoded = 30
    
    
    predictionDate = dt.datetime(year_encoded, month_encoded, day_encoded, hour_encoded, minute_encoded, 0)
    timeDifference = int((abs(predictionDate - shipmentStatusDate).total_seconds()) / 60)
    
    bestSolutionSet = preprocess(predictionDate, originalRow, datasetRow, index)
    
    
    predictionStart = time.time()

    # main function
    prediction = predictor.predict(bestSolutionSet)
    # main function

    predictionEnd = time.time()
    predictionTime = predictionEnd - predictionStart
        
    
    result_Test.at[i, "prediction"] = prediction
    result_Test.at[i, "actualDate"] = shipmentStatusDate
    result_Test.at[i, "predictionDate"] = predictionDate
    result_Test.at[i, "timeDifference"] = timeDifference
    result_Test.at[i, "predictionTime"] = predictionTime
    result_Test.at[i, "optimizationTime"] = optimizationTime
    result_Test.at[i, "status"] = target
    

    print(f"{a}, TD: {timeDifference}, Pre.T: {predictionTime}, Opt.T: {optimizationTime}")
    
    totalTD += timeDifference
    totalPT += predictionTime
    totalOP += optimizationTime
    
    a += 1