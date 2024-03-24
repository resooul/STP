def objective_function(solution):
    global target, originalRow, datasetRow, index
    
    value = 10000000
    referenceDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentCreatedDate'], "%Y-%m-%d %H:%M:%S")

    year_encoded = 2019 + math.ceil(solution[0] * 3)
    month_encoded = math.ceil(solution[1] * 11) % 12
    day_encoded = math.ceil(solution[2] * 30) % 31
    hour_encoded = math.ceil(solution[3] * 23)
    minute_encoded = math.ceil(solution[4]* 59)
    
    if (month_encoded == 0):
        month_encoded = 12
    if (day_encoded == 0):
        day_encoded = 31
    
    if (month_encoded == 2 and (day_encoded > 28)):
        day_encoded = 28
    if ((month_encoded == 4 or month_encoded == 6 or month_encoded == 9 or month_encoded == 11) and (day_encoded > 30)):
        day_encoded = 30
    
    # calculate prediction date
    try:
        predictionDate = dt.datetime(year_encoded, month_encoded, day_encoded, hour_encoded, minute_encoded, 0)
    except:
        print("Invalid date in objective function: ", year_encoded, month_encoded, day_encoded, hour_encoded, minute_encoded, "-")
        return value
    
    # prepare the prediction row
    predictionRow = preprocess(predictionDate, originalRow, datasetRow, index)
    
    # predict
    prediction = predictor.predict(predictionRow)
    
    value = int((abs(predictionDate - referenceDate).total_seconds()) / 60) + penalty_function(target,prediction)
  
    return value