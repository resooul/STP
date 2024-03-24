def preprocess(shipmentStatusDate, row, datasetRow, index):

    predictionRow = copy.deepcopy(row)
    
    shipmentStatusDate = dt.datetime.strptime(str(shipmentStatusDate), "%Y-%m-%d %H:%M:%S")
    shipmentCreatedDate = dt.datetime.strptime(datasetRow.at[index, 'shipmentCreatedDate'], "%Y-%m-%d %H:%M:%S")
 
    timeDifference = (shipmentStatusDate - shipmentCreatedDate).total_seconds()
    predictionRow['timeDifference'] = (timeDifference / 60) #in minutes
    
    # Create time series features based on time series index.
    predictionRow.at[index, 'hour_S'] = shipmentStatusDate.hour
    predictionRow.at[index, 'dayofweek_S'] = shipmentStatusDate.weekday()
    
    predictionRow.at[index, 'month_S'] = shipmentStatusDate.month
    predictionRow.at[index, 'year_S'] = shipmentStatusDate.year

    predictionRow.at[index, 'dayofmonth_S'] = shipmentStatusDate.day
    predictionRow.at[index, 'weekofyear_S'] = shipmentStatusDate.isocalendar().week
    
    predictionRow.at[index, 'dayofyear_S'] = shipmentStatusDate.timetuple().tm_yday
    
    predictionRow.at[index, 'quarter_S'] = ((shipmentStatusDate.month - 1) / 3 + 1)
    
    # Weekend updates     
    senderCountry = row.at[index, 'senderCountry']
    
    dayofweek_S = str(row.at[index, 'dayofweek_S'])
    hour_S = str(row['hour_S'])

    if (senderCountry != 2):
        if (dayofweek_S == 4 or dayofweek_S == 5):
            predictionRow.at[index, 'isWeekend_S'] = 1
        else:
            predictionRow.at[index, 'isWeekend_S'] = 0
    elif (senderCountry == 2):
        if (dayofweek_S == 5 or dayofweek_S == 6 or (dayofweek_S == 4 and hour_S > 12)):
            predictionRow.at[index, 'isWeekend_S'] = 1
        else:
            predictionRow.at[index, 'isWeekend_S'] = 0

            
    # Part of day updates    
    hour_S = row.at[index, 'hour_S']
    
    predictionRow.at[index, 'partOfDay_S_morning'] = 0
    predictionRow.at[index, 'partOfDay_S_afternoon'] = 0
    predictionRow.at[index, 'partOfDay_S_evening'] = 0
    predictionRow.at[index, 'partOfDay_S_night'] = 0

    if (hour_S >= 6 and hour_S < 12):
        predictionRow.at[index, 'partOfDay_S_morning'] = 1
    elif (hour_S >= 12 and hour_S < 18):
        predictionRow.at[index, 'partOfDay_S_afternoon'] = 1
    elif (hour_S >= 18 and hour_S < 23):
        predictionRow.at[index, 'partOfDay_S_evening'] = 1
    elif (hour_S >= 23 or hour_S < 6):
        predictionRow.at[index, 'partOfDay_S_night'] = 1
    
    return predictionRow