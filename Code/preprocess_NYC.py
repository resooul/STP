import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime
import calendar
from math import sin, cos, sqrt, atan2, radians,asin
import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap
from folium.plugins import TimestampedGeoJson
from folium.plugins import MarkerCluster
from geopy.distance import great_circle
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import timedelta
import datetime as dt
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
import folium
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from geopy.distance import geodesic
np.random.seed = 42

df=pd.read_csv("Data/yellow_tripdata_2022-01.csv")
df.head()

df = df.rename(columns={"tpep_pickup_datetime": "pickup_datetime", "tpep_dropoff_datetime": "dropoff_datetime"})

df['store_and_fwd_flag'] = df['store_and_fwd_flag'].str.replace('N','0')
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].str.replace('Y','1')
df['store_and_fwd_flag'] = pd.to_numeric(df['store_and_fwd_flag'])

df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')
df['dropoff_datetime']=pd.to_datetime(df['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S')

df.isna().sum().sort_values(ascending = False)
df = df.dropna()

def generate_random_datetime_before(pickup_datetime, minutes_range=5):
    
    # Calculate the range in seconds
    seconds_range = minutes_range * 60
    
    # Convert pickup_datetime to timestamp
    pickup_timestamp = pickup_datetime.timestamp() - (3 * 60 * 60)
    
    # Generate a random offset within the range
    random_offset = np.random.randint(0, seconds_range)
    
    # Calculate the random datetime
    random_datetime = datetime.fromtimestamp(pickup_timestamp - random_offset)
    
    return random_datetime

start_datetime = df['pickup_datetime'].apply(lambda x: generate_random_datetime_before(x))
df['start_datetime'] = start_datetime

import copy
dfPickup = copy.deepcopy(df)
dfDropoff = copy.deepcopy(df)

dfPickup["statusDate"] = copy.deepcopy(dfPickup["pickup_datetime"])
dfPickup["status"] = 0

start_array = df['start_datetime'].values
pickup_array = df['pickup_datetime'].values
trip_duration = np.subtract(pickup_array, start_array)
dfPickup['statusDuration'] = pd.Series(trip_duration)
dfPickup['statusDuration'] = dfPickup['statusDuration'].dt.total_seconds()



dfDropoff["statusDate"] = copy.deepcopy(dfDropoff["dropoff_datetime"])
dfDropoff["status"] = 1

start_array = df['start_datetime'].values
dropoff_array = df['dropoff_datetime'].values
trip_duration = np.subtract(dropoff_array, start_array)
dfDropoff['statusDuration'] = pd.Series(trip_duration)
dfDropoff['statusDuration'] = dfDropoff['statusDuration'].dt.total_seconds()

start_datetime = dfPickup['pickup_datetime'].apply(lambda x: generate_random_datetime_before(x))

frames = [dfPickup, dfDropoff]
dfResult = pd.concat(frames)
dfResult['index'] = dfResult.index
dfResult.set_index('index', inplace = True)

def create_features(df, suffix):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour' + suffix] = df.index.hour
    df['minute' + suffix] = df.index.minute
    df['dayofweek' + suffix] = df.index.dayofweek
    df['quarter' + suffix] = df.index.quarter
    df['month' + suffix] = df.index.month
    df['year' + suffix] = df.index.year
    df['dayofyear' + suffix] = df.index.dayofyear
    df['dayofmonth' + suffix] = df.index.day
    df['weekofyear' + suffix] = df.index.isocalendar().week
    return df

dfResult = dfResult.set_index('statusDate')
dfResult.index = pd.to_datetime(dfResult.index)
dfResult = create_features(dfResult, '')

dfResult = dfResult.reset_index()

dfResult['index'] = dfResult.index
dfResult.set_index('index', inplace = True)

dfResult.head()

dfResult.isna().sum().sort_values(ascending = False)

X = dfResult.drop(columns=["status"])
y = dfResult["status"]

# Split data as train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify=y, shuffle=True)

X_train.to_csv('Data/X_Train.csv')
y_train.to_csv('Data/y_Train.csv')
X_test.to_csv('Data/X_Test.csv')
y_test.to_csv('Data/y_Test.csv')

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))