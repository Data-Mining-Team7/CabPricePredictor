import pandas as pd
import numpy as np
import pytz
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def cab_preprocessor(df,en): 
    #convert from epoch time to EST timezone since data was originally from Boston
    est_time = pd.to_datetime(df['time_stamp'], unit='ms').dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    #parse data into year, month, day, hour, minute columns
    df['year'] = est_time.apply(lambda x: pd.Timestamp(x).year)
    df['month'] = est_time.apply(lambda x: pd.Timestamp(x).month)
    df['day'] = est_time.apply(lambda x: pd.Timestamp(x).day)
    df['hour'] = est_time.apply(lambda x: pd.Timestamp(x).hour)
    df['minute'] = est_time.apply(lambda x: pd.Timestamp(x).minute)
    #parse data into weekend column
    df['weekend'] = est_time.dt.day_name()
    #drop categorical data from the df
    df.drop(['weekend','time_stamp','id'], inplace=True, axis=1)
    #remove rows without entries; now there are 637976 rows from df
    df.dropna(inplace=True)
    #key1 of 'location+month+day+hour' to map to source weather df
    df['key1'] = df['source'].astype(str) + df['month'].astype(str) + df['day'].astype(str) + df['hour'].astype(str)
    #key2 of 'location+month+day+hour' to map to destination weather df
    df['key2'] = df['source'].astype(str) + df['month'].astype(str) + df['day'].astype(str) + df['hour'].astype(str)
    return df

def encoder(df,categorical_columns,en):
    #use one hot encoder to encode source and destination and name of the cab
    en_df = pd.DataFrame(en.fit_transform(df[categorical_columns]).toarray(),columns=[list(en.get_feature_names())])
    #concat the extracted encoded columns to the main dataframe 
    df = pd.concat([df,en_df], axis=1) 
    #drop categorical data from the df
    df.drop(categorical_columns, inplace=True, axis=1)
    return df
    

def weather_preprocessor(df):
    #There are missing values in rain column we will replace them with the avg rain fall
    avg_weather_df = df.groupby('location').mean().reset_index(drop=False)
    for i in df.index:
        if pd.isna(df.rain[i]):
            #get location of the null rainfall row
            location = df.location[i]
            #get avg rain for that location
            avg_rain = avg_weather_df.loc[avg_weather_df['location'] == location]['rain'].values[0]
            #replace null with the avg_rain
            df.rain[i] = avg_rain
    #convert from epoch time to EST timezone since data was originally from Boston
    est_time = pd.to_datetime(df['time_stamp'], unit='s').dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    df['year'] = est_time.apply(lambda x: pd.Timestamp(x).year)
    df['month'] = est_time.apply(lambda x: pd.Timestamp(x).month)
    df['day'] = est_time.apply(lambda x: pd.Timestamp(x).day)
    df['hour'] = est_time.apply(lambda x: pd.Timestamp(x).hour)
    df['minute'] = est_time.apply(lambda x: pd.Timestamp(x).minute)
    #Create df2 for source weather 
    df2 = df.rename(
        columns={
            'temp': 'source_temp',
            'clouds': 'source_clouds',
            'pressure': 'source_pressure',
            'rain': 'source_rain',
            'humidity': 'source_humidity',
            'wind': 'source_wind'
        }
    )
    #create df3 for destionation weather
    df3 = df.rename(
        columns={
            'temp': 'destination_temp',
            'clouds': 'destination_clouds',
            'pressure': 'destination_pressure',
            'rain': 'destination_rain',
            'humidity': 'destination_humidity',
            'wind': 'destination_wind'
        }
    )
    #key1 of 'location+month+day+hour' to map to cab rides df
    df2['key1'] = df['location'].astype(str) + df['month'].astype(str) + df['day'].astype(str) + df['hour'].astype(str)
    #key2 of 'location+month+day+hour' to map to cab rides df
    df3['key2'] = df['location'].astype(str) + df['month'].astype(str) + df['day'].astype(str) + df['hour'].astype(str)
    df2.drop(['time_stamp','year', 'month', 'day', 'hour', 'minute','location'], inplace=True, axis=1)
    df3.drop(['time_stamp','year', 'month', 'day', 'hour', 'minute','location'], inplace=True, axis=1)
    return df2,df3


#initialize the encoder
en = OneHotEncoder(handle_unknown='ignore')
df = pd.read_csv('data/cab_rides.csv')
df2 = pd.read_csv('data/weather.csv')
#categorical column names for encoding
categorical_columns = ['source','destination','product_id','name','cab_type']
#fetch the preprocessed dataframe
source_weather_df, destination_weather_df = weather_preprocessor(df2)
#initialize the encoder
en = OneHotEncoder(handle_unknown='ignore')
#fetch the preprocessed dataframe
cab_rides_df = cab_preprocessor(df,en)
#key of 'location+month+day+hour' to map to other df; drop duplicates in source and destionation weather df
source_weather_df = source_weather_df.drop_duplicates(subset=['key1'])
destination_weather_df = destination_weather_df.drop_duplicates(subset=['key2'])
#Merge cab rides with source weather df on key1
records = pd.merge(cab_rides_df, source_weather_df, on=['key1'])
#Merge with destionation weather df 
records = pd.merge(records,destination_weather_df, on=['key2'])
#Drop the key1 and key2 columns 
records.drop(['key1','key2'],inplace=True,axis=1)
records = encoder(records,categorical_columns,en)
print(records.head())

def linearRegressor(records):
    y = records['price']
    X = records.drop('price', axis=1)
    kf20 = KFold(n_splits=20, shuffle=False)
    model = LinearRegression()
    for train_index , test_index in kf20.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index],y[test_index]
        
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc = model.score(X_test , y_test)
        print(acc)

linearRegressor(records)