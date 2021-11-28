import pandas as pd
import numpy as np
import pytz
from sklearn.preprocessing import OneHotEncoder

def cab_preprocessor(df): 
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
    #Getting average value of the weather for source and destination places
    avg_weather_df = df.groupby('location').mean().reset_index(drop=False)
    avg_weather_df = avg_weather_df.drop('time_stamp', axis=1)
    df2 = avg_weather_df.rename(
        columns={
            'location': 'source',
            'temp': 'source_temp',
            'clouds': 'source_clouds',
            'pressure': 'source_pressure',
            'rain': 'source_rain',
            'humidity': 'source_humidity',
            'wind': 'source_wind'
        }
    )
    df3 = avg_weather_df.rename(
        columns={
            'location': 'destination',
            'temp': 'destination_temp',
            'clouds': 'destination_clouds',
            'pressure': 'destination_pressure',
            'rain': 'destination_rain',
            'humidity': 'destination_humidity',
            'wind': 'destination_wind'
        }
    )
    return df2, df3


#initialize the encoder
en = OneHotEncoder(handle_unknown='ignore')
df = pd.read_csv('data/cab_rides.csv')
df2 = pd.read_csv('data/weather.csv')
#filling weather df with zeroes for missing values
df2 = df2.fillna(0)
categorical_columns = ['source','destination','product_id','name','cab_type']
#fetch the preprocessed dataframe
source_weather_df,destination_weather_df = weather_preprocessor(df2)
#initialize the encoder
en = OneHotEncoder(handle_unknown='ignore')
#fetch the preprocessed dataframe
cab_rides_df = cab_preprocessor(df)
print(len(cab_rides_df))
#merge the cab df with source weather df on the source column
records = pd.merge(cab_rides_df, source_weather_df, on='source')
#merge the cab df with destination weather df on the destination column
records = pd.merge(records,destination_weather_df,on='destination')
#call one hot encoder
records = encoder(records,categorical_columns,en)
print(len(records))
print(records.head())
