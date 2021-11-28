#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pytz
from sklearn.preprocessing import OneHotEncoder

def preprocessing(df,en): 
    #convert "cab_type" into binary, Lyft = 0, Uber = 1 & replace
    df['cab_type'] = df['cab_type'].map(dict(Lyft = 0, Uber = 1))
    
    #convert from epoch time to EST timezone since data was originally from Boston
    est_time = pd.to_datetime(df['time_stamp'], unit='ms').dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    
    #parse data into year, month, day, hour, minute columns
    df['year'] = est_time.apply(lambda x: pd.Timestamp(x).year)
    df['month'] = est_time.apply(lambda x: pd.Timestamp(x).month)
    df['day'] = est_time.apply(lambda x: pd.Timestamp(x).day)
    df['hour'] = est_time.apply(lambda x: pd.Timestamp(x).hour)
    df['minute'] = est_time.apply(lambda x: pd.Timestamp(x).minute)
    
    #parse data into weekend column
    df['weekend'] = est_time.dt.day_name().map(dict(Monday = 0, Tuesday = 0, Wednesday = 0, Thursday = 0, Friday = 0, Saturday = 1, Sunday=1))
    
    #use one hot encoder to encode source and destination and name of the cab
    en_df = pd.DataFrame(en.fit_transform(df[['source','destination','name']]).toarray(),columns=[list(en.get_feature_names())])
    #concat the extracted encoded columns to the main dataframe 
    df = pd.concat([df,en_df], axis=1)
    #drop categorical data from the df
    df.drop(['source','destination','id','name','product_id'], inplace=True, axis=1)
    #output
    return df



def main():
    #initialize the encoder
    en = OneHotEncoder(handle_unknown='ignore')
    df = pd.read_csv('data/cab_rides.csv')
    #fetch the preprocessed dataframe
    ndf = preprocessing(df,en)
    print(ndf.head())

if __name__ == '__main__':
    main()