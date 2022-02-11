' Dataset generator for initial project version'
#   Output structure:
#   Dict{[Day],[Pgen_pred],[Pgen_real],[DM_pred],[DM_real],[Dev_against_way],[Dev_against_coef],[ID_1],[ID_2],[ID_3],
#   [ID_4],[ID_5],[ID_6]}

#%% Importing libraries
import numpy as np
import pandas as pd

#%% Parameters
starting_day =  '2018-01-01 00:00:00'               # First day to evaluate
ending_day =  '2018-12-31 00:00:00'                 # last day to evaluate
Dataset = {}
#%% Loading external datasets
Prices_dataset = np.load('Price_pred.npy', allow_pickle=True).item()
Pgen_dataset = np.load('Pgen_pred.npy', allow_pickle=True).item()
ID1_df = pd.read_csv('Data/ID1.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
ID2_df = pd.read_csv('Data/ID2.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
ID3_df = pd.read_csv('Data/ID3.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
ID4_df = pd.read_csv('Data/ID4.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
ID5_df = pd.read_csv('Data/ID5.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
ID6_df = pd.read_csv('Data/ID6.csv', sep=';', usecols=['value','datetime'],
                     parse_dates=['datetime'], index_col="datetime")
Devs_df = pd.read_csv('Data/Deviations.csv', sep=';', usecols=['Way','Coefficient','Hour'],
                     parse_dates=['Hour'], index_col="Hour")
#%% Extracting data and saving data
day = starting_day

while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    # Generating daily data dictionary
    Daily_dict = {}
    daily_key = pd.Timestamp(day).strftime("%Y-%m-%d")
    # Storing generated power data
    Pgen_day_dict = Pgen_dataset[daily_key]
    for key in Pgen_day_dict:
        Daily_dict[f'{key}'] = Pgen_day_dict[f'{key}']
    # Storing DM price
    Daily_dict['Price_pred_DM'] = Prices_dataset[daily_key]
    # Storing ID prices
    Daily_dict['Price_real_ID2'] = ID2_df[f'{daily_key}'].iloc[:,0].values
    Daily_dict['Price_real_ID3'] = ID3_df[f'{daily_key}'].iloc[:,0].values
    Daily_dict['Price_real_ID4'] = ID4_df[f'{daily_key}'].iloc[:,0].values
    Daily_dict['Price_real_ID5'] = ID5_df[f'{daily_key}'].iloc[:,0].values
    Daily_dict['Price_real_ID6'] = ID6_df[f'{daily_key}'].iloc[:,0].values
    # Storing deviation data
    Daily_dict['Dev_against_way'] = Devs_df[f'{daily_key}'].iloc[:,0].values
    Daily_dict['Dev_against_coef'] = Devs_df[f'{daily_key}'].iloc[:,1].values
    # Storing on general dataset
    Dataset[pd.Timestamp(day).strftime("%Y-%m-%d")] = Daily_dict
    print(f"Saved day {day}")
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')

np.save('Dataset.npy', Dataset)
#%% Saving data into dictionary
