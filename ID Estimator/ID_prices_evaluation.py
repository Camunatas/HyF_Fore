import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timezone
from matplotlib import pyplot as plt
import os
#%% Importing csvs
DM_df = pd.read_csv('Datasets/Prices.csv', sep=';', usecols=['DM','datetime'], parse_dates=['datetime'], index_col="datetime")
ID1_df = pd.read_csv('Datasets/ID1.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")
ID2_df = pd.read_csv('Datasets/ID2.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")
ID3_df = pd.read_csv('Datasets/ID3.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")
ID4_df = pd.read_csv('Datasets/ID4.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")
ID5_df = pd.read_csv('Datasets/ID5.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")
ID6_df = pd.read_csv('Datasets/ID6.csv', sep=';', usecols=['value','datetime'], parse_dates=['datetime'], index_col="datetime")

#%% Comparing MD & ID prices
def ID_compartor(ID, ID_df, DM_df, date_init, date_end, foldername):
    # Adjusting initial and end dates in case there is no ID at such hour
    while datetime.fromtimestamp(date_init.value / 1e9) not in ID_df.index:
        date_init = date_init + pd.Timedelta('1h')
    while datetime.fromtimestamp(date_end.value / 1e9) not in ID_df.index:
        date_end = date_end - pd.Timedelta('1h')
    date_init = date_init + pd.Timedelta('1h')
    date_end = date_end - pd.Timedelta('1h')
    # Slicing dataframes
    DM_prices = DM_df[date_init:date_end].iloc[:,0]
    ID_prices = ID_df[date_init:date_end].iloc[:,0]
    # Comparing prices
    difference = []
    for a,b in zip(DM_prices.values, ID_prices.values):
        difference.append(a-b)
    difference_df = pd.DataFrame({"Difference betwen ID1 and DM": difference}, index = ID_prices.index)
    # Creating figures folder
    if not os.path.exists(f"Histograms/{foldername}"):
        os.makedirs(f"Histograms/{foldername}")
    if not os.path.exists(f"Lineplots/{foldername}"):
        os.makedirs(f"Lineplots/{foldername}")
    fig = difference_df.plot(kind='hist',xlabel='Difference (€/MWh)',bins=100, legend=0, grid=1,
                             title=f'Histogram for difference between MD and ID{ID}').get_figure()
    fig.savefig(f'Histograms/{foldername}/ID{ID}.png')
    fig = difference_df.plot(xlabel='Date', ylabel='Difference (€/MWh)', legend=0, grid=1,
                       title=f'Difference between MD and ID{ID}').get_figure()
    fig.savefig(f'Lineplots/{foldername}/ID{ID}.png')

# Calling comparator
date_init = pd.Timestamp('2019-03-06 00:00:00') - pd.Timedelta('7d')
date_end = pd.Timestamp('2019-03-06 23:00:00') - pd.Timedelta('1d')
folder_name = "1 week before March the 6th 2016"
ID_compartor(1, ID1_df, DM_df, date_init, date_end, folder_name)
ID_compartor(2, ID2_df, DM_df, date_init, date_end, folder_name)
ID_compartor(3, ID3_df, DM_df, date_init, date_end, folder_name)
ID_compartor(4, ID4_df, DM_df, date_init, date_end, folder_name)
ID_compartor(5, ID5_df, DM_df, date_init, date_end, folder_name)
ID_compartor(6, ID6_df, DM_df, date_init, date_end, folder_name)





