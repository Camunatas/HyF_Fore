import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import os
# from forecast_fcns import *
from aux_fcns import *
from forecast_fcns import *
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling Statsmodels warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
#%% General parameters
# Forecast parameters
hours = [-12, -4, 1, 4, 8, 12]
labels = ['DM', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6']
# hours = [-12]
# labels = ['DM']
SARIMA_train_length = 3     # [days] Training set length for SARIMA prediction
day_start = '2018-03-01'
day_end = '2018-03-01'
day_start_ts = pd.Timestamp(day_start)
day_end_ts = pd.Timestamp(day_end) + pd.Timedelta('23h')
# Data import parameters
df_day_start = '2017-01-01'
df_day_end = '2018-12-31'
df_day_start_ts = pd.Timestamp(df_day_start)
df_day_end_ts = pd.Timestamp(df_day_end) + pd.Timedelta('23h')
# Random forest regressor parameters
day_rf = day_start
rf_train_length = 100
# SARIMA parameters
model_order = (2,0,3)                       # Order
model_seasonal_order = (2,1,3,24)           # Seasonal order
# Storage variables
Pgen_pred_dict = {}
Predictions = {}
windspe_errors = []
Pgen_errors = []
save_folder = 'Tests plots' + '/March the 6th 2019 (only 6 forecasts)/'
#%% Preparing dataset
# Importing data
dataset_raw = pd.read_csv('Data/Sotavento Historical Data.csv', sep=';',parse_dates=['Date'], index_col="Date")
# Slicing data
dataset = dataset_raw[df_day_start_ts:df_day_end_ts]
# Removing nans
nans = dataset.isna().sum()
print('Amount of nans: ')
print(nans)
dataset = dataset.fillna(method='ffill')
#%% WORKBENCH: Implementing wind turbine power curve
def WTG_curve(windspeed):
    # Enercon E-126
    speed = [0,1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20, 21,22,23,24,25]
    power = [0,0,0,0.1,0.3,0.4, 0.8,1.2,1.8,3,3.6, 4.8,5.65,6.4,7,7.3, 7.58,7.58,7.58,7.58,7.58, 7.58,7.58,7.58,7.58,7.58]
    WTG_curve = {}
    for p,v in enumerate(speed):
        WTG_curve[f'{v}'] = power[p]
    Pgen = []
    for v in windspeed:
        if v < 0 or v > speed[-1]:
            Pgen.append(WTG_curve[f'{0}'])
        else:
            Pgen.append(WTG_curve[f'{round(v)}']*5)
    return(Pgen)
#%% Generate wind power prediction
day = day_start_ts
pred_start = time.time()
while day != day_end_ts + pd.Timedelta('1h'):
    print(f'Generating prediction for {day.strftime("%Y-%m-%d")}')
    day_pred_start = time.time()
    Predictions = {}
    for i, hour in enumerate(hours):
        hour_rf = pd.Timestamp(day) + pd.Timedelta(f'{hour}h')
        # Generating training sets
        SARIMA_train = SARIMA_train_set_gen(day, dataset, hour, SARIMA_train_length)
        # Generating test sets
        if hour < 1:
            test_start = pd.Timestamp(day)
        elif hour == 0:
            test_start = pd.Timestamp(day)
        else:
            test_start = pd.Timestamp(day) + pd.Timedelta('{}h'.format((hour)))
        test_end = pd.Timestamp('{} {}'.format(pd.Timestamp(day).strftime('%Y-%m-%d'), '23:00:00'))
        windspe_real = dataset[test_start:test_end].iloc[:, 0].values
        windspe_real = windspe_real.astype(np.float)
        Pgen_real = WTG_curve(windspe_real)
        # Generating wind speed prediction
        windspe_pred = windspe_predictor(SARIMA_train, model_order, model_seasonal_order, hour)
        # Generating wind power estimation
        Pgen_pred = WTG_curve(windspe_pred)
        # Analizing and storing prediction results
        windspe_error = round(np.mean(windspe_real-windspe_pred),2)
        Pgen_error = round(np.mean(np.array(Pgen_real)-np.array(Pgen_pred)),2)
        windspe_errors.append(windspe_error)
        Pgen_errors.append(Pgen_error)
        # Saving real values (only in first iteration)
        if i == 0:
            Predictions['Pgen_real'] = Pgen_real
            Predictions['windspe_real'] = windspe_real
        # Saving prediction arrays
        Predictions['Pgen_pred_{}'.format(labels[i])] = Pgen_pred
        Predictions['windspe_pred_{}'.format(labels[i])] = windspe_pred
        # Printing results
        print("Wind speed prediction error: {} m/s".format(windspe_error))
        print("Mean hourly generation deviation: {} MWh".format(Pgen_error))
    Predictions['hours'] = hours
    Pgen_pred_dict[day.strftime("%Y-%m-%d")] = Predictions
    print(f'Day elapsed time: {round(time.time() - day_pred_start, 2)}s')
    day = day + pd.Timedelta('1d')
print(f'Day elapsed time: {round((time.time() - pred_start)/3600, 2)}h')

#%% Plotting results
day = day_start
Predictions = Pgen_pred_dict[day]
fig = plt.figure('Prediction results for {}'.format(day))
plt.suptitle('Prediction results for {}'.format(day))
hour_ticks = hourly_xticks(Predictions['hours'][0])
# Wind speed forecasting subplot
windspe_plot= fig.add_subplot(2, 1, 1)
ticks_x = np.arange(0, len(hour_ticks), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(hour_ticks), 1), '', rotation=45)
for i,hour in enumerate(Predictions['hours']):
    windspe_pred_list = Predictions['windspe_pred_{}'.format(labels[i])].tolist()
    while len(windspe_pred_list) != len(hour_ticks):
        windspe_pred_list.insert(0,None)            # Filling with nones to the left to adjust plot length
    plt.plot(windspe_pred_list, label=f'{labels[i]}')
plt.plot(Predictions['windspe_real'], '--', label='Observation')
plt.ylabel('Wind speed (m/s)')
plt.legend()
plt.grid()
# Generated power subplot
pgen_plot= fig.add_subplot(2, 1, 2)
ticks_x = np.arange(0, len(hour_ticks), 1)
plt.xticks(np.arange(0, len(hour_ticks), 1), hour_ticks, rotation=45)
for i, hour in enumerate(Predictions['hours']):
    Pgen_pred_list = Predictions['Pgen_pred_{}'.format(labels[i])]
    while len(Pgen_pred_list) != len(hour_ticks):
        Pgen_pred_list.insert(0,None)               # Filling with nones to the left to adjust plot length
    plt.plot(Pgen_pred_list, label=f'{labels[i]}')
plt.plot(Predictions['Pgen_real'], '--', label='Observation')
plt.ylabel('Generated power (MW)')
plt.legend()
plt.grid()
plt.savefig(save_folder + 'Prediction results')
plt.show()
