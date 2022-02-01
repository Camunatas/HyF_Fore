import pandas as pd
import tensorflow as tf
import numpy as np
import time
from datetime import timezone
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor as rf
import statsmodels.api as sm
import os
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
SARIMA_train_length = 5                     # [days] Training set length for SARIMA prediction
day = '2019-03-06'
# Random forest regressor parameters
day_rf = '2019-06-01'
# SARIMA parameters
model_order = (2,0,3)                       # Order
model_seasonal_order = (2,1,3,24)           # Seasonal order
# Storage variables
Predictions = {}
windspe_errors = []
Pgen_errors = []
save_folder = 'Tests plots' + '/March the 6th 2019 (only 6 forecasts)/'
#%% Importing data
dataset = pd.read_csv('wind_dataset.csv', sep=';',parse_dates=['DateTime'], index_col="DateTime")
#%% Auxiliary functions
# Train set splitter
def train_set_generator(day, dataset, hour, train_length):
    day_utc = pd.Timestamp(day)
    train_end = day_utc + pd.Timedelta('{}h'.format(hour-1))
    train_start = train_end - pd.Timedelta('{}d'.format(train_length))
    train_set = dataset[train_start:train_end].iloc[:, 2].values
    return train_set
# Hourly xlabel ticks for plotting
def hourly_xticks(hour):
    hour_ticks = []  # X axis dates label
    for i in range(hour, 24):  # Filling X axis dates label
        if i < 0:
            pass
        else:
            hour_ticks.append('{}:00'.format(i))
    return hour_ticks
#%% SARIMA prediction function
def SARIMA_pred(train_set, model_order, model_seasonal_order, hour):
    print('******************************************************************')
    print('Generating wind speed forecast, hour: {}'.format(hour))
    SARIMA_start = time.time()
    model = sm.tsa.SARIMAX(SARIMA_train, order=model_order, seasonal_order=model_seasonal_order,
                            initialization='approximate_diffuse',
                            enforce_stationarity=True,
                            enforce_invertibility=True)
    model_fit = model.fit(disp=False)
    print(f'Prediction generation time: {round(time.time() - SARIMA_start, 2)}s')
    prediction = model_fit.forecast(24-hour)
    if hour < 0:
        prediction = prediction[-24:]
    else:
        prediction = prediction[-(24-hour):]
    return prediction
#%% Generating random forest estimator
# Selecting training data
print('Generating wind power estimator with random forest')
rf_train_end = pd.Timestamp(day_rf) - pd.Timedelta('1h')
rf_train_start = rf_train_end - pd.Timedelta('100d')
rf_dataset = dataset[rf_train_start:rf_train_end]
rf_train_x = dataset.iloc[:, 2].values.reshape(-1, 1)
rf_train_y = dataset.iloc[:, 3].values
# Splitting train test sets
rf_X_train, rf_X_val, rf_y_train, rf_y_val = train_test_split(rf_train_x, rf_train_y, test_size=24, shuffle=False)
# Building forest
forest_start = time.time()
rf_WindPower = rf(bootstrap = False, max_depth = 10)
rf_WindPower.fit(rf_train_x, rf_train_y)
print(f'Estimator generation time: {round(time.time() - forest_start, 2)}s')

#%% Generating predictions
pred_start = time.time()
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# Generating prediction
for i, hour in enumerate(hours):
    # Generating training set
    SARIMA_train = train_set_generator(day, dataset, hour, SARIMA_train_length)
    # Generating test sets
    if hour < 1:
        test_start = pd.Timestamp(day)
    elif hour == 0:
        test_start = pd.Timestamp(day)
    else:
        test_start = pd.Timestamp(day) + pd.Timedelta('{}h'.format((hour)))
    test_end = pd.Timestamp('{} {}'.format(pd.Timestamp(day).strftime('%Y-%m-%d'), '23:00:00'))
    windspe_real = dataset[test_start:test_end].iloc[:, 2].values
    Pgen_real = dataset[test_start:test_end].iloc[:, 3].values
    # Generating wind speed prediction
    windspe_pred = SARIMA_pred(SARIMA_train, model_order, model_seasonal_order, hour)
    # Generating wind power estimation
    Pgen_pred = rf_WindPower.predict(windspe_pred.reshape(-1, 1))
    # Analizing and storing prediction results
    windspe_error = round(np.mean(windspe_real-windspe_pred),2)
    Pgen_error = round(np.mean(Pgen_real-Pgen_pred),2)
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
print('******************************************************************')
print('Predictions generation finished')
print(f'Elapsed time: {round((time.time() - pred_start)/60, 2)} min')
#%% Plotting results
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
    Pgen_pred_list = Predictions['Pgen_pred_{}'.format(labels[i])].tolist()
    while len(Pgen_pred_list) != len(hour_ticks):
        Pgen_pred_list.insert(0,None)               # Filling with nones to the left to adjust plot length
    plt.plot(Pgen_pred_list, label=f'{labels[i]}')
plt.plot(Predictions['Pgen_real'], '--', label='Observation')
plt.ylabel('Generated power (MW)')
plt.legend()
plt.grid()
plt.savefig(save_folder + 'Prediction results')
plt.show()
# Wind forecasting errort
fig = plt.figure('Wind forecasting error {}'.format(day))
ticks_x = np.arange(0, len(windspe_errors), 1)
plt.xticks(np.arange(0, len(windspe_errors), 1), hours)
plt.suptitle('Wind forecasting error {}'.format(day))
plt.plot(windspe_errors)
plt.xlabel('Forecasting hour')
plt.ylabel('Mean hourly deviation (m/s)')
plt.grid()
plt.savefig(save_folder + 'Wind forecasting error')
plt.show()
# Power forecasting error
fig = plt.figure('Power forecasting error {}'.format(day))
ticks_x = np.arange(0, len(windspe_errors), 1)
plt.xticks(np.arange(0, len(windspe_errors), 1), hours)
plt.suptitle('Power forecasting error {}'.format(day))
plt.plot(Pgen_errors)
plt.xlabel('Forecasting hour')
plt.ylabel('Mean hourly deviation (MW)')
plt.grid()
plt.savefig(save_folder + 'Power forecasting error')
plt.show()
#%% Saving wind power forecasts
Pgen_pred = {}
Pgen_pred['Day'] = day
Pgen_pred['hours'] = Predictions['hours']
for i, hour in enumerate(Predictions['hours']):
    Pgen_pred['Pgen_pred_{}'.format(labels[i])] = Predictions['Pgen_pred_{}'.format(labels[i])]
Pgen_pred['Pgen_real'] = Predictions['Pgen_real']
np.save('Pgen_pred.npy', Pgen_pred)
#%% DISABLED: Seasonal analyis of training set
# train_length = 10
# day = '2019-03-01'
# dataset = pd.read_csv('dataset.csv', sep=';',parse_dates=['DateTime'], index_col="DateTime")
# day_utc = pd.Timestamp(day)
# train_end = day_utc + pd.Timedelta('{}h'.format(hour-1))
# train_start = train_end - pd.Timedelta('{}d'.format(train_length))
# train_set = dataset[train_start:train_end].iloc[:, 2]
# dataset_und = train_set[~train_set.index.duplicated()]
# dataset = dataset_und.asfreq(freq='1h', method='bfill')
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(dataset)
# result.plot()
#%% DISABLED: SARIMA Grid Search
# import pmdarima as pm
# SARIMA_train = train_set_generator(day, dataset, -12, 15)
# model = pm.auto_arima(SARIMA_train, start_p=3, start_q=3,
# 					    m=24,start_P=0, max_p=5, max_q=5,
#                         seasonal=True,
# 						trace=True, error_action='ignore',
# 					    suppress_warnings=True,
#                         stepwise=True, n_fits=50)
# print(model.summary())
