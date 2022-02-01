# Generate dataset
'Dict{[Day],[Gen_fore],[Gen_real],[Price_fore],[Price,real],[Dev_against_way],[Dev_against_coef]}'

#%% Importing libraries & functions
import pandas as pd
from forecast_fcns import *
from datetime import timezone
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
#%% General parameters
first_day = '2019-03-06'        # Starting day YYYY-MM-DD
last_day = '2019-03-06'         # Ending day YYYY-MM-DD
train_length = 100              # [Days] Training data length for ANN
Dataset = {}                    # Dataset dictionary
#%% Loading datasets
# Generation dataset
# Day-ahead market prices dataset
prices_df = pd.read_csv('data/DM_Prices.csv', sep=';', usecols=['Hour','Price D-6','Price D-1', 'Price D'],
                        parse_dates=['Hour'], index_col="Hour")
# Deviation  dataset
deviations_df = pd.read_csv('data/Deviations.csv', sep=';', usecols=['Hour','Way','Coefficient'],
                        parse_dates=['Hour'], index_col="Hour")
#%% Creating neural network for price prediction
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
ann.compile(optimizer=opt, loss='mean_absolute_percentage_error')
#%%  Generating price dataset
day = pd.Timestamp(first_day)
while day != pd.Timestamp(last_day) + pd.Timedelta('1d'):
    # Creating daily timedeltas
    day_start = day.replace(tzinfo=timezone.utc)
    day_end = day_start + pd.Timedelta('23h')

    # Generating price forecast & saving day-ahead prices data
    df_slice_end = day_start + pd.Timedelta('22h')
    df_slice_start = df_slice_end - pd.Timedelta('{}d 24h'.format(100))
    prices_df_slice = prices_df[df_slice_start:df_slice_end]
    X = prices_df_slice.iloc[:, 0:2].values
    y = prices_df_slice.iloc[:,2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 24, random_state = 0, shuffle=False)
    ann.fit(X_train, y_train, batch_size=32, epochs=100,verbose=0)
    y_pred_ANN = ann.predict(X_test)
    y_pred_ANN = y_pred_ANN.tolist()
    DM_Price_Fore = np.array([round(a[0], 2) for a in y_pred_ANN])
    DM_Price_Real = y_test

    # Saving deviation data
    Dev_Way = deviations_df[day_start:day_end].iloc[:, 0].values
    Dev_Coef = deviations_df[day_start:day_end].iloc[:, 1].values

    # Saving daily data
    Dataset[day.strftime("%Y-%m-%d")] = [DM_Price_Fore, DM_Price_Real,
                                         Dev_Way, Dev_Coef]
    print("*********************************************************************************")
    print("Day: {}".format(day))
    day = day + pd.Timedelta('1d')

#%% Saving dataset
np.save('data/Prices_pred.npy', Dataset)

