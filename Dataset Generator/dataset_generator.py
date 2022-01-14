' Dataset generator for initial project version'
#   Output structure:
#   Dict{[Day],[Pgen_pred],[Pgen_real],[DM_pred],[DM_real],[Dev_against_way],[Dev_against_coef],[ID_1],[ID_2],[ID_3],
#   [ID_4],[ID_5],[ID_6]}

#%% Importing libraries
import numpy as np
import pandas as pd

#%% Loading external datasets
Prices_dataset = np.load('Prices_pred.npy', allow_pickle=True).item()
Pgen_dataset = np.load('Pgen_pred.npy', allow_pickle=True).item()

#%% Extracting data and saving data
Dataset = {}
Dataset['Day'] = Pgen_dataset['Day']
Day = Dataset['Day']
Dataset['Pgen_pred'] = Pgen_dataset['Pgen_pred_-12']
Dataset['Pgen_real'] = Pgen_dataset['Pgen_real']
Dataset['DM_pred'] = Prices_dataset[f'{Day}'][2]
Dataset['DM_real'] = Prices_dataset[f'{Day}'][3]
Dataset['Dev_against_way'] = Prices_dataset[f'{Day}'][4]
Dataset['Dev_against_coef'] = Prices_dataset[f'{Day}'][5]
Dataset['ID1'] = pd.read_csv('ID1.csv',sep=';', usecols=['value']).values
Dataset['ID2'] = pd.read_csv('ID2.csv',sep=';', usecols=['value']).values
Dataset['ID3'] = pd.read_csv('ID3.csv',sep=';', usecols=['value']).values
Dataset['ID4'] = pd.read_csv('ID4.csv',sep=';', usecols=['value']).values
Dataset['ID5'] = pd.read_csv('ID5.csv',sep=';', usecols=['value']).values
Dataset['ID6'] = pd.read_csv('ID6.csv',sep=';', usecols=['value']).values
np.save('Dataset.npy', Dataset)
#%% Saving data into dictionary
