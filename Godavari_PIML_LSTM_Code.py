# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import matplotlib.dates as dates
from sklearn.model_selection import train_test_split

## Code for Godavari basin- can be extended to other basins as well 
basins = ['Godavari', 'Krishna', 'Cauvery']
station = ['PVM', 'VIJ', 'KDM']

i = 0

df_Final = pd.read_csv("E:\\PhD Datasets\\Final Observed Datasets\\"+str(basins[i])+"_"+str(station[i])+"_Obs.csv") ## Change the directory accordingly
df_Final = df_Final.set_index('Unnamed: 0')
############################   PIML MODEL FOR POLAVARAM STATION in GODAVARI BASIN ################################################################################################
########## SVM model simulate Excess Rainfall at POLAVARAM ########################################
# Define the input sequences and target variable
X = df_Final[['Rainfall', 'Temperature']].values
y = df_Final['ER'].values

# Assuming X and y are your input features (rainfall and temperature) and target variable (streamflow)
# Split the data into calibration and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

## Preprocess Data ###
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Support Vector Machine (SVM) model
svm_model = SVR(kernel='rbf', C= 10, gamma= 1, epsilon= 0.1) ## For Godavari -- best fit model obtained after hyperparameter search for GODAVARI
svm_model.fit(X_train_scaled, y_train)

import pickle
with open("E:\PhD PIML codes\SVM_ER_"+str(basins[i])+"_"+str(station[i])+".pkl", 'wb') as f: ## Change the directory accordigly
    pickle.dump(svm_model,f)
    

# Make predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)


# Plot the actual vs. predicted values for the training set
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_train, label='Observed')
plt.plot(y_train_pred, label='Predicted')
plt.title('SVM Excess Rainfall (Training Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Excess Rainfall(mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\ER_Train_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly

# Plot the actual vs. predicted values for the validation set
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Observed')
plt.plot(y_test_pred, label='Predicted')
plt.title('SVM Excess Rainfall (Testing Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Excess Rainfall(mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\ER_Test_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly


# Calculate Nash-Sutcliffe Efficiency (NSE)
def calculate_nse(observed, predicted):
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

#  Calculating KGE (Kling-Gupta Efficiency )
def calculate_kge(observed, predicted):
    r = np.corrcoef(observed, predicted)[0,1]
    alpha = (np.std(observed))/ (np.std(predicted))
    beta = (np.mean(observed))/ (np.mean(predicted))
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge


# Calculate NSE for the training and validation sets
train_nse = calculate_nse(y_train, y_train_pred)
val_nse = calculate_nse(y_test, y_test_pred)

# Calculate KGE for the training and testing sets
train_kge = calculate_kge(y_train, y_train_pred)
test_kge = calculate_kge(y_test, y_test_pred)

# Calculate correlation values
train_corr = np.corrcoef(y_train.squeeze(), y_train_pred.squeeze())[0, 1]
val_corr = np.corrcoef(y_test.squeeze(), y_test_pred.squeeze())[0, 1]

print('Training NSE:', train_nse)
print('Validation NSE:', val_nse)
print('Training KGE:', train_kge)
print('Validation KGE:', test_kge)
print('Training Correlation:', train_corr)
print('Validation Correlation:', val_corr)

correl_P1 = df_Final['Rainfall'].corr(df_Final['ER'])
correl_T1 = df_Final['Temperature'].corr(df_Final['ER'])

X_scaled = scaler.fit_transform(X)
y_predict = svm_model.predict(X_scaled)

# Plot the actual vs. predicted values for the entire time series
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y, label='Observed')
plt.plot(y_predict, label='Predicted')
plt.title('SVM simulated Excess Rainfall')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Excess Rainfall(mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\ER_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly

nse = calculate_nse(y, y_predict)
kge = calculate_kge(y, y_predict)
correl = np.corrcoef(y.squeeze(), y_predict.squeeze())[0, 1]
print('NSE', nse)
print('Correlation', correl)
print('KGE', kge)

df_Final['ER_PIML_simulated'] = y_predict

#################################################################################################################################################################################
##### Base FLow simulation using LSTM ######
X = df_Final[["Rainfall","Temperature","ER_PIML_simulated"]].values
#Y = df_Final["BF"].values
Y = df_Final[["BF", "QF", "SF"]].values

####### Normalizing the data ##################################################
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

######### Creating tensor form of training and testing sets for LSTM ##########
def create_dataset(X, Y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i: (i + time_steps)]
        Xs.append(v)
        ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(ys)

########## Fixing the time steps for memory cell ##############################
time_steps = 10
X_tensor, y_tensor = create_dataset(X_scaled, Y, time_steps)
print(X_tensor.shape, y_tensor.shape)

########## Splitting the dataset into training and testing sets ###############
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size= 0.2, random_state= 42) 

########## Building LSTM Model ################################################
model = Sequential()
model.add(LSTM(units= 50, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2]))) ## For Godavari -- best fit model obtained after hyperparameter search for GODAVARI
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units =1))
model.compile(optimizer= 'adam', loss = 'mean_squared_error')

########## Training the LSTM model ############################################
model.fit(X_train, y_train[:,0], epochs= 50, batch_size = 32, validation_split= 0.1)

model.save("E:\PhD PIML codes\LSTM_BF_"+str(basins[i])+"_"+str(station[i])+".keras") ## Change the directory accordigly
    
########## Evaluate the model #################################################
train_predictions = model.predict(X_train)
train_predictions[train_predictions < 0] = 0
BF_train = train_predictions
predictions = model.predict(X_test)
predictions[predictions < 0 ] = 0
BF_test = predictions

cal_nse = calculate_nse(y_train[:,0].flatten(), train_predictions.flatten())
print(f"Train NSE of Baseflow: {cal_nse}")
val_nse = calculate_nse(y_test[:,0].flatten(), predictions.flatten())
print(f"Test NSE of Baseflow: {val_nse}")

train_kge = calculate_kge(y_train[:,0].flatten(), train_predictions.flatten())
test_kge = calculate_kge(y_test[:,0].flatten(), predictions.flatten())
print('Training KGE:', train_kge)
print('Validation KGE:', test_kge)

train_corr = np.corrcoef(y_train[:,0].squeeze(),train_predictions.squeeze())[0, 1]
print(f"Train correlation of Baseflow: {train_corr}")
val_corr = np.corrcoef(y_test[:,0].squeeze(), predictions.squeeze())[0, 1]
print(f"Test correlation of Baseflow: {val_corr}")


########### Plotting training and testing sets ################################
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_test[:,0], label="Observed")
plt.plot(predictions, label="Predicted")
plt.title('LSTM Base Flow (Testing Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Baseflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\BF_Test_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_train[:,0], label="Observed")
plt.plot(train_predictions, label="Predicted")
plt.title('LSTM Base Flow (Training Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Baseflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\BF_Train_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()        
     
############ Predicting entire timeseries #####################################
simulation1 = model.predict(X_tensor)
simulation1[simulation1 < 0] = 0
nse = calculate_nse(y_tensor[:,0].flatten(), simulation1.flatten())
print(f"NSE: {nse}")

kge = calculate_kge(y_tensor[:,0].flatten(), simulation1.flatten())
print(f"KGE: {kge}")

corr = np.corrcoef(y_tensor[:,0].squeeze(), simulation1.squeeze())[0, 1]
print(f"Correlation of Baseflow: {corr}")

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(np.array(y_tensor), label="Observed")
plt.plot(np.array(simulation1), label="Predicted")
plt.title('LSTM simulated Base Flow')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Baseflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\BF_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

df_Final['BF_PIML_simulated'] = 0
df_Final['BF_PIML_simulated'][10:] = simulation1.flatten()

#######################################################################################################################
##### Quick flow simulation using LSTM ######
########## Building LSTM Model ################################################
model = Sequential()
model.add(LSTM(units= 100, activation = 'relu',  return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units =1))
model.compile(optimizer= 'adam', loss = 'mean_squared_error')

########## Training the LSTM model ############################################
model.fit(X_train, y_train[:,1], epochs= 50, batch_size = 32, validation_split= 0.1)

model.save("E:\PhD PIML codes\LSTM_QF_"+str(basins[i])+"_"+str(station[i])+".keras") ## Change the directory accordigly

########## Evaluate the model #################################################
train_predictions = model.predict(X_train)
QF_train = train_predictions
predictions = model.predict(X_test)
QF_test = predictions

cal_nse = calculate_nse(y_train[:,1].flatten(), train_predictions.flatten())
print(f"Train NSE of Quickflow: {cal_nse}")
val_nse = calculate_nse(y_test[:,1].flatten(), predictions.flatten())
print(f"Test NSE of Quickflow: {val_nse}")

train_kge = calculate_kge(y_train[:,1].flatten(), train_predictions.flatten())
test_kge = calculate_kge(y_test[:,1].flatten(), predictions.flatten())
print('Training KGE:', train_kge)
print('Validation KGE:', test_kge)

train_corr = np.corrcoef(y_train[:,1].squeeze(),train_predictions.squeeze())[0, 1]
print(f"Train correlation of Quickflow: {train_corr}")
val_corr = np.corrcoef(y_test[:,1].squeeze(), predictions.squeeze())[0, 1]
print(f"Test correlation of Quickflow: {val_corr}")

########### Plotting training and testing sets ################################
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Observed")
plt.plot(predictions, label="Predicted")
plt.title('LSTM Quick Flow (Testing Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Quickflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\QF_Test_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_train, label="Observed")
plt.plot(train_predictions, label="Predicted")
plt.title('LSTM Quick Flow (Training Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Quickflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\QF_Train_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()        
             
     
############ Predicting entire timeseries #####################################
simulation2 = model.predict(X_tensor)
nse = calculate_nse(y_tensor[:,1].flatten(), simulation2.flatten())
print(f"NSE: {nse}")

kge = calculate_kge(y_tensor[:,1].flatten(), simulation2.flatten())
print(f"KGE: {kge}")

corr = np.corrcoef(y_tensor[:,1].squeeze(), simulation2.squeeze())[0, 1]
print(f"Correlation of Quickflow: {corr}")

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(np.array(y_tensor), label="Observed")
plt.plot(np.array(simulation2), label="Predicted")
plt.title('LSTM simulated Quick Flow')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Quickflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\QF_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

df_Final['QF_PIML_simulated'] =0
df_Final['QF_PIML_simulated'][10:] = simulation2.flatten() 

##########################################################################################################################################################################################

Streamflow_train = BF_train + QF_train
Streamflow_test = BF_test + QF_test

sf_nse_train = calculate_nse(y_train[:,2].flatten(), Streamflow_train.flatten())
sf_kge_train = calculate_kge(y_train[:,2].flatten(), Streamflow_train.flatten())
corr_train = np.corrcoef(y_train[:,2].squeeze(), Streamflow_train.squeeze())[0, 1]

sf_nse_test = calculate_nse(y_test[:,2].flatten(), Streamflow_test.flatten())
sf_kge_test = calculate_kge(y_test[:,2].flatten(), Streamflow_test.flatten())
corr_test = np.corrcoef(y_test[:,2].squeeze(), Streamflow_test.squeeze())[0, 1]

Streamflow = (df_Final['BF_PIML_simulated'] +df_Final['QF_PIML_simulated']).values.flatten()
sf_obs = df_Final[["SF"]].values.flatten()

sf_nse = calculate_nse(sf_obs, Streamflow)
sf_kge = calculate_kge(sf_obs, Streamflow)
corr = np.corrcoef(sf_obs.squeeze(), Streamflow.squeeze())[0, 1]

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(sf_obs, label='Observed')
plt.plot(Streamflow, label='Predicted')
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rc('legend', fontsize = 14)
#plt.title('Observed vs. Predicted Stream flow')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Streamflow (mm)', fontsize = 16)
plt.legend()
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\PIML Simulations\\SF_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
print(sf_nse, sf_kge, corr)

df_Final['SF_PIML_Simulated'] = Streamflow

df_Final.to_csv("E:\\PhD Datasets\\PIML Results\\PIML_simulations_"+str(basins[i])+"_"+str(station[i])+"_updated.csv") ## Change the directory accordigly

#### DATA ANALYSIS 
correl_P1 = df_Final['Rainfall'].corr(df_Final['ER'])
correl_T1 = df_Final['Temperature'].corr(df_Final['ER'])
correl_P2 = df_Final['Rainfall'].corr(df_Final['BF'])
correl_T2 = df_Final['Temperature'].corr(df_Final['BF'])
correl_ER1 = df_Final['ER_PIML_simulated'].corr(df_Final['BF'])
correl_P3 = df_Final['Rainfall'].corr(df_Final['QF'])
correl_T3 = df_Final['Temperature'].corr(df_Final['QF'])
correl_ER2 = df_Final['ER_PIML_simulated'].corr(df_Final['QF'])
correl_P4 = df_Final['Rainfall'].corr(df_Final['SF'])
correl_T4 = df_Final['Temperature'].corr(df_Final['SF'])
coreel_P5 = df_Final['Rainfall'].corr(df_Final['Temperature'])

#######################################  LSTM ML model to simulate Streamflow    #####################################################################################

X = df_Final[["Rainfall","Temperature"]].values
Y = df_Final[["SF"]].values

#### Normalizing the data
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

#### Fixing the time steps for memory cell 
time_steps = 10
X_tensor, y_tensor = create_dataset(X_scaled, Y, time_steps)
print(X_tensor.shape, y_tensor.shape)

########## Splitting the dataset into training and testing sets ###############
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size= 0.2, random_state= 42) 

########## Building LSTM Model ################################################
model = Sequential()
model.add(LSTM(units= 125, activation = 'relu',  return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=75, return_sequences=True))
model.add(LSTM(units=75))
model.add(Dense(units =1))
model.compile(optimizer= 'adam', loss = 'mean_squared_error')

########## Training the LSTM model ############################################
model.fit(X_train, y_train, epochs= 50, batch_size = 32, validation_split= 0.1)

model.save("E:\PhD PIML codes\LSTM_SF_"+str(basins[i])+"_"+str(station[i])+".keras")

########## Evaluate the model #################################################
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)

cal_nse = calculate_nse(y_train, train_predictions)
print(f"Train NSE of Streamflow: {cal_nse}")
val_nse = calculate_nse(y_test, predictions)
print(f"Test NSE of Streamflow: {val_nse}")

train_kge = calculate_kge(y_train.flatten(), train_predictions.flatten())
test_kge = calculate_kge(y_test.flatten(), predictions.flatten())
print('Training KGE:', train_kge)
print('Validation KGE:', test_kge)

train_corr = np.corrcoef(y_train.squeeze(),train_predictions.squeeze())[0, 1]
print(f"Train correlation of Streamflow: {train_corr}")
val_corr = np.corrcoef(y_test.squeeze(), predictions.squeeze())[0, 1]
print(f"Test correlation of Streamflow: {val_corr}")

########### Plotting training and testing sets ################################
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Observed")
plt.plot(predictions, label="Predicted")
plt.title('LSTM Stream Flow (Testing Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Streamflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\LSTM Simulations\\LSTM_SF_Test_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(y_train, label="Observed")
plt.plot(train_predictions, label="Predicted")
plt.title('LSTM Stream Flow (Training Set)')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Streamflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\LSTM Simulations\\LSTM_SF_Train_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()  

SF_predict = model.predict(X_tensor)
        
nse = calculate_nse(y_tensor, SF_predict)
kge = calculate_kge(y_tensor.flatten(), SF_predict.flatten())
corr = np.corrcoef(y_tensor.squeeze(), SF_predict.squeeze())[0, 1]
print(f"NSE: {nse}")
print(f"KGE: {kge}")
print(f"Correlation of Streamflow: {corr}")

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(np.array(y_tensor), label="Observed")
plt.plot(np.array(simulation2), label="Predicted")
plt.title('LSTM simulated Stream Flow')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Streamflow (mm)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\"+str(basins[i])+"\\LSTM Simulations\\LSTM_SF_"+str(basins[i])+"_"+str(station[i])+".png", dpi = 600) ## Change the directory accordigly
plt.show()

df_Final['SF_LSTM_simulated'] =0
df_Final['SF_LSTM_simulated'][10:] = SF_predict.flatten() 

df_Final.to_csv("E:\\PhD Datasets\\PIML Results\\Hydrological_simulations_"+str(basins[i])+"_"+str(station[i])+".csv") ## Change the directory accordigly
