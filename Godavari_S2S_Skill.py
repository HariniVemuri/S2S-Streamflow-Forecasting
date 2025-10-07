# -*- coding: utf-8 -*-
## S2S Rainfall forecast skill verification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import calendar
import datetime
from datetime import date, timedelta, datetime

def get_thursdays(year, month):
    # Create a calendar object
    cal = calendar.Calendar()

    # List to store the day numbers of Thursdays
    thursdays = []

    # Iterate over the days of the specified month
    for day in cal.itermonthdays(year, month):
        if day != 0:  # Skip days outside the specified month
            date = calendar.weekday(year, month, day)
            if date == calendar.THURSDAY:
                thursdates = datetime(year, month, day).date()
                thursdays.append(thursdates)
    return thursdays

model_name = ['NCEP','ECMWF', 'UKMO', 'ECCC', 'CMA']
df_fst_final = pd.DataFrame()

for iter_year in range(2016,2019):
    for iter_month in range(6,11):

        thursdays = get_thursdays(iter_year, iter_month)   
        
        for j in range(0, len(thursdays)):
            df_forecast_2 = pd.DataFrame()
            for i in range(0,5):
                df_forecast = pd.read_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\"+str(model_name[i])+" PVM\\"+str(thursdays[j])+"_SF_PVM_forecast.csv")  ## Change the directory accordingly
                #df_forecast_1 = df_forecast.loc[(df_forecast['Streamflow'] >= tsd_1) | (df_forecast['SF_S2S_PIML'] >= tsd_1) | (df_forecast['SF_S2S_LSTM'] >= tsd_1) | (df_forecast['SF_S2S_IHACRES'] >= tsd_1)]
                df_forecast_1 = df_forecast[['Date', 'Rainfall', 'IMD']]
                df_forecast_2 = pd.concat([df_forecast_2, df_forecast_1], axis = 1)

            df_forecast_2.drop(df_forecast_2.iloc[:, [0,3,6,9,12]], inplace=True, axis=1)
            df_fst = pd.DataFrame()   
            df_fst = df_forecast_2
            df_fst['Date'] = df_forecast_1['Date']
            #df_fst = df_fst.set_index('Date')  
            
            df_fst_final = pd.concat([df_fst_final, df_fst], axis = 0, ignore_index = True)
            

df_rain_hist = pd.read_csv("E:\PhD Datasets\IMD Rainfall Peninsular\Godavari_PVM_Avg_PPT.csv") ## Change the directory accordingly
df_rain_hist['Unnamed: 0'] = pd.to_datetime(df_rain_hist['Unnamed: 0'])
df_rain_hist['day'] = df_rain_hist['Unnamed: 0'].dt.day
df_rain_hist['month'] = df_rain_hist['Unnamed: 0'].dt.month
df_rain_hist['Rain_clim'] = df_rain_hist.groupby(["day", "month"])['Rainfall'].transform('mean')
df_rain_hist['Date'] = df_rain_hist['Unnamed: 0']
df_rain_hist['Date'] = pd.to_datetime(df_rain_hist['Date'])

historical_values_sf = df_rain_hist['Rainfall']
historical_values_sf_sorted = historical_values_sf.sort_values()
total_values = len(historical_values_sf_sorted)
probabilities = np.array([(i + 1) / total_values for i in range(total_values)])
hist_sf_idx_1 =  np.array(np.where(np.isclose(probabilities, 0.90, atol = 0.00005))).flatten()
tsd_1 = (historical_values_sf_sorted.values[hist_sf_idx_1]).mean()

df_rain_fst = df_fst_final.iloc[:,[0,2,4,6,8,9,10]]
df_rain_fst['Date'] = pd.to_datetime(df_rain_fst['Date'])
df_rain_fst = pd.merge(df_rain_fst, df_rain_hist[['Date','Rain_clim']], on = 'Date', how = 'left')
df_rain_fst.to_csv("E:\\PhD Datasets\\S2S Datasets Rainfall\\S2S_PVM\\S2S_PCP_PVM_Godavari.csv")

df_rain_fst_binary = pd.DataFrame(np.where(df_rain_fst[['Rainfall','IMD','Rain_clim']] >= tsd_1,1, 0 ))
df_rain_fst_binary.columns = ['NCEP','ECMWF', 'UKMO', 'ECCC', 'CMA','IMD','Rain_clim']
df_rain_fst_binary['Ensemble_rain'] =  df_rain_fst_binary.iloc[:, :5].mean(axis = 1)
df_rain_fst_binary['Date'] = df_rain_fst['Date']

df_rain_fst_binary.to_csv("E:\\PhD Datasets\\S2S Datasets Rainfall\\S2S_PVM\\S2S_Binary_PCP_PVM_Godavari.csv") ## Change the directory accordingly

df_rain_fst_binary = pd.read_csv("E:\\PhD Datasets\\S2S Datasets Rainfall\\S2S_PVM\\S2S_Binary_PCP_PVM_Godavari.csv") ## Change the directory accordingly

## Developing ROC curve
# True binary labels
y_true = df_rain_fst_binary['IMD']
# Forecasted probabilities
y_scores = df_rain_fst_binary['Ensemble_rain']
y_clim = df_rain_fst_binary['Rain_clim']
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fpr_c, tpr_c, thresholds = roc_curve(y_true, y_clim)
# Compute ROC AUC
roc_auc = auc(fpr, tpr)
roc_auc_c = auc(fpr_c, tpr_c)

plt.figure(figsize = (8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Ensemble (AUC = %0.2f)' % roc_auc)
plt.plot(fpr_c, tpr_c, color='red', lw=2, label='Climatology (AUC = %0.2f)' % roc_auc_c)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
#plt.title('Receiver Operating Characteristic (ROC) Curve' , fontsize = 22, pad = 20)
plt.legend(loc='lower right', fontsize = 16)
plt.tick_params(axis='both', labelsize=14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Rainfall skill\\S2S_90_rain_Clim_PVM_roc.png", dpi = 600) ## Change the directory accordingly
plt.show()

### Calculating no.of rainfall values > threshold in observed rainfall data
df_rain_fst_binary_1 = pd.DataFrame(df_rain_fst_binary['IMD'].groupby(df_rain_fst_binary['Date']).sum())
530 - (df_rain_fst_binary_1['IMD'] == 0).sum()

from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
# Convert probabilities to binary labels using a threshold (e.g., 0.5)
# Convert probabilities to binary labels using a threshold (e.g., 0.5)
threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)
y_pred_clim = (y_clim >= threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_c = confusion_matrix(y_true, y_pred_clim)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Rainfall skill\\S2S_90_rain_PVM_cm.png", dpi = 600) ## Change the directory accordingly
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_c,
                              display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Rainfall skill\\S2S_90_rain_Clim_PVM_cm.png", dpi = 600) ## Change the directory accordingly
plt.show()

### ROC for different lead times ############################################################################################################################################################
roc = np.empty((32,1))
roc_c = np.empty((32,1))
cm_all = np.empty((2,2,32))
for i in range(0,32):
    idx = np.arange(0+i, 2080+i, 32)
    df_rain_fst_lead = df_rain_fst_binary.iloc[idx] 
      
    y_true = df_rain_fst_lead['IMD']
    # Forecasted probabilities
    y_scores = df_rain_fst_lead['Ensemble_rain']
    y_clim = df_rain_fst_lead['Rain_clim']
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fpr_c, tpr_c, thresholds = roc_curve(y_true, y_clim)
    # Compute ROC AUC
    roc_auc = auc(fpr, tpr)
    roc_auc_c = auc(fpr_c, tpr_c)
    
    roc[i] = roc_auc
    roc_c[i] = roc_auc_c


from numpy.polynomial.polynomial import Polynomial
x = np.arange(1,33)
roc = np.asarray(roc).flatten()
poly_coeffs = Polynomial.fit(x, roc, deg=1)
trend_line = poly_coeffs(x)
true_coeffs = poly_coeffs.convert().coef
slope = true_coeffs[1]
intercept = true_coeffs[0]

# Format the equation of the trendline
equation = f"y = {slope:.2f}x + {intercept:.2f}"

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(x, roc,  marker='o', linestyle='-', label = 'AUC')
plt.plot(x, trend_line, color='red', linestyle='--', label='Trend line')
plt.text(17, 0.8 * max(roc), equation, fontsize=14, color='red')
plt.xlabel('Lead Time', fontsize = 16)
plt.ylabel('Area Under Curve(AUC)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Rainfall skill\\S2S_90_rain_PVM_roclead.png", dpi = 600)  ## Change the directory accordingly

roc_c = np.asarray(roc_c).flatten()
roc_ratio = roc/roc_c
poly_coeffs = Polynomial.fit(x, roc_ratio, deg=1)
trend_line = poly_coeffs(x)
true_coeffs = poly_coeffs.convert().coef
slope = true_coeffs[1]
intercept = true_coeffs[0]

# Format the equation of the trendline
equation = f"y = {slope:.2f}x + {intercept:.2f}"

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(x, roc_ratio,  marker='o', linestyle='-', label = 'AUC')
plt.plot(x, trend_line, color='red', linestyle='--', label='Trend line')
plt.text(17, 0.8 * max(roc_ratio), equation, fontsize=14, color='red')
plt.xlabel('Lead Time', fontsize = 16)
plt.ylabel('AUC_S2S/AUC_Clim', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Rainfall skill\\S2S_90_rain_PVM_roclead_clim.png", dpi = 600)  ## Change the directory accordingly


## S2S Streamflow forecast skill verification ############################################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import calendar
import datetime
from datetime import date, timedelta, datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import seaborn as sns


###### NCEP datasets is used for trial run########

def get_thursdays(year, month):
    # Create a calendar object
    cal = calendar.Calendar()

    # List to store the day numbers of Thursdays
    thursdays = []

    # Iterate over the days of the specified month
    for day in cal.itermonthdays(year, month):
        if day != 0:  # Skip days outside the specified month
            date = calendar.weekday(year, month, day)
            if date == calendar.THURSDAY:
                thursdates = datetime(year, month, day).date()
                thursdays.append(thursdates)
    return thursdays

## Getting 90th percentile streamflow 
df_hist = pd.read_csv("E:\PhD Datasets\PIML Results\Hydrological_simulations_Godavari_PVM.csv") ## Change the directory accordingly

historical_values_sf = df_hist['SF']
historical_values_sf_sorted = historical_values_sf.sort_values()
total_values = len(historical_values_sf_sorted)
probabilities = np.array([(i + 1) / total_values for i in range(total_values)])
hist_sf_idx_1 =  np.array(np.where(np.isclose(probabilities, 0.90, atol = 0.00005))).flatten()
tsd_1 = (historical_values_sf_sorted.values[hist_sf_idx_1]).mean()

model_name = ['NCEP','ECMWF', 'UKMO', 'ECCC', 'CMA']
df_fst_final = pd.DataFrame()
df_fst_final_binary = pd.DataFrame()

for iter_year in range(2016,2019):
    for iter_month in range(6,11):

        thursdays = get_thursdays(iter_year, iter_month)   
        
        for j in range(0, len(thursdays)):
            df_forecast_2 = pd.DataFrame()
            for i in range(0,5):
                df_forecast = pd.read_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\"+str(model_name[i])+" PVM\\"+str(thursdays[j])+"_SF_PVM_forecast.csv") ## Change the directory accordingly
                #df_forecast_1 = df_forecast.loc[(df_forecast['Streamflow'] >= tsd_1) | (df_forecast['SF_S2S_PIML'] >= tsd_1) | (df_forecast['SF_S2S_LSTM'] >= tsd_1) | (df_forecast['SF_S2S_IHACRES'] >= tsd_1)]
                df_forecast_1 = df_forecast[['Date', 'Streamflow', 'SF_S2S_PIML', 'SF_S2S_LSTM', 'SF_S2S_IHACRES']]
                df_forecast_2 = pd.concat([df_forecast_2, df_forecast_1], axis = 1)

            df_forecast_2.drop(df_forecast_2.iloc[:, [5,6,10,11,15,16,20,21]], inplace=True, axis=1)
            df_fst = pd.DataFrame()   
            df_fst = df_forecast_2
            df_fst['SF_obs'] = df_forecast_1['Streamflow']
            df_fst['Date'] = df_forecast_1['Date']
            df_fst_final = pd.concat([df_fst_final, df_fst], axis = 0, ignore_index = True)
            df_fst = df_fst.set_index('Date')   
    
            arr_fst = np.where(df_fst >= tsd_1, 1, 0)
            arr_fst = pd.DataFrame(arr_fst, columns= df_fst.columns)
            arr_fst['Date'] = df_forecast_1['Date']
            arr_fst['Ensemble'] =  arr_fst.iloc[:, :15].mean(axis = 1)
            
            df_fst_final_binary = pd.concat([df_fst_final_binary, arr_fst], axis = 0, ignore_index = True)

df_fst_final.to_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\S2S PVM\\S2S_90_SF_PVM_Godavari.csv") ## Change the directory accordingly
df_fst_final_binary.to_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\S2S PVM\\S2S_90_SF_binary_PVM_Godavari.csv") ## Change the directory accordingly


##### Extreme Streamflow Verification for 90th percentile ################################################################################################################
### Calulation climatological streamflow forecast for S2S time period ###############################
df_hist['Date'] = df_hist['Unnamed: 0']
df_hist['Date'] = pd.to_datetime(df_hist['Date'], format='%d-%m-%Y')
# Access the day of the month
df_hist['day'] = df_hist['Date'].dt.day
df_hist['month'] = df_hist['Date'].dt.month
df_hist['Q_clim'] = df_hist.groupby(["day", "month"])['SF'].transform('mean')

df_fst_final['Date'] = pd.to_datetime(df_fst_final['Date'])
df_fst_final = pd.merge(df_fst_final, df_hist[['Date','Q_clim']], on = 'Date', how = 'left')
df_fst_final.to_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\S2S PVM\\S2S_90_SF_PVM_Godavari.csv") ## Change the directory accordingly
### 90th percentile is fixed as threshold and converted the data into binary format ####################

df_fst_binary_90 = np.where(df_fst_final.iloc[:,list(range(0, 16)) + [17]] >= tsd_1, 1, 0)
df_fst_binary_90 = pd.DataFrame(df_fst_binary_90, columns=df_fst_final.columns[list(range(0, 16)) + [17]])
df_fst_binary_90['Ensemble'] = df_fst_binary_90.iloc[:,:16].mean(axis = 1)
df_fst_binary_90['Date'] = df_fst_final['Date']

df_fst_binary_90.to_csv("E:\\PhD Datasets\\S2S Streamflow Forecast\\S2S PVM\\S2S_90_SF_binary_PVM_Godavari.csv") ## Change the directory accordingly


# True binary labels
y_true = df_fst_binary_90['SF_obs']
# Forecasted probabilities
y_scores = df_fst_binary_90['Ensemble']
y_clim = df_fst_binary_90['Q_clim']
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fpr_c, tpr_c, thresholds = roc_curve(y_true, y_clim)
# Compute ROC AUC
roc_auc = auc(fpr, tpr)
roc_auc_c = auc(fpr_c, tpr_c)

plt.figure(figsize = (8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Ensemble (AUC = %0.2f)' % roc_auc)
plt.plot(fpr_c, tpr_c, color='red', lw=2, label='Climatology (AUC = %0.2f)' % roc_auc_c)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('ROC Curve for Streamflow Forecast' , fontsize = 22)
plt.legend(loc='lower right', fontsize = 16)
plt.tick_params(axis='both', labelsize=14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Streamflow skill\\S2S_90_SF_Clim_PVM_roc.png", dpi = 600)  ## Change the directory accordingly
plt.show()

# Convert probabilities to binary labels using a threshold (e.g., 0.5)
threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)
y_pred_clim = (y_clim >= threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_c = confusion_matrix(y_true, y_pred_clim)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Streamflow skill\\S2S_90_SF_PVM_cm.png", dpi = 600) ## Change the directory accordingly
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_c,
                              display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Streamflow skill\\S2S_90_SF_Clim_PVM_cm.png", dpi = 600) ## Change the directory accordingly
plt.show()

### ROC for different lead times ###########################################################
roc = np.empty((32,1))
roc_c = np.empty((32,1))
cm_all = np.empty((2,2,32))
for i in range(0,32):
    idx = np.arange(0+i, 2080+i, 32)
    df_SF_fst_lead = df_fst_binary_90.iloc[idx] 
      
    # True binary labels
    y_true = df_SF_fst_lead['SF_obs']
    # Forecasted probabilities
    y_scores = df_SF_fst_lead['Ensemble']
    y_clim = df_SF_fst_lead['Q_clim']
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fpr_c, tpr_c, thresholds = roc_curve(y_true, y_clim)
    # Compute ROC AUC
    roc_auc = auc(fpr, tpr)
    roc_auc_c = auc(fpr_c, tpr_c)
    roc[i] = roc_auc
    roc_c[i] = roc_auc_c

from numpy.polynomial.polynomial import Polynomial
x = np.arange(1,33)
roc = np.asarray(roc).flatten()
poly_coeffs = Polynomial.fit(x, roc, deg=1)
trend_line = poly_coeffs(x)
true_coeffs = poly_coeffs.convert().coef
slope = true_coeffs[1]
intercept = true_coeffs[0]

# Format the equation of the trendline
equation = f"y = {slope:.2f}x + {intercept:.2f}"

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(x, roc,  marker='o', linestyle='-', label = 'AUC')
plt.plot(x, trend_line, color='red', linestyle='--', label='Trend line')
plt.text(17, 0.935 * max(roc), equation, fontsize=14, color='red')
plt.xlabel('Lead Time', fontsize = 16)
plt.ylabel('Area Under Curve(AUC)', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Streamflow skill\\S2S_90_SF_PVM_roclead.png", dpi = 600)  ## Change the directory accordingly

roc_c = np.asarray(roc_c).flatten()
roc_ratio = roc/roc_c
poly_coeffs = Polynomial.fit(x, roc_ratio, deg=1)
trend_line = poly_coeffs(x)
true_coeffs = poly_coeffs.convert().coef
slope = true_coeffs[1]
intercept = true_coeffs[0]

# Format the equation of the trendline
equation = f"y = {slope:.2f}x + {intercept:.2f}"

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(12, 6))
plt.plot(x, roc_ratio,  marker='o', linestyle='-', label = 'AUC')
plt.plot(x, trend_line, color='red', linestyle='--', label='Trend line')
plt.text(17, 0.8 * max(roc_ratio), equation, fontsize=14, color='red')
plt.xlabel('Lead Time', fontsize = 16)
plt.ylabel('AUC_S2S/AUC_Clim', fontsize = 16)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.legend(fontsize = 14)
plt.savefig("E:\\PhD Figures\\Godavari\\S2S Streamflow skill\\S2S_90_SF_PVM_roclead_clim.png", dpi = 600)  ## Change the directory accordingly

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################