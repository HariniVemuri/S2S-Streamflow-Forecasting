% Assuming you have your data loaded as rain, temp, and Q_obs
clc
clear all
% This is the source codes for model calibration 
data=xlsread("E:\PhD Datasets\PIML Results\Hydrological_simulations_Godavari_PVM.xlsx"); % loading the P, temp and Q_o % Change the directory accordigly


rain=data(:,1);
temp=data(:,2);
Q_obs=data(:,6);
Q_obs(isnan(Q_obs))=0;
% Define the number of folds
numFolds = 3;

% Create indices for 3-fold cross-validation
cv = cvpartition(height(rain), 'KFold', numFolds);

% Initialize an array to store the calibrated parameters
calibratedParams = cell(numFolds, 1);

% Loop over the folds
for fold = 1:numFolds
    % Get the training and validation indices for this fold
    trainIdx = training(cv, fold);
    valIdx = test(cv, fold);
    
    % Extract the training and validation data for this fold
    trainRain = rain(trainIdx);
    trainTemp = temp(trainIdx);
    trainQ_obs = Q_obs(trainIdx);
    valRain = rain(valIdx);
    valTemp = temp(valIdx);
    valQ_obs = Q_obs(valIdx);
    
end

    % Run GA algorithm for calibration using the training data
    LB=[0.01,0.5,0.5,1,1,0.01];UB=[1,100,10,1000,1000,1];
    nn = length(LB);
    %options = gaoptimset('generations', 200, 'PopulationSize', 100);
    pars = ga(@(pars)ObjFun(trainRain, trainTemp, pars, trainQ_obs), nn, [], [], [], [], LB, UB);
    
    % Store the calibrated parameters
    calibratedParams{fold} = pars;
    [Q_pred_train, uk_cal, xkq_cal, xks_cal]=ihacres(trainRain,trainTemp,pars);

    % Apply the calibrated parameters to predict streamflow for the validation data
    pars = calibratedParams{fold}
    [Q_pred_val, uk_val, xkq_val, xks_val] = ihacres(valRain, valTemp, pars);
    
    NSE_cal = 1-sum((trainQ_obs-Q_pred_train).^2)/sum((trainQ_obs-mean(trainQ_obs)).^2)
    corr_cal = corr(Q_pred_train,trainQ_obs)
    alpha = (std(trainQ_obs)/std(Q_pred_train))
    beta = (mean(trainQ_obs)/mean(Q_pred_train))
    KGE_cal = 1 - sqrt((corr_cal-1)^2 + (alpha - 1)^2 + (beta - 1)^2)


    NSE_val = 1-sum((valQ_obs-Q_pred_val).^2)/sum((valQ_obs-mean(valQ_obs)).^2)
    corr_val = corr(Q_pred_val,valQ_obs)
    alpha = (std(valQ_obs)/std(Q_pred_val))
    beta = (mean(valQ_obs)/mean(Q_pred_val))
    KGE_val = 1 - sqrt((corr_val-1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    % Evaluate the performance of the calibrated model for this fold
    % You can calculate performance metrics such as R-squared, RMSE, etc.
    % Store or display the performance metrics as needed
end

%pars = calibratedParams{3} %% Change here
pars = [ 0.6331,    0.6159,    0.5205,  176.6650 ,   9.2420,    0.2630]
[Q_pred_train, uk_cal, xkq_cal, xks_cal]=ihacres(trainRain,trainTemp,pars);
[Q_pred_val, uk_val, xkq_val, xks_val] = ihacres(valRain, valTemp, pars);
[Q_pred, uk_pred, xkq_pred, xks_pred] = ihacres(rain, temp, pars);

% Define start and end dates
start_date = datetime(1980, 1, 1);
end_date = datetime(2020, 12, 31);

% Generate the date range with daily frequency
date_range = start_date:end_date;
date_range = (date_range)';

plot(trainQ_obs);                                                                                                                                                                                                                                
hold                                                                                                                                                                                                                                                                                                                          
plot( Q_pred_train,'r-');       
xlabel('Time','FontSize', 14, 'FontName', 'Times New Roman')
ylabel('Streamflow(mm)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('IHACRES Simulated Streamflow (Training set)', 'FontSize', 16, 'FontName', 'Times New Roman');
legend('Observed','Predicted');
ax = gca;  % Get current axes handle
ax.FontSize = 12;             % Increase font size for tick labels
ax.FontName = 'Times New Roman';  % Set font to Times New Roman
print("E:\PhD Figures\Godavari\IHACRES Simulations\SF_train_Godavari_PVM", '-dpng', '-r600'); % Change the directory accordigly

plot(valQ_obs);                                                                                                                                                                                                                                
hold                                                                                                                                                                                                                                                                                                                          
plot(Q_pred_val,'r-');       
xlabel('Time', 'FontSize', 14, 'FontName', 'Times New Roman')
ylabel('Streamflow(mm)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('IHACRES Simulated Streamflow (Testing set)', 'FontSize', 16, 'FontName', 'Times New Roman');
legend('Observed','Predicted');
ax = gca;  % Get current axes handle
ax.FontSize = 12;             % Increase font size for tick labels
ax.FontName = 'Times New Roman';  % Set font to Times New Roman
print("E:\PhD Figures\Godavari\IHACRES Simulations\SF_test_Godavari_PVM", '-dpng', '-r600'); % Change the directory accordigly

plot(date_range, Q_obs);                                                                                                                                                                                                                                
hold                                                                                                                                                                                                                                                                                                                          
plot(date_range, Q_pred,'r-');       
xlabel('Time', 'FontSize', 14, 'FontName', 'Times New Roman')
ylabel('Streamflow (mm)','FontSize', 14, 'FontName', 'Times New Roman');
%title('IHACRES Simulated Streamflow', 'FontSize', 16, 'FontName', 'Times New Roman');
legend('Observed','Predicted');
ax = gca;  % Get current axes handle
ax.FontSize = 14;             % Increase font size for tick labels
ax.FontName = 'Times New Roman';  % Set font to Times New Roman
ax.FontWeight = 'bold';

% Set the figure size for saving
fig = gcf;  
fig.PaperUnits = 'inches';  
fig.PaperPosition = [0 0 12 6];  % Set width and height (7200/600 = 12 inches, 3600/600 = 6 inches)

print("E:\PhD Figures\Godavari\IHACRES Simulations\SF_Godavri_PVM", '-dpng', '-r600'); % Change the directory accordigly

NSE = 1-sum((Q_obs-Q_pred).^2)/sum((Q_obs-mean(Q_obs)).^2)
correl = corr(Q_pred,Q_obs)
alpha = (std(Q_obs)/std(Q_pred))
beta = (mean(Q_obs)/mean(Q_pred))
KGE = 1 - sqrt((correl-1)^2 + (alpha - 1)^2 + (beta - 1)^2)