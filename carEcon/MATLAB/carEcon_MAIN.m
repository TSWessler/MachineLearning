%{
name:
carEcon_MAIN

version:
wessler
2024 October 17
1st version


description:
*runs several regression models to predict fuel economy of cars given
several features
*reads carData_cat.mat and carData_1hot.mat which were made by
"func_ReadData_carData.m" which reads data from the Table in carEcon.txt
*carEcon data adapted from MATLAB ML course


used by:
NOTHING
this is the code to run


uses:
*func_ReadData_carData.m (if not run yet)
*func_ComputeModelPerformance.m
    -> *func_UnitConvert_SecToDayHrMinSec.m
*func_MakePlots_PredAndActual.m
*func_get_mdl_fitlm.m
*func_get_mdl_stepwiselm.m
*func_get_mdl_ridge.m
*func_get_mdl_lasso.m
*func_get_mdl_rtree.m
*func_get_mdl_rsvm.m
*func_get_mdl_rnet.m
*func_get_mdl_rgp.m





NOTES:

The target term name is: FuelEcon

For algorithms that take categorical data (linear regression, stepwise
linear regression) will use XTrain_cat, yTrain_cat, XTest_cat, yTest_cat.

For algorithms that need all numerical data, will use XTrain_1hot,
yTrain_1hot, XTest,_1hot yTest_1hot and will have the category names
XNames and yName.


Basic Summary:

Linear Regression:
Training: (00:00,02:46)
MSE: (1.4,5.3)--best is Model 4
Best Overall: Model 3

Regression Tree:
Training: <1sec
MSE: 1--best is Model 10
Best Overall: Model 10

Regression SVM:
Training: <1sec
MSE: (1.1,2.1)--best is Model 12
Best Overall: Model 12

Regression Neural Network:
Training: (00:00,00:04)
MSE: (.8,2.6)--best is Model 17
Best Overall: Model 17 (or 16)

Gaussian Process Regression:
Training: (00:00,00:32)
MSE: (.7,1.9)--best is Model 23
Best Overall: Models 20 & 22 OR 21 & 23 (speed vs accuracy)


Regression Trees probably the best since so fast and so easy to set up, with
all models doing very well.

Gaussian Process Regression models take a bit of investigation to find the
right parameters, but they are fast and the most accurate.
NOTE: the "pureQuadratic" basis functions require the 1-hot data since
can't use categorical data while the other basis functions can use
categorical data; this causes the MSE to be higher for the "pureQuadratic"
models.


todo:
*put in mse for training to see how well does on training set?
*make marker sizes larger/change colors to match python better?

%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}
%%
clear
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% training models
%--------------------------------------------------------------------------

%__________________________________________________________________________
%IDs:
%1: Linear Regression Model--Default
%2: Linear Regression Model--Quadratic (Outliers Removed)
%3: Linear Regression Model--Stepwise "Linear" using "bic"
%4: Linear Regression Model--Stepwise Quadratic using "bic"
%5: Linear Regression Model--Ridge Coefficients
%6: Linear Regression Model--Lasso Coefficients
%7: Linear Regression Model--Elastic Net Coefficients (\alpha=0.9)
%8: Linear Regression Model--Elastic Net Coefficients (\alpha=0.9)
%9: Regression Decision Tree--Default
%10: Regression Decision Tree--Pruned (15 Layers)
%11: Support Vector Machine Regression Model--Defualt
%12: Support Vector Machine Regression Model--Optimized Gaussian
%13: Support Vector Machine Regression Model--Optimized 2nd-Order Polynomial
%14: Neural Network Regression Model--Default
%15: Neural Network Regression Model--Optimized 1-layer "relu"
%16: Neural Network Regression Model--Optimized 3-layer "relu"
%17: Neural Network Regression Model--Optimized 3-layer "sigmoid"
%18: Neural Network Regression Model--Optimized 3-layer "tanh"
%19: Gaussian Process Regression Model--Default
%20: Gaussian Process Regression Model--"none" using "rationalquadratic"
%21: Gaussian Process Regression Model--"none" using "ardrationalquadratic"
%22: Gaussian Process Regression Model--"constant" using "rationalquadratic"
%23: Gaussian Process Regression Model--"constant" using "ardrationalquadratic"
%24: Gaussian Process Regression Model--"linear" using "matern52"
%25: Gaussian Process Regression Model--"pureQuadratic" using "squaredexponential"

%101: Run Optimze Hyperparameters--Regression Support Vector Machine
%102: Run Optimze Hyperparameters--Regression Neural Network

List_ModelIDsToRun=[1,3,5,8,9,10,12,13,15,16,17,18,20,21,22];
List_ModelIDsToRun=[14,15,16,17,18];
List_ModelIDsToRun=[19,20,21,22,23,24,25];
%__________________________________________________________________________


%--------------------------------------------------------------------------
%data for models
%--------------------------------------------------------------------------

PathToData=pwd;

NameOfData_cat='carData_cat.mat';
NameOfData_1hot='carData_1hot.mat';


%--------------------------------------------------------------------------
%terms for plots
%--------------------------------------------------------------------------

CustomTerms.Plot.Marker='.';
CustomTerms.Plot.SizeData=100;
CustomTerms.Plot.LineWidth=2;

CustomTerms.Axes.FontSize=16;

CustomTerms.Legend.String={'Actual','Prediction'};


%--------------------------------------------------------------------------
%for displaying error
%--------------------------------------------------------------------------

formatSpec_ModelTraining='Model: %s';
formatSpec_ModelSummary='Summary of Model (%s):';
formatSpec_TimeToTrain='\n\nTime to train (hh:mm:ss): %s:%s:%s\n\n';
formatSpec_Error01='\nMax Error: %f\nMSE: %f\nMAPE: %f\n';
formatSpec_Error02='R^2 (ord): %f\nR^2 (adj): %f\n';
string_FormatTraining='\n=====================================================================================================\n';
string_FormatSummary='\n-----------------------------------------------------------------------------------------------------\n';
string_SayTraining='\ntraining...\n';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare data (if necessary) then read data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfile(NameOfData_cat)
     func_ReadData_carData
end

load([PathToData,'/',NameOfData_cat])
load([PathToData,'/',NameOfData_1hot])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(0)

CurrentWorkingDirectoryPath=pwd;
AllDirectories=genpath(CurrentWorkingDirectoryPath);
addpath(AllDirectories)

PerformanceFuncModelInfo.formatSpec_TimeToTrain=formatSpec_TimeToTrain;
PerformanceFuncModelInfo.formatSpec_Error01=formatSpec_Error01;
PerformanceFuncModelInfo.formatSpec_Error02=formatSpec_Error02;
PerformanceFuncModelInfo.Type='';
PerformanceFuncModelInfo.mdl=[];
PerformanceFuncModelInfo.lambda_vals=[];
PerformanceFuncModelInfo.coeffs=[];
PerformanceFuncModelInfo.lasso_info=[];
PerformanceFuncModelInfo.XNames=XNames;
PerformanceFuncModelInfo.yName=yName;


%##########################################################################
% regression models
%##########################################################################


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% linear regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%==========================================================================
%MODEL 1: "linear" linear regression
%==========================================================================

ModelID=1;
ModelDesc='Linear Regression Model--Default';
ModelOpts=[];

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitlm([XTrain_cat,yTrain_cat],ModelOpts);

    yPred = predict(mdl,XTest_cat);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Linear Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 6.434289
% MSE: 2.012743
% MAPE: 0.986492


%%
%==========================================================================
%MODEL 2: quadratic linear regression without outliers
%==========================================================================

ModelID=2;
ModelDesc='Linear Regression Model--Quadratic (Outliers Removed)';
ModelOpts={'quadratic','RobustOpts','on'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitlm([XTrain_cat,yTrain_cat],ModelOpts);

    yPred = predict(mdl,XTest_cat);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Linear Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 11.935891
% MSE: 5.228466
% MAPE: 1.200407


%%
%==========================================================================
%MODEL 3: stepwise "linear" linear regression--"bic"
%==========================================================================

ModelID=3;
ModelDesc='Linear Regression Model--Stepwise "Linear" using "bic"';
ModelOpts={'Upper','linear','Criterion','bic'};


if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);
    
    [mdl,time_FromStartTrain]=func_get_mdl_stepwiselm([XTrain_cat,yTrain_cat],ModelOpts);

    yPred = predict(mdl,XTest_cat);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Linear Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:01
% Max Error: 6.421315
% MSE: 1.953139
% MAPE: 0.973476


%%
%==========================================================================
%MODEL 4: stepwise quadratic linear regression--"bic"
%==========================================================================

ModelID=4;
ModelDesc='Linear Regression Model--Stepwise Quadratic using "bic"';
ModelOpts={'quadratic','Criterion','bic'};


if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_stepwiselm([XTrain_cat,yTrain_cat],ModelOpts);

    yPred = predict(mdl,XTest_cat);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Linear Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:02:22
% Max Error: 11.007362
% MSE: 1.410342
% MAPE: 0.654430


%%
%==========================================================================
%MODEL 5: ridge regression
%==========================================================================

ModelID=5;
ModelDesc='Linear Regression Model--Ridge Coefficients';
lambda_vals=0:200;
CODE_scale_YesOrNo=0; %scale coefficients back to original scaling (0: yes; 1: no)

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [coeffs,time_FromStartTrain] = func_get_mdl_ridge(XTrain_1hot,yTrain_1hot,lambda_vals,CODE_scale_YesOrNo);

    yPred=coeffs(1,:)+XTest_1hot*coeffs(2:end,:);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Ridge Regression';
    PerformanceFuncModelInfo.lambda_vals=lambda_vals; PerformanceFuncModelInfo.coeffs=coeffs;

    [IndexMin]=func_ComputeModelPerformance(yTest_1hot,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    yPred=yPred(:,IndexMin);

    CustomTerms.Axes.Title.String=ModelDesc;
    ModelID=func_MakePlots_PredAndActual(yTest_1hot,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 7.394316
% MSE: 2.756652
% MAPE: 0.996550


%%
%==========================================================================
%MODEL 6: lasso regression
%==========================================================================

ModelID=6;
ModelDesc='Linear Regression Model--Lasso Coefficients';
lambda_vals=(0:0.1:20)/length(yTrain_1hot);
alpha_val=1;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [coeffs,lasso_info,time_FromStartTrain] = func_get_mdl_lasso(XTrain_1hot,yTrain_1hot,lambda_vals,alpha_val);

    yPred=lasso_info.Intercept+XTest_1hot*coeffs;


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Lasso Regression';
    PerformanceFuncModelInfo.lambda_vals=lambda_vals; PerformanceFuncModelInfo.coeffs=coeffs; PerformanceFuncModelInfo.lasso_info=lasso_info;

    [IndexMin]=func_ComputeModelPerformance(yTest_1hot,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    yPred=yPred(:,IndexMin);

    CustomTerms.Axes.Title.String=ModelDesc;
    ModelID=func_MakePlots_PredAndActual(yTest_1hot,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 7.197783
% MSE: 2.406888
% MAPE: 0.963893


%%
%==========================================================================
%MODEL 7: elastic net regression (0.9)
%==========================================================================

ModelID=7;
ModelDesc='Linear Regression Model--Elastic Net Coefficients (\alpha=0.9)';
lambda_vals=(0:0.1:20)/length(yTrain_1hot);
alpha_val=0.9;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [coeffs,lasso_info,time_FromStartTrain] = func_get_mdl_lasso(XTrain_1hot,yTrain_1hot,lambda_vals,alpha_val);

    yPred=lasso_info.Intercept+XTest_1hot*coeffs;


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Elastic Net Regression';
    PerformanceFuncModelInfo.lambda_vals=lambda_vals; PerformanceFuncModelInfo.coeffs=coeffs; PerformanceFuncModelInfo.lasso_info=lasso_info;

    [IndexMin]=func_ComputeModelPerformance(yTest_1hot,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    yPred=yPred(:,IndexMin);

    CustomTerms.Axes.Title.String=ModelDesc;
    ModelID=func_MakePlots_PredAndActual(yTest_1hot,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 7.197766
% MSE: 2.406909
% MAPE: 0.963828


%%
%==========================================================================
%MODEL 8: elastic net regression (0.99)
%==========================================================================

ModelID=8;
ModelDesc='Linear Regression Model--Elastic Net Coefficients (\alpha=0.99)';
lambda_vals=(0:0.1:20)/length(yTrain_1hot);
alpha_val=0.99;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [coeffs,lasso_info,time_FromStartTrain] = func_get_mdl_lasso(XTrain_1hot,yTrain_1hot,lambda_vals,alpha_val);

    yPred=lasso_info.Intercept+XTest_1hot*coeffs;


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Elastic Net Regression';
    PerformanceFuncModelInfo.lambda_vals=lambda_vals; PerformanceFuncModelInfo.coeffs=coeffs; PerformanceFuncModelInfo.lasso_info=lasso_info;

    [IndexMin]=func_ComputeModelPerformance(yTest_1hot,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    yPred=yPred(:,IndexMin);

    CustomTerms.Axes.Title.String=ModelDesc;
    ModelID=func_MakePlots_PredAndActual(yTest_1hot,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 7.198521
% MSE: 2.406883
% MAPE: 0.963821





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regression tree
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%==========================================================================
%MODEL 9: default regression tree
%==========================================================================

ModelID=9;
ModelDesc='Regression Decision Tree--Default';
PruneLevel=nan;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rtree([XTrain_cat,yTrain_cat],yName{1},PruneLevel);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression Tree';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.414900
% MSE: 1.025658
% MAPE: 0.687541


%%
%==========================================================================
%MODEL 10: pruned regression tree
%==========================================================================

ModelID=10;
ModelDesc='Regression Decision Tree--Pruned (15 Layers)';
PruneLevel=15;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rtree([XTrain_cat,yTrain_cat],yName{1},PruneLevel);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression Tree';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.414900
% MSE: 1.003598
% MAPE: 0.689380





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regression support vector machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%==========================================================================
%MODEL 11: default svm
%==========================================================================

ModelID=11;
ModelDesc='SVM Regression Model--Default';
ModelOpts={"Standardize",true};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rsvm([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression SVM';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 6.846266
% MSE: 2.071577
% MAPE: 0.915592


%%
%==========================================================================
%MODEL 12: optimized gaussian svm
%==========================================================================

ModelID=12;
ModelDesc='SVM Regression Model--Optimized Gaussian';
ModelOpts={"Standardize",true,"BoxConstraint",998.14,"KernelScale",4.3109,"Epsilon",0.0034954,"KernelFunction","gaussian"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);
    
    [mdl,time_FromStartTrain] = func_get_mdl_rsvm([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression SVM';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.741800
% MSE: 1.106497
% MAPE: 0.661574


%%
%==========================================================================
%MODEL 13: optimized 2nd-order svm
%==========================================================================

ModelID=13;
ModelDesc='SVM Regression Model--Optimized 2nd-Order Polynomial';
ModelOpts={"Standardize",true,"BoxConstraint",0.033987,"Epsilon",0.011618,"KernelFunction","polynomial","PolynomialOrder",2};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rsvm([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression SVM';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.741800
% MSE: 1.207203
% MAPE: 0.717381





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% neural network regression model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%==========================================================================
%MODEL 14: default neural net
%==========================================================================

ModelID=14;
ModelDesc='Neural Network Regression Model--Default';
ModelOpts={"Standardize",true};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 12.009625
% MSE: 2.625530
% MAPE: 0.886112


%%
%==========================================================================
%MODEL 15: optimized 1-layer "relu" neural net
%==========================================================================

ModelID=15;
ModelDesc='Neural Network Regression Model--Optimized 1-layer "relu"';
ModelOpts={"Standardize",true,"Activations","relu","Lambda",0.0063649,"LayerWeightsInitializer","he","LayerBiasesInitializer","ones","LayerSizes",150};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:02
% Max Error: 5.787580
% MSE: 0.926187
% MAPE: 0.596199


%%
%==========================================================================
%MODEL 16: optimized 3-layer "relu" neural net
%==========================================================================

ModelID=16;
ModelDesc='Neural Network Regression Model--Optimized 3-layer "relu"';
ModelOpts={"Standardize",true,"Activations","relu","Lambda",0.019489,"LayerWeightsInitializer","he","LayerBiasesInitializer","zeros","LayerSizes",[34    29     2]};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:01
% Max Error: 4.896121
% MSE: 0.967674
% MAPE: 0.627470


%%
%==========================================================================
%MODEL 17: optimized 3-layer "sigmoid" neural net
%==========================================================================

ModelID=17;
ModelDesc='Neural Network Regression Model--Optimized 3-layer "sigmoid"';
ModelOpts={"Standardize",true,"Activations","sigmoid","Lambda",0.0050978,"LayerWeightsInitializer","glorot","LayerBiasesInitializer","zeros","LayerSizes",[147     18    299]};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:04
% Max Error: 6.233688
% MSE: 0.995324
% MAPE: 0.621137


%%
%==========================================================================
%MODEL 18: optimized 3-layer "tanh" neural net
%==========================================================================

ModelID=18;
ModelDesc='Neural Network Regression Model--Optimized 3-layer "tanh"';
ModelOpts={"Standardize",true,"Activations","tanh","Lambda",2.7315e-08,"LayerWeightsInitializer","glorot","LayerBiasesInitializer","zeros","LayerSizes",[49     22    242]};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:04
% Max Error: 4.741800
% MSE: 0.842269
% MAPE: 0.599043





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gaussian process regression model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%==========================================================================
%MODEL 19: default gaussian process regression model
%==========================================================================

ModelID=19;
ModelDesc='Gaussian Process Regression Model--Default';
ModelOpts=[];

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 8.113114
% MSE: 7.226597
% MAPE: 2.337765


%%
%==========================================================================
%MODEL 20: gaussian process regression--no basis function with "rationalquadratic" kernel
%==========================================================================

ModelID=20;
ModelDesc='Gaussian Process Regression Model--"none" using "rationalquadratic"';
ModelOpts={"Standardize",true,"BasisFunction","none","KernelFunction","rationalquadratic"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.741800
% MSE: 0.821109
% MAPE: 0.563000


%%
%==========================================================================
%MODEL 21: gaussian process regression--no basis function with "ardrationalquadratic" kernel
%==========================================================================

ModelID=21;
ModelDesc='Gaussian Process Regression Model--"none" using "ardrationalquadratic"';
ModelOpts={"Standardize",true,"BasisFunction","none","KernelFunction","ardrationalquadratic"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:32
% Max Error: 4.741800
% MSE: 0.736544
% MAPE: 0.565087


%%
%==========================================================================
%MODEL 22: gaussian process regression--constant basis function with "rationalquadratic" kernel
%==========================================================================

ModelID=22;
ModelDesc='Gaussian Process Regression Model--"constant" using "rationalquadratic"';
ModelOpts={"Standardize",true,"BasisFunction","constant","KernelFunction","rationalquadratic"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.741800
% MSE: 0.822382
% MAPE: 0.563503


%%
%==========================================================================
%MODEL 23: gaussian process regression--constant basis function with "ardrationalquadratic" kernel
%==========================================================================

ModelID=23;
ModelDesc='Gaussian Process Regression Model--"constant" using "ardrationalquadratic"';
ModelOpts={"Standardize",true,"BasisFunction","constant","KernelFunction","ardrationalquadratic"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:25
% Max Error: 4.741800
% MSE: 0.736255
% MAPE: 0.565311


%%
%==========================================================================
%MODEL 24: gaussian process regression--linear basis function with "matern52" kernel
%==========================================================================

ModelID=24;
ModelDesc='Gaussian Process Regression Model--"linear" using "matern52"';
ModelOpts={"Standardize",true,"BasisFunction","linear","KernelFunction","matern52"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 4.741800
% MSE: 0.832925
% MAPE: 0.556439


%%
%==========================================================================
%MODEL 25: gaussian process regression--"pureQuadratic" basis function with "squaredexponential" kernel
%==========================================================================

ModelID=25;
ModelDesc='Gaussian Process Regression Model--"pureQuadratic" using "squaredexponential"';
ModelOpts={"Standardize",true,"BasisFunction","pureQuadratic","KernelFunction","squaredexponential"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rgp(XTrain_1hot,yTrain_1hot,ModelOpts);
    
    yPred = predict(mdl,XTest_1hot);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_1hot,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end

% Time to train (hh:mm:ss): 00:00:00
% Max Error: 9.939315
% MSE: 1.913627
% MAPE: 0.693034









%%
%==========================================================================
%MODEL 101: optimize svn
%==========================================================================

ModelID=101;
ModelDesc='Run Optimze Hyperparameters--Regression SVM';
ModelOpts={"Standardize","on","OptimizeHyperparameters","all"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rsvm([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression SVM';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end


%%
%==========================================================================
%MODEL 102: optimize neural net
%==========================================================================

ModelID=102;
ModelDesc='Run Optimze Hyperparameters--Regression Neural Network';
ModelOpts={"Standardize",true,"OptimizeHyperparameters","all"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain] = func_get_mdl_rnet([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='Regression NN';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end


%%
%==========================================================================
%MODEL 103: optimize gaussian process
%==========================================================================

ModelID=103;
ModelDesc='Run Optimze Hyperparameters--Gaussian Process Regression';
ModelOpts={"OptimizeHyperparameters","all"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelDesc));
    fprintf(string_FormatTraining); fprintf('\n');

    [mdl,time_FromStartTrain] = func_get_mdl_rgp([XTrain_cat,yTrain_cat],yName{1},ModelOpts);

    yPred = predict(mdl,[XTest_cat,yTest_cat]);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    PerformanceFuncModelInfo.Type='GP Regression';
    PerformanceFuncModelInfo.mdl=mdl;
    [~]=func_ComputeModelPerformance(yTest_cat.FuelEcon,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID);

    CustomTerms.Axes.Title.String=ModelDesc;
    func_MakePlots_PredAndActual(yTest_cat.FuelEcon,yPred,CustomTerms,ModelID);


end







