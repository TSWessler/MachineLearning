%{
name:
pumpFeatures_MAIN

version:
wessler
2025 January 2
1st version


description:
*runs several classification models to predict fault codes of pump data
*reads "pumpFeatures.txt"
*pumpFeatures data adapted from MATLAB ML course


used by:
NOTHING
this is the code to run


uses:
*func_ComputeModelPerformance.m
    -> *func_UnitConvert_SecToDayHrMinSec.m
*func_MakePlots_ConfusionChart.m
*func_get_mdl_fitknn.m





NOTES:

The target term name is: faultCode

Basic Summary:

k-Nearest Neighbors:
Training: 00:00
Accuracy: 0.940

(Pruned) Classification Tree:
Training: 00:00
Accuracy: 0.913

Naive Bayes:
Training: 00:00
Accuracy: 0.873

Discriminant Analysis:
Training: 00:00
Accuracy: 0.883

Support Vector Machine:
Training: 00:00
Accuracy: 0.933

Neural Network:
Training: 00:00
Accuracy: 0.877


The best overall appears to be kNN. It is good on these data even with the
default model, and optimizing it increases the accuracy.

SVM does well, but it must be optimized, and the optimization takes several
minutes because this is a multiclass classification set.

Naive Bayes does the worst for this set. Optimizing improves it slightly,
but even after optimizing it doesn't perform as well as the default models
for other techniques.

todo:

%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

%%
clear
clc
rng('shuffle')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% training models
%--------------------------------------------------------------------------

%__________________________________________________________________________
%IDs:
%1: k-Nearest Neighbors Classification Model--Default
%2: k-Nearest Neighbors Classification Model--Optimized
%3: Classification Tree Model--Default
%4: Classification Tree Model--Optimized
%5: Classification Tree Model--Pruned
%6: Naive Bayes Classification Model--Default
%7: Naive Bayes Classification Model--Optimized
%8: Discriminant Analysis Classification Model--Default
%9: Discriminant Analysis Classification Model--Optimized
%10: Support Vector Machine Classification Model--Default
%11: Support Vector Machine Classification Model--Optimized
%12: Neural Network Classification Model--Default
%13: Neural Network Machine Classification Model--1-Layer Optimized
%14: Neural Network Machine Classification Model--2-Layer Optimized
%101: Run Optimize Hyperparameters--k-Nearest Neighbors
%102: Run Optimize Hyperparameters--Classification Tree
%103: Run Optimize Hyperparameters--Naive Bayes
%104: Run Optimize Hyperparameters--Discriminant Analysis
%105: Run Optimize Hyperparameters--Support Vector Machine
%106: Run Optimize Hyperparameters--Neural Network

List_ModelIDsToRun=[2,5,7,9,11,13,14];

%__________________________________________________________________________


%--------------------------------------------------------------------------
%data for models
%--------------------------------------------------------------------------

PathToData=pwd;
NameOfDataFile='pumpFeatures.txt';

HoldoutFrac = 3/10;


%--------------------------------------------------------------------------
%terms for plots
%--------------------------------------------------------------------------

CustomTerms.Axes.FontSize=16;


%--------------------------------------------------------------------------
%for displaying error
%--------------------------------------------------------------------------

formatSpec_ModelTraining='Model %i: %s';
formatSpec_ModelSummary='Summary of Model (%s):';
formatSpec_TimeToTrain='\n\nTime to train (hh:mm:ss): %s:%s:%s\n\n';
formatSpec_Accuracy='\nAccuracy (raw rate): %f\nAccuracy (using Prior probabilities): %f\n';
formatSpec_Error='\nPrecision Rate (macro-ave): %f\nRecall Rate (macro-ave): %f\nf1 Score (macro-ave): %f';
string_FormatTraining='\n=====================================================================================================\n';
string_FormatSummary='\n-----------------------------------------------------------------------------------------------------\n';
string_SayTraining='\ntraining...\n';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read and prepare data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rawData=readtable([PathToData,'/',NameOfDataFile]);
cv = cvpartition(size(rawData,1), "HoldOut", HoldoutFrac);
XTrain = rawData(cv.training,1:end-1);
XTest = rawData(cv.test,1:end-1);
yTrain = rawData(cv.training,end);
yTest = rawData(cv.test,end);
yTest_array = table2array(yTest);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CurrentWorkingDirectoryPath=pwd;
AllDirectories=genpath(CurrentWorkingDirectoryPath);
addpath(AllDirectories)

PerformanceFuncModelInfo.formatSpec_TimeToTrain=formatSpec_TimeToTrain;
PerformanceFuncModelInfo.formatSpec_Accuracy=formatSpec_Accuracy;
PerformanceFuncModelInfo.formatSpec_Error=formatSpec_Error;
PerformanceFuncModelInfo.Type='';
PerformanceFuncModelInfo.mdl=[];


%%
%##########################################################################
% classification models
%##########################################################################


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% k Nearest Neighbors Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 1: k-Nearest Neighbors Classification Model--Default
%==========================================================================

ModelID=1;
ModelDesc='k-Nearest Neighbors Classification Model--Default';
ModelOpts=[];

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitknn(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.916667
% Accuracy (using Prior probabilities): 0.892161
% Precision Rate (macro-ave): 0.888662
% Recall Rate (macro-ave): 0.879603
% f1 Score (macro-ave): 0.880814


%%
%==========================================================================
%MODEL 2: k-Nearest Neighbors Classification Model--Optimized
%==========================================================================

ModelID=2;
ModelDesc='k-Nearest Neighbors Classification Model--Optimized';
ModelOpts={'Standardize','off','Distance','minkowski','DistanceWeight','equal','Exponent',0.58111,'NumNeighbors',1};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitknn(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.940000
% Accuracy (using Prior probabilities): 0.920085
% Precision Rate (macro-ave): 0.922653
% Recall Rate (macro-ave): 0.910150
% f1 Score (macro-ave): 0.914356


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classification Trees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 3: Classification Tree--Default
%==========================================================================

ModelID=3;
ModelDesc='Classification Tree--Default';
ModelOpts=[];

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fittree(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.900000
% Accuracy (using Prior probabilities): 0.882755
% Precision Rate (macro-ave): 0.874516
% Recall Rate (macro-ave): 0.871928
% f1 Score (macro-ave): 0.871485


%%
%==========================================================================
%MODEL 4: Classification Tree--Optimized
%==========================================================================

ModelID=4;
ModelDesc='Classification Tree--Optimized';
ModelOpts={'MinLeafSize',1,'MaxNumSplits',100,'SplitCriterion','gdi','NumVariablesToSample',84};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fittree(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.900000
% Accuracy (using Prior probabilities): 0.882755
% Precision Rate (macro-ave): 0.874516
% Recall Rate (macro-ave): 0.871928
% f1 Score (macro-ave): 0.871485


%%
%==========================================================================
%MODEL 5: Classification Tree--Pruned
%==========================================================================

ModelID=5;
ModelDesc='Classification Tree--Pruned';
ModelOpts={};
PrunedLevel=2;

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fittree(XTrain,yTrain,ModelOpts);

    mdl = prune(mdl,"level",PrunedLevel);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.913333
% Accuracy (using Prior probabilities): 0.893863
% Precision Rate (macro-ave): 0.884171
% Recall Rate (macro-ave): 0.882153
% f1 Score (macro-ave): 0.882478


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Naive Bayes Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 6: Naive Bayes Classification Model--Default
%==========================================================================

ModelID=6;
ModelDesc='Naive Bayes Classification Model--Default';
ModelOpts={};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitnb(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.846667
% Accuracy (using Prior probabilities): 0.833460
% Precision Rate (macro-ave): 0.856803
% Recall Rate (macro-ave): 0.811958
% f1 Score (macro-ave): 0.810317


%%
%==========================================================================
%MODEL 7: Naive Bayes Classification Model--Optimized
%==========================================================================

ModelID=7;
ModelDesc='Naive Bayes Classification Model--Optimized';
ModelOpts={"Standardize",false,"DistributionNames","Kernel","Kernel","normal","Width",60};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitnb(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.873333
% Accuracy (using Prior probabilities): 0.841287
% Precision Rate (macro-ave): 0.833392
% Recall Rate (macro-ave): 0.821327
% f1 Score (macro-ave): 0.824761


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discriminant Analysis Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 8: Discriminant Analysis Classification Model--Default
%==========================================================================

ModelID=8;
ModelDesc='Discriminant Analysis Classification Model--Default';
ModelOpts={};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitdiscr(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.876667
% Accuracy (using Prior probabilities): 0.864900
% Precision Rate (macro-ave): 0.878817
% Recall Rate (macro-ave): 0.848407
% f1 Score (macro-ave): 0.853804


%%
%==========================================================================
%MODEL 9: Discriminant Analysis Classification Model--Optimized
%==========================================================================

ModelID=9;
ModelDesc='Discriminant Analysis Classification Model--Optimized';
ModelOpts={"DiscrimType","linear","Delta",7.9759e-06,"Gamma",3.3532e-05};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitdiscr(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.883333
% Accuracy (using Prior probabilities): 0.872727
% Precision Rate (macro-ave): 0.887104
% Recall Rate (macro-ave): 0.857659
% f1 Score (macro-ave): 0.862551


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Support Vector Machine Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 10: Support Vector Machine Classification Model--Default
%==========================================================================

ModelID=10;
ModelDesc='Support Vector Machine Classification Model--Default';
ModelOpts={};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitsvm(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:20
% Accuracy (raw rate): 0.673333
% Accuracy (using Prior probabilities): 0.649985
% Precision Rate (macro-ave): 0.687929
% Recall Rate (macro-ave): 0.622941
% f1 Score (macro-ave): 0.622599


%%
%==========================================================================
%MODEL 11: Support Vector Machine Classification Model--Optimized
%==========================================================================

ModelID=11;
ModelDesc='Support Vector Machine Classification Model--Optimized';
ModelOpts={"Standardize",false,"BoxConstraint",971.26,"KernelScale",836.72,"KernelFunction","gaussian"};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitsvm(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.890000
% Accuracy (using Prior probabilities): 0.873601
% Precision Rate (macro-ave): 0.862179
% Recall Rate (macro-ave): 0.860379
% f1 Score (macro-ave): 0.860096


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Network Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%==========================================================================
%MODEL 12: Neural Network Classification Model--Default
%==========================================================================

ModelID=12;
ModelDesc='Neural Network Classification Model--Default';
ModelOpts={};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitnet(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.096667
% Accuracy (using Prior probabilities): 0.108571
% Precision Rate (macro-ave): NaN
% Recall Rate (macro-ave): 0.125000
% f1 Score (macro-ave): NaN


%%
%==========================================================================
%MODEL 13: Neural Network Classification Model--1-Layer Optimized
%==========================================================================

ModelID=13;
ModelDesc='Neural Network Classification Model--1-Layer Optimized';
ModelOpts={"Standardize",true,"Activations","sigmoid","Lambda",7.7737e-06,...
    "LayerWeightsInitializer","he","LayerBiasesInitializer","ones","LayerSizes",[4]};


if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitnet(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:00
% Accuracy (raw rate): 0.876667
% Accuracy (using Prior probabilities): 0.866941
% Precision Rate (macro-ave): 0.855702
% Recall Rate (macro-ave): 0.857611
% f1 Score (macro-ave): 0.855771


%%
%==========================================================================
%MODEL 14: Neural Network Classification Model--2-Layer Optimized
%==========================================================================

ModelID=14;
ModelDesc='Neural Network Classification Model--2-Layer Optimized';
ModelOpts={"Standardize",true,"Activations","relu","Lambda",1.4543e-06,...
    "LayerWeightsInitializer","he","LayerBiasesInitializer","zeros","LayerSizes",[140,151]};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);

    [mdl,time_FromStartTrain]=func_get_mdl_fitnet(XTrain,yTrain,ModelOpts);


    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end

% Time to train (hh:mm:ss): 00:00:05
% Accuracy (raw rate): 0.886667
% Accuracy (using Prior probabilities): 0.879444
% Precision Rate (macro-ave): 0.868378
% Recall Rate (macro-ave): 0.869516
% f1 Score (macro-ave): 0.866660



%%
%##########################################################################
% Optimization of Models
%##########################################################################


%%
%==========================================================================
%MODEL 101: Run Optimization--k-Nearest Neighbors
%==========================================================================

ModelID=101;
ModelDesc='Run Optimization--k-Nearest Neighbors';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fitknn(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end


%%
%==========================================================================
%MODEL 102: Run Optimization--Classification Tree
%==========================================================================

ModelID=102;
ModelDesc='Run Optimization--Classification Tree';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fittree(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end


%%
%==========================================================================
%MODEL 103: Run Optimization--Naive Bayes
%==========================================================================

ModelID=103;
ModelDesc='Run Optimization--Naive Bayes';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fitnb(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end


%%
%==========================================================================
%MODEL 104: Run Optimize Hyperparameters--Discriminant Analysis
%==========================================================================

ModelID=104;
ModelDesc='Run Optimize Hyperparameters--Discriminant Analysis';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fitdiscr(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end


%%
%==========================================================================
%MODEL 105: Run Optimize Hyperparameters--Support Vector Machine
%==========================================================================

ModelID=105;
ModelDesc='Run Optimize Hyperparameters--Support Vector Machine';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fitsvm(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end


%%
%==========================================================================
%MODEL 106: Run Optimize Hyperparameters--Support Vector Machine
%==========================================================================

ModelID=106;
ModelDesc='Run Optimize Hyperparameters--Neural Network';
ModelOpts={'OptimizeHyperparameters','all'};

if ismember(ModelID,List_ModelIDsToRun)

    %--------------------------------------------------------------------------
    %train, predict
    %--------------------------------------------------------------------------

    fprintf(string_FormatTraining); fprintf('%s',sprintf(formatSpec_ModelTraining,ModelID,ModelDesc));
    fprintf(string_FormatTraining); fprintf(string_SayTraining);


    [mdl,time_FromStartTrain]=func_get_mdl_fitnet(XTrain,yTrain,ModelOpts);

    %--------------------------------------------------------------------------
    %output performance and make plots of Pred and Actual
    %--------------------------------------------------------------------------

    fprintf(string_FormatSummary)
    fprintf('%s',sprintf(formatSpec_ModelSummary,ModelDesc))
    fprintf(string_FormatSummary)

    yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo);

    CustomTerms.Figure.figID=ModelID;
    CustomTerms.Axes.Title=ModelDesc;
    func_MakePlots_ConfusionChart(yTest_array,yPred,CustomTerms);

end












