%{
name:
func_ComputeModelPerformance

version:
wessler
2025 January 3
1st version


description:
compute and display metrics for quantifying effectiveness of model


used by:
pumpFeatures_MAIN


uses:
NOTHING


NOTES:



%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

function yPred = func_ComputeModelPerformance(mdl,XTest,yTest,yTest_array,time_FromStartTrain,PerformanceFuncModelInfo)

%--------------------------------------------------------------------------
%*get prediction, compute errors
%--------------------------------------------------------------------------

%predicted classifications
yPred = predict(mdl,XTest);

%confusion matrix
ConfusionMat=confusionmat(yTest_array,yPred);
NumTerms=size(ConfusionMat,1);
TotalVals=sum(sum(ConfusionMat));

%true/false positives/negatives in confusion matrix
TruePos=zeros(NumTerms,1);
FalsePos=zeros(NumTerms,1);
FalseNeg=zeros(NumTerms,1);
termID=0;
while termID<NumTerms
    termID=termID+1;
    %true positives
    TruePos_Val=ConfusionMat(termID,termID);
    TruePos(termID)=TruePos_Val;
    %false positives
    FalsePos(termID)=sum(ConfusionMat(:,termID))-TruePos_Val;
    %false negatives
    FalseNeg(termID)=sum(ConfusionMat(termID,:))-TruePos_Val;
end
TruePos_tot=sum(TruePos);
% FalsePos_tot=sum(FalsePos);
% FalseNeg_tot=sum(FalseNeg);

%accuracy
MisclassRate_Unweighted=(TotalVals-TruePos_tot)/TotalVals;
MisclassRate_Weighted=loss(mdl,[XTest,yTest]);
Accuracy_Unweighted=1-MisclassRate_Unweighted;
Accuracy_Weighted=1-MisclassRate_Weighted;

% %micro-aves (testing purposes)
% PrecisionRate_MicroAve=TruePos_tot/(TruePos_tot+FalsePos_tot); % fraction of predicted positives that are true positives
% RecallRate_MicroAve=TruePos_tot/(TruePos_tot+FalseNeg_tot); % fraction of true positives that were predicted positives
% f1_score_MicroAve=2*(PrecisionRate_MicroAve*RecallRate_MicroAve)/(PrecisionRate_MicroAve+RecallRate_MicroAve);

%macro-aves
PrecisionRate_temp=TruePos./(TruePos+FalsePos);  %fraction of predicted positives that are true positives
RecallRate_temp=TruePos./(TruePos+FalseNeg);  %fraction of true positives that were predicted positives
PrecisionRate_MacroAve=mean(PrecisionRate_temp); 
RecallRate_MacroAve=mean(RecallRate_temp); 
f1_score_MacroAve=mean(2*(PrecisionRate_temp.*RecallRate_temp)./(PrecisionRate_temp+RecallRate_temp));

%--------------------------------------------------------------------------
%*display time to train
%--------------------------------------------------------------------------

TimeToTrain=func_UnitConvert_SecToDayHrMinSec(time_FromStartTrain);
disp_TimeToTrain=sprintf(PerformanceFuncModelInfo.formatSpec_TimeToTrain,TimeToTrain.Hr,TimeToTrain.Min,TimeToTrain.Sec);
fprintf('%s',disp_TimeToTrain)


%--------------------------------------------------------------------------
%*display errors
%--------------------------------------------------------------------------

disp_Error=sprintf(PerformanceFuncModelInfo.formatSpec_Accuracy,Accuracy_Unweighted,Accuracy_Weighted);
fprintf('%s',disp_Error)
disp_Error=sprintf(PerformanceFuncModelInfo.formatSpec_Error,PrecisionRate_MacroAve,RecallRate_MacroAve,f1_score_MacroAve);
fprintf('%s',disp_Error)

fprintf('\n\n\n\n\n')







