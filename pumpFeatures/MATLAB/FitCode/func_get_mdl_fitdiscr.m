%{
name:
func_get_mdl_fitdiscr

version:
wessler
2024 January 3
1st version


description:
*runs "fitcdiscr" to get model "mdl" (and times how long it takes to train)
*inputs: XTrain, yTrain, ModelOpts
*outputs: mdl, time_FromStartTrain


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

function [mdl,time_FromStartTrain] = func_get_mdl_fitdiscr(XTrain,yTrain,ModelOpts)


if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitcdiscr(XTrain,yTrain);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitcdiscr(XTrain,yTrain,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end








