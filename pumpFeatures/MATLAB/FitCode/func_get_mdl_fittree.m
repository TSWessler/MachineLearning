%{
name:
func_get_mdl_fittree

version:
wessler
2024 January 2
1st version


description:
*runs "fitctree" to get model "mdl" (and times how long it takes to train)
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

function [mdl,time_FromStartTrain] = func_get_mdl_fittree(XTrain,yTrain,ModelOpts)


if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitctree(XTrain,yTrain);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitctree(XTrain,yTrain,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end








