%{
name:
func_get_mdl_fitlm

version:
wessler
2024 November 12
update to: 2024 October 28
changes:
    *made proper function


description:
*runs "fitlm" to get model "mdl" (and times how long it takes to train)
*inputs: DataTrain, ModelOpts
*outputs: mdl, time_FromStartTrain


used by:
carEcon_MAIN


uses:
NOTHING


NOTES:



%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

function [mdl,time_FromStartTrain] = func_get_mdl_fitlm(DataTrain,ModelOpts)

if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitlm(DataTrain);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitlm(DataTrain,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end












