%{
name:
func_get_mdl_stepwiselm

version:
wessler
2024 November 12
update to: 2024 October 28
changes:
    *made proper function


description:
*runs "stepwiselm" to get model "mdl" (and times how long it takes to train)
*needs: DataTrain, ModelOpts
*gets: mdl, time_FromStartTrain


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

function [mdl,time_FromStartTrain] = func_get_mdl_stepwiselm(DataTrain,ModelOpts)

if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = stepwiselm(DataTrain);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = stepwiselm(DataTrain,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end












