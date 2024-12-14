%{
name:
func_get_mdl_rsvm

version:
wessler
2024 November 13
1st version


description:
*runs "fitrsvm" to make mdl, a regression decision tree
*needs: DataTrain
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

function [mdl,time_FromStartTrain] = func_get_mdl_rsvm(DataTrain,yName,ModelOpts)


if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitrsvm(DataTrain,yName);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitrsvm(DataTrain,yName,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end












