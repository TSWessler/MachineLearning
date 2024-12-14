%{
name:
func_get_mdl_rnet

version:
wessler
2024 November 18
1st version


description:
*runs "fitrnet" to make mdl, a regression neural network
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

function [mdl,time_FromStartTrain] = func_get_mdl_rnet(DataTrain,yName,ModelOpts)

if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitrnet(DataTrain,yName);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitrnet(DataTrain,yName,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end












