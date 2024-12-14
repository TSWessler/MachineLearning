%{
name:
func_get_mdl_rtree

version:
wessler
2024 November 13
1st version


description:
*runs "fitrtree" to make mdl, a regression decision tree
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

function [mdl,time_FromStartTrain] = func_get_mdl_rtree(DataTrain,yName,PruneLevel)

time_StartTrain=tic;
mdl = fitrtree(DataTrain,yName);

if ~ isnan(PruneLevel)
    mdl = prune(mdl,"Level",PruneLevel);
end


time_FromStartTrain=toc(time_StartTrain);






