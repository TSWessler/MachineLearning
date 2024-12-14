%{
name:
func_get_mdl_ridge

version:
wessler
2024 November 13
1st version


description:
*runs "ridge" to get linear regression coefficents
*needs: XTrain,yTrain,lambda_vals,CODE_scale_YesOrNo
*gets: ridge_coeffs, time_FromStartTrain


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

function [ridge_coeffs,time_FromStartTrain] = func_get_mdl_ridge(XTrain,yTrain,lambda_vals,CODE_scale_YesOrNo)

time_StartTrain=tic;
ridge_coeffs = ridge(yTrain,XTrain,lambda_vals,CODE_scale_YesOrNo);
time_FromStartTrain=toc(time_StartTrain);












