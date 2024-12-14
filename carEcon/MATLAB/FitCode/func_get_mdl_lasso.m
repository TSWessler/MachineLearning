%{
name:
func_get_mdl_lasso

version:
wessler
2024 November 14
1st version


description:
*runs "lasso" to get linear regression coefficents
*needs: yTrain,XTrain,lambda_vals,alpha_val
*gets: lasso_coeffs, lasso_info, time_FromStartTrain


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

function [lasso_coeffs,lasso_info,time_FromStartTrain] = func_get_mdl_lasso(XTrain,yTrain,lambda_vals,alpha_val)

time_StartTrain=tic;
[lasso_coeffs,lasso_info]  = lasso(XTrain,yTrain,"Lambda",lambda_vals,"Alpha",alpha_val);
time_FromStartTrain=toc(time_StartTrain);












