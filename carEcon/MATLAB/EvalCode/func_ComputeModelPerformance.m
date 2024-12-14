%{
name:
func_ComputeModelPerformance

version:
wessler
2024 November 12
update to: 2024 October 28
changes:
    *combined this and func_OutputModelPerformance
    *made into a proper function


description:



used by:
carEcon_MAIN


uses:



NOTES:



%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

function ReturnVals=func_ComputeModelPerformance(yTest,yPred,time_FromStartTrain,PerformanceFuncModelInfo,ModelID)

%--------------------------------------------------------------------------
%*get and display "Function" for ridge and lm models
%--------------------------------------------------------------------------

if strcmp(PerformanceFuncModelInfo.Type,'Ridge Regression') | ...
        strcmp(PerformanceFuncModelInfo.Type,'Lasso Regression') | ...
        strcmp(PerformanceFuncModelInfo.Type,'Elastic Net Regression')
    MSE_temp=mean((yPred-yTest).^2,'omitnan');
    [~,idx]=min(MSE_temp);
    figure(1000+ModelID); clf;
    plot(PerformanceFuncModelInfo.lambda_vals,MSE_temp)
    xlabel("\lambda")
    ylabel("MSE")
    title([PerformanceFuncModelInfo.Type," model"])
    drawnow

    yPred=yPred(:,idx);

    if strcmp(PerformanceFuncModelInfo.Type,'Ridge Regression')
        intercept=PerformanceFuncModelInfo.coeffs(1,idx);
        coeffs=PerformanceFuncModelInfo.coeffs(2:end,idx);
    else
        intercept=PerformanceFuncModelInfo.lasso_info.Intercept(idx);
        coeffs=PerformanceFuncModelInfo.coeffs(:,idx);
    end

    Formula=[PerformanceFuncModelInfo.yName{1},' ~ ' num2str(intercept)];
    num_preds=length(PerformanceFuncModelInfo.XNames);
    ID_preds=0;
    while ID_preds<num_preds
        ID_preds=ID_preds+1;
        if coeffs(ID_preds)~=0
            Formula=[Formula,' + ',PerformanceFuncModelInfo.XNames{ID_preds}];
        end
    end
    fprintf('\n\n'); disp(Formula)
elseif strcmp(PerformanceFuncModelInfo.Type,'Linear Regression')
    Formula=PerformanceFuncModelInfo.mdl.Formula;
    fprintf('\n\n'); disp(Formula)
end

%--------------------------------------------------------------------------
%*display time to train
%--------------------------------------------------------------------------

TimeToTrain=func_UnitConvert_SecToDayHrMinSec(time_FromStartTrain);
disp_TimeToTrain=sprintf(PerformanceFuncModelInfo.formatSpec_TimeToTrain,TimeToTrain.Hr,TimeToTrain.Min,TimeToTrain.Sec);
fprintf('%s',disp_TimeToTrain)


%--------------------------------------------------------------------------
%*get and display errors
%--------------------------------------------------------------------------


err=yTest-yPred;
MaxErr=max(abs(err));
MSE=mean(err.^2,'omitnan');
MAPE=mean(abs(err),'omitnan');

disp_Error=sprintf(PerformanceFuncModelInfo.formatSpec_Error01,MaxErr,MSE,MAPE);
fprintf('%s',disp_Error)


if strcmp(PerformanceFuncModelInfo.Type,'Linear Regression')
    RSquared_Ord=PerformanceFuncModelInfo.mdl.Rsquared.Ordinary;
    RSquared_Adj=PerformanceFuncModelInfo.mdl.Rsquared.Adjusted;
    disp_Error=sprintf(PerformanceFuncModelInfo.formatSpec_Error02,RSquared_Ord,RSquared_Adj);
    fprintf('%s',disp_Error)
end

fprintf('\n\n\n\n\n')



%--------------------------------------------------------------------------
%*get return values
%--------------------------------------------------------------------------

if strcmp(PerformanceFuncModelInfo.Type,'Ridge Regression') | ...
        strcmp(PerformanceFuncModelInfo.Type,'Lasso Regression') | ...
        strcmp(PerformanceFuncModelInfo.Type,'Elastic Net Regression')
    ReturnVals=idx;
else
    ReturnVals=[];
end



