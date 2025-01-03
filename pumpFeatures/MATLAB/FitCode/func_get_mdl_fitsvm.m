%{
name:
func_get_mdl_fitsvm

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

function [mdl,time_FromStartTrain] = func_get_mdl_fitsvm(XTrain,yTrain,ModelOpts)


if isempty(ModelOpts)
    template=templateSVM();
    time_StartTrain=tic;
    mdl = fitcecoc(XTrain,yTrain,"Learners",template);
    time_FromStartTrain=toc(time_StartTrain);
else
    if strcmp(ModelOpts{1},'OptimizeHyperparameters')
        [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
        time_StartTrain=tic;
        mdl = fitcecoc(XTrain,yTrain,ModelOpts{:});
        time_FromStartTrain=toc(time_StartTrain);
    else
        [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
        template=templateSVM(ModelOpts{:});
        time_StartTrain=tic;
        mdl = fitcecoc(XTrain,yTrain,"Learners",template);
        time_FromStartTrain=toc(time_StartTrain);
    end
end








