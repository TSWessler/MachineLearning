%{
name:
func_get_mdl_rgp

version:
wessler
2024 November 19
1st version


description:
*runs "fitrgp" to make mdl, a regression decision tree
*needs: DataTrain
*gets: mdl, time_FromStartTrain


used by:
carEcon_MAIN


uses:
NOTHING


NOTES:

if strcmp(class(Term2),'table')
    Term2 is yName{1}
    Term1 is [XTrain_cat,yTrain_cat]
elseif sstrcmp(class(Term2),'double')
    Term2 is yTrain_1hot
    Term1 is XTrain_1hot  
end

...both end up passed to fitrgp the same way, though

%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

function [mdl,time_FromStartTrain] = func_get_mdl_rgp(Term1,Term2,ModelOpts)


if isempty(ModelOpts)
    time_StartTrain=tic;
    mdl = fitrgp(Term1,Term2);
    time_FromStartTrain=toc(time_StartTrain);
else
    [ModelOpts{:}] = convertStringsToChars(ModelOpts{:});
    time_StartTrain=tic;
    mdl = fitrgp(Term1,Term2,ModelOpts{:});
    time_FromStartTrain=toc(time_StartTrain);
end












