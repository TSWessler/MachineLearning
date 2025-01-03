%{
name:
func_MakePlots_ConfusionChart

version:
wessler
2024 January 2
1st version


description:
makes plot for correct/incorrect classificatoins of fault codes


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


function func_MakePlots_ConfusionChart(yTest,yPred,CustomTerms)

Figure=figure(CustomTerms.Figure.figID);
clf;
Plot=confusionchart(yTest,yPred);
Axes=gca;

Axes.FontSize=CustomTerms.Axes.FontSize;
Axes.Title=CustomTerms.Axes.Title;

drawnow

end









