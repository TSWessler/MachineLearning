%{
name:
func_MakePlots_PredAndActual

version:
wessler
2024 November 12
update to: 2024 October 24
changes:
    *made into a proper function


description:
makes plots to summarize 


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


function func_MakePlots_PredAndActual(yTest,yPred,CustomTerms,ModelID)


%--------------------------------------------------------------------------
%sort terms for clarity
%--------------------------------------------------------------------------

[yTest,indices] = sort(yTest);
yPred=yPred(indices);

%--------------------------------------------------------------------------
%make plots
%--------------------------------------------------------------------------

%plot 1: Predicted and Actual v Observation ID
CustomTerms.Axes.XLabel.String='Sorted Sample ID';
CustomTerms.Axes.YLabel.String='Fuel Economy';
CustomTerms.Legend.String={'Actual','Prediction'};


CustomTerms.Figure.figID=ModelID;
PlotPredsAndActual(yTest,yPred,CustomTerms)

%plot 2: Predicted v Actual
CustomTerms.Axes.XLabel.String='Actual Fuel Economy';
CustomTerms.Axes.YLabel.String='Predicted Fuel Economy';

CustomTerms.Figure.figID=100+ModelID;
PlotPredsVsActual(yTest,yPred,CustomTerms)

end


%##########################################################################
%plot funcs
%##########################################################################



function PlotPredsAndActual(yData,yPred,CustomTerms)

xMin=0;
xMax=length(yData)+1;
xRange=[xMin,xMax];

Figure=figure(CustomTerms.Figure.figID);
clf
hold on
PlotData=scatter(1:length(yData),yData);
PlotPred=scatter(1:length(yData),yPred);
hold off
Axes=gca;
Legend=legend;

PlotData.Marker=CustomTerms.Plot.Marker;
PlotData.SizeData=CustomTerms.Plot.SizeData;
PlotPred.Marker=CustomTerms.Plot.Marker;
PlotPred.SizeData=CustomTerms.Plot.SizeData;

Axes.FontSize=CustomTerms.Axes.FontSize;
Axes.XLim=xRange;
Axes.Title.String=CustomTerms.Axes.Title.String;
Axes.XLabel.String=CustomTerms.Axes.XLabel.String;
Axes.YLabel.String=CustomTerms.Axes.YLabel.String;

Legend.String=CustomTerms.Legend.String;

drawnow

end



function PlotPredsVsActual(yData,yPred,CustomTerms)

xMin=min(min(yData),min(yPred));
xMax=max(max(yData),max(yPred));
xRange=[xMin,xMax];


Figure=figure(CustomTerms.Figure.figID);
clf
hold on
PlotDiag=plot(xRange,xRange,'k-');
PlotPred=scatter(yData,yPred);
hold off
Axes=gca;

PlotDiag.LineWidth=CustomTerms.Plot.LineWidth;

PlotPred.Marker=CustomTerms.Plot.Marker;
PlotPred.SizeData=CustomTerms.Plot.SizeData;

Axes.FontSize=CustomTerms.Axes.FontSize;
Axes.XLim=xRange;
Axes.YLim=xRange;
Axes.Title.String=CustomTerms.Axes.Title.String;
Axes.XLabel.String=CustomTerms.Axes.XLabel.String;
Axes.YLabel.String=CustomTerms.Axes.YLabel.String;

drawnow

end









