%{
name:
func_ReadData_carData

version:
wessler
2024 October 29
1st version


description:
*reads in data in Table format
*changes categories to "one-hot"
*writes: carData_cat.mat and carData_1hot.mat


used by:
*carEcon_MAIN (if hasn't been run yet)


uses:
NOTHING
but: needs carData.txt


NOTES:

Response Term (Target/Expected Outcome):
FuelEcon (column 15)

categorical predictors (features):
Car_Truck
Transmission
Drive
AC
City_Highway


%##########################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


PathToData=pwd;
NameOfData='carData.txt';
Holdout=0.3;


NamesOfCategoricalTerms={'Car_Truck','Transmission','Drive','AC','City_Highway'};
ColForTargetTerm=15;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%read/prep table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

carData=readtable([PathToData,'/',NameOfData]);
TableColsNames=carData.Properties.VariableNames;


%==========================================================================
%make sure categorical terms are categorical
%==========================================================================

carData.Car_Truck=categorical(carData.Car_Truck);
carData.Transmission=categorical(carData.Transmission);
carData.Drive=categorical(carData.Drive);
carData.AC=categorical(carData.AC);
carData.City_Highway=categorical(carData.City_Highway);


%==========================================================================
%make sure data has feature at end
%==========================================================================

tempCol=carData(:,ColForTargetTerm);
carData(:,ColForTargetTerm)=[];
carData=[carData tempCol];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%data with categorical terms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==========================================================================
%partition categorical version of data
%==========================================================================

cv = cvpartition(size(carData,1),"HoldOut",Holdout);

XTrain_cat=carData(cv.training,1:end-1);
yTrain_cat=carData(cv.training,end);
XTest_cat=carData(cv.test,1:end-1);
yTest_cat=carData(cv.test,end);

%==========================================================================
%save categorical version of data
%==========================================================================

save('carData_cat.mat','XTrain_cat','yTrain_cat','XTest_cat','yTest_cat')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make new table with one-hot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OneHotTable=table();

ID_ColsOldTable=0;
MaxID_ColsOldTable=width(carData);

while ID_ColsOldTable<MaxID_ColsOldTable
    ID_ColsOldTable=ID_ColsOldTable+1;

    tempName=TableColsNames{ID_ColsOldTable};

    if ismember(tempName,NamesOfCategoricalTerms)
        OneHotTable=[OneHotTable,onehotencode(carData(:,ID_ColsOldTable))];
    else
        OneHotTable=[OneHotTable,carData(:,ID_ColsOldTable)];
    end

end

%==========================================================================
%partition one-hot version of data
%==========================================================================

cv = cvpartition(size(OneHotTable,1),"HoldOut",Holdout);

XTrain_1hot=OneHotTable(cv.training,1:end-1);
yTrain_1hot=OneHotTable(cv.training,end);
XTest_1hot=OneHotTable(cv.test,1:end-1);
yTest_1hot=OneHotTable(cv.test,end);


%==========================================================================
%save one-hot version of data
%==========================================================================

XNames_1hot=XTest_1hot.Properties.VariableNames;
yName_1hot=yTest_1hot.Properties.VariableNames;

XTrain_1hot=XTrain_1hot{:,:};
yTrain_1hot=yTrain_1hot{:,:};
XTest_1hot=XTest_1hot{:,:};
yTest_1hot=yTest_1hot{:,:};

save('carData_1hot.mat','XTrain_1hot','yTrain_1hot','XTest_1hot','yTest_1hot','XNames_1hot','yName_1hot')









