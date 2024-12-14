%{
Name:
    func_UnitConvert_SecToDayHrMinSec


Version:
    wessler
    2023 July 6
    1st version


Description:
    *Converts seconds to days, hours, minutes, and seconds


Inputs:
    *Seconds_Raw (1x1 double)


Outputs/does:
    *TimeStruc
        *TimeStruc.Day
        *TimeStruc.Hr
        *TimeStruc.Min
        *TimeStruc.Sec



Used by:
    *main_PDEversion.m
    *func_RunAllSamplesInSet.m



Uses:
    NOTHING


NOTES:


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
%--------------------------------------------------------------------------
%__________________________________________________________________________
%}

function TimeStruc=func_UnitConvert_SecToDayHrMinSec(Seconds_Raw)


if ~isnumeric(Seconds_Raw) || ~isscalar(Seconds_Raw)
    fprintf('\n\nERROR!'); asdf; %go ahead and crash it
end



TempVal_raw=Seconds_Raw*1/24*1/60*1/60;
TempDay=floor(TempVal_raw);
TempVal_raw=(TempVal_raw-TempDay)*24;
TempHr=floor(TempVal_raw);
TempVal_raw=(TempVal_raw-TempHr)*60;
TempMin=floor(TempVal_raw);
TempVal_raw=(TempVal_raw-TempMin)*60;
TempSec=floor(TempVal_raw);



if TempDay<10; TempStr=['0',num2str(TempDay)];
else; TempStr=num2str(TempDay);
end
TimeStruc.Day=TempStr;

if TempHr<10; TempStr=['0',num2str(TempHr)];
else; TempStr=num2str(TempHr);
end
TimeStruc.Hr=TempStr;

if TempMin<10; TempStr=['0',num2str(TempMin)];
else; TempStr=num2str(TempMin);
end
TimeStruc.Min=TempStr;

if TempSec<10; TempStr=['0',num2str(TempSec)];
else; TempStr=num2str(TempSec);
end
TimeStruc.Sec=TempStr;





end











