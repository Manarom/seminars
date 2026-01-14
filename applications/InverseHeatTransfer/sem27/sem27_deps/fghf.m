%%
%data  =readtable("E:\p" + ...
%    "rojects\matlab-seminar\applications\InverseHeatTransfer\sem27\sem27_deps\finite_difference_functions.jl")
%%
% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 3, "Encoding", "UTF16-LE");

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = "=";

% Specify column names and types
opts.VariableNames = ["contourpy", "Var2", "x1_3_1"];
opts.VariableTypes = ["string", "string", "datetime"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["contourpy", "Var2"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["contourpy", "Var2"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "x1_3_1", "InputFormat", "dd.MM.yy", "DatetimeFormat", "preserveinput");

% Import the data
requirements = readtable("E:\projects\matlab-seminar\requirements.txt", opts);

% Clear temporary variables
clear opts

% Display results
requirements