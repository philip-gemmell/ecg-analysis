%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Derives Simulated ECGs from Torso Phie data
% Martin Bishop
% KCL
% 4th October 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paths
path(path,'/home/mb12/Programmes/MatlabIGB/igb') ;
path(path,'/home/mb12/Code/Matlab/CEPT/') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads-in phie data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filepath = strcat('/data/Projects/Torso/simulations/sinus/sinusFeb/');
filepath = strcat('/data/ecg/phi1.5708-3.1416_rho0.1-0.9_z0.3-0.9_bcl600/ecg/');
data = readPhieFile(filepath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defines electrodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reads-in list of nodal electrodes
% electrodes = dlmread('/data/Projects/Torso/ECG/12LeadElectrodes.dat',' ',0,1);
electrodes = dlmread('/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat',' ',0,1);

% extracts data associated with nodal electrodes and converts to 12-lead
% ECG
[V1 V2 V3 V4 V5 V6 LI LII LIII aVR aVL aVF ECG] = electrodeDataExtractor(electrodes,data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adds data to a structured array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ECG = createDataStructureECG(V1,V2,V3,V4,V5,V6,LI,LII,LIII,aVR,aVL,aVF,electrodes);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots 12-lead traces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot12leadECG(ECG,-3,3);
