function [V1 V2 V3 V4 V5 V6 LI LII LIII aVR aVL aVF ECG] = electrodeDataExtractor(electrodes,data)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracts phie data from file giving list of electrodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paths
path(path,'/home/mb12/Programmes/MatlabIGB/igb') ;
path(path,'/home/mb12/Code/Matlab/CEPT/') ;
  
% pulls-out data from individual electrodes
nV1 = electrodes(1)+1;
eV1 = data(nV1,:);
nV2 = electrodes(2)+1;
eV2 = data(nV2,:);
nV3 = electrodes(3)+1;
eV3 = data(nV3,:);
nV4 = electrodes(4)+1;
eV4 = data(nV4,:);
nV5 = electrodes(5)+1;
eV5 = data(nV5,:);
nV6 = electrodes(6)+1;
eV6 = data(nV6,:);
nRA = electrodes(7)+1;
eRA = data(nRA,:);
nLA = electrodes(8)+1;
eLA = data(nLA,:);
nRL = electrodes(9)+1;
eRL = data(nRL,:);
nLL = electrodes(10)+1;
eLL = data(nLL,:);

% Wilson Central Terminal
WCT = (eLA + eRA + eLL);

% V leads
V1 = eV1 - WCT/3;
V2 = eV2 - WCT/3;
V3 = eV3 - WCT/3;
V4 = eV4 - WCT/3;
V5 = eV5 - WCT/3;
V6 = eV6 - WCT/3;

% Einhoven limb leads
LI = eLA - eRA;
LII = eLL - eRA;
LIII = eLL - eLA;

% augmented leads
aVR = eRA - 0.5*(eLA + eLL);
aVL = eLA - 0.5*(eRA + eLL);
aVF = eLL - 0.5*(eLA + eRA);

V1 = V1';
V2 = V2';
V3 = V3';
V4 = V4';
V5 = V5';
V6 = V6';
LI = LI';
LII = LII';
LIII = LIII';
aVR = aVR';
aVL = aVL';
aVF = aVF';

% Also creates a data structure of ECG data
ECG.V1.data = V1;
ECG.V2.data = V2;
ECG.V3.data = V3;
ECG.V4.data = V4;
ECG.V5.data = V5;
ECG.V6.data = V6;
ECG.LI.data = LI;
ECG.LII.data = LII;
ECG.LIII.data = LIII;
ECG.aVR.data = aVR;
ECG.aVL.data = aVL;
ECG.aVF.data = aVF;

