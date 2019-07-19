%% FILES INCORPORATED TO THIS LIST
% CONFIRMED TO MATCH PYTHON:
% electrodeDataExtractor.m
% FrankLeadTransformer_KORS.m
% QRSdetection_spatialVCGvelocity.m (with flag enabled)
% QRSarea_VCG.m
% QRSmaxDipole_VCG.m
% QRSmeanDipoleMag.m
% 		QRSmeanDipoleMag_VCG.m (identical)
% QRSweightedDipoleAngles_VCG.m
% 		QRSweighted_VCG.m (identical)
%
% TO BE CONFIRMED:

%%% FILES NOT INCORPORATED
% UNNECESSARY:
% createDataStructureECG.m
% ecg12LeadElectrodes.m
% filterECG.m
% filterEGM.m
% plot12leadECG.m
% read_igb_header.m
% read_igb.m
% read_igb_slices.m
% readPhieFile.m
%
% TO TEST MANUALLY:
% QRS_deltaDipoleVectors.m
% QRS_deltaDecimatedDipoles_VCG.m
%
% TO DO:


phie_files = {'/data/ecg/lv_phi1.5708-3.1416_rho0.1-0.9_z0.3-0.9_bcl300/ecg/phie.igb',...
    '/data/ecg/lv_phi1.5708-3.1416_rho0.2-0.8_z0.3-0.9_bcl300/ecg/phie.igb',...
    '/data/ecg/lv_phi1.5708-3.1416_rho0.3-0.7_z0.3-0.9_bcl300/ecg/phie.igb',...
    '/data/ecg/lv_phi1.5708-3.1416_rho0.4-0.6_z0.3-0.9_bcl300/ecg/phie.igb'};

%%% SET-UP CONSTANT VARIABLES

KORS = zeros(8,3);
KORS(1,1) = 0.38;
KORS(1,2) = -0.07;
KORS(1,3) = 0.11;
KORS(2,1) = -0.07;
KORS(2,2) = 0.93;
KORS(2,3) = -0.23;
KORS(3,1) = -0.13;
KORS(3,2) = 0.06;
KORS(3,3) = -0.43;
KORS(4,1) = 0.05;
KORS(4,2) = -0.02;
KORS(4,3) = -0.06;
KORS(5,1) = -0.01;
KORS(5,2) = -0.05;
KORS(5,3) = -0.14;
KORS(6,1) = 0.14;
KORS(6,2) = 0.06;
KORS(6,3) = -0.20;
KORS(7,1) = 0.06;
KORS(7,2) = -0.17;
KORS(7,3) = -0.11;
KORS(8,1) = 0.54;
KORS(8,2) = 0.13;
KORS(8,3) = 0.31;
electrodes = dlmread('/home/pg16/Documents/ecg-scar/ecg-analysis/martin_original/12LeadElectrodes.dat',' ',0,1);
nV1 = electrodes(1)+1;
nV2 = electrodes(2)+1;
nV3 = electrodes(3)+1;
nV4 = electrodes(4)+1;
nV5 = electrodes(5)+1;
nV6 = electrodes(6)+1;
nRA = electrodes(7)+1;
nLA = electrodes(8)+1;
nRL = electrodes(9)+1;
nLL = electrodes(10)+1;
dt=2;
blank=5;
sampleRate = 1000/dt;
lowP = 40;
order = 2;

%%% CREATE STORAGES MATRICES

n_files = length(phie_files);
qrs_duration_full = zeros(n_files,1);
qrs_start_full = zeros(n_files,1);
qrs_end_full = zeros(n_files,1);
qrs_area_full = zeros(n_files,1);
qrs_areaX_full = zeros(n_files,1);
qrs_areaY_full = zeros(n_files,1);
qrs_areaZ_full = zeros(n_files,1);
waa_full = zeros(n_files,1);
wae_full = zeros(n_files,1);
unitWeightedDipole_full = zeros(n_files,3);
maxDipole_mag = zeros(n_files,1);
maxDipole_time = zeros(n_files,1);
meanDipole_mag = zeros(n_files,1);
vcg_full = cell(n_files,1);

for idx = 1:n_files
	
	%% EXTRACT ECG DATA (electrodeDataExtractor.m)
	
	[data_tmp,hd] = read_igb(phie_files{idx});
	data = zeros(size(data_tmp,1),size(data_tmp,4));
	time = size(data_tmp,4);
	for i = 1:time
		data(:,i) = data_tmp(:,1,1,i);
	end
	eV1 = data(nV1,:);
	eV2 = data(nV2,:);
	eV3 = data(nV3,:);
	eV4 = data(nV4,:);
	eV5 = data(nV5,:);
	eV6 = data(nV6,:);
	eRA = data(nRA,:);
	eLA = data(nLA,:);
	eRL = data(nRL,:);
	eLL = data(nLL,:);
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
	ECGmatrix = [ECG.LI.data,ECG.LII.data,ECG.V1.data,ECG.V2.data,ECG.V3.data,ECG.V4.data,ECG.V5.data,ECG.V6.data];

	%% CONVERT TO VCG (FrankLeadTransformer_KORS.m)

	VCGmatrix = ECGmatrix * KORS;
	Vx = VCGmatrix(:,1);
	Vy = VCGmatrix(:,2);
	Vz = VCGmatrix(:,3);
	VCG.Vx.data = Vx;
	VCG.Vy.data = Vy;
	VCG.Vz.data = Vz;
    vcg_full{idx} = VCG;

	%% EXTRACT QRS START/END/DURATION DATA (QRSdetection_spatialVCGvelocity.m)
	
	dVx = zeros(length(VCG.Vx.data),1);
	dVy = zeros(length(VCG.Vx.data),1);
	dVz = zeros(length(VCG.Vx.data),1);
	for i = 2:length(VCG.Vx.data)-1
		 dVx(i) = (VCG.Vx.data(i+1) - VCG.Vx.data(i-1))/2*dt;
		 dVy(i) = (VCG.Vy.data(i+1) - VCG.Vy.data(i-1))/2*dt;
		 dVz(i) = (VCG.Vz.data(i+1) - VCG.Vz.data(i-1))/2*dt; 
	end
	% computes the spatial velocity from these gradients
	for i = 2:length(VCG.Vx.data)-1
		 SV(i) = sqrt(dVx(i)^2 + dVy(i)^2 + dVz(i)^2);
	end

	SV = SV(blank:end);
	% filters it first
	SVfilt = filterEGM(SV,sampleRate,lowP,order);
	% finds max
	maxSV = max(SV);
	% determines threshold
	thresh = maxSV*0.2;
	% finds start of QRS
	QRS_on = -1;
	QRS_off = -1;
	for i = 1:length(SVfilt)-1
		 if SVfilt(i) > thresh
		     QRS_on = (blank+i)*dt;
		     i_qrs_on = i;
		     break;
		 end
	end
	% finds end of QRS
	for i = length(SVfilt)-1:-1:QRS_on/dt
		 if SVfilt(i) > thresh 
		     QRS_off = (blank+i)*dt;
		     i_qrs_off = i;
		     break;
		 end
	end
	QRSduration = QRS_off - QRS_on;
	
	qrs_duration_full(idx) = QRSduration;
	qrs_start_full(idx) = QRS_on;
	qrs_end_full(idx) = QRS_off;
	
	%% CALCULATE QRS AREA (QRSarea_VCG.m)
	
	QRSareaX = 0;
	QRSareaY = 0;
	QRSareaZ = 0;
	% performs numerical integtration to obtain area under the curve during QRS
	for i = (QRS_on/dt):(QRS_off/dt)
		 
		 QRSareaX = QRSareaX + 0.5*(VCG.Vx.data(i+1)+VCG.Vx.data(i))*dt;
		 QRSareaY = QRSareaY + 0.5*(VCG.Vy.data(i+1)+VCG.Vy.data(i))*dt;
		 QRSareaZ = QRSareaZ + 0.5*(VCG.Vz.data(i+1)+VCG.Vz.data(i))*dt;
		 
	end
	% computes QRS area
	QRSarea = sqrt(QRSareaX^2 + QRSareaY^2 + QRSareaZ^2);
	
	qrs_area_full(idx) = QRSarea;
	qrs_areaX_full(idx) = QRSareaX;
	qrs_areaY_full(idx) = QRSareaY;
	qrs_areaZ_full(idx) = QRSareaZ;
	
	%% CALCULATE MEAN DIPOLE MAGNITUDE (QRSmeanDipoleMag.m)
	
	totalDipoleMag = 0;

	counter = 0;
	% iterates over entire QRS wave
	for i = (QRS_on/dt):(QRS_off/dt)

		 % defines the dipole explicitly from the VCG components
		 dipole = [VCG.Vx.data(i);VCG.Vy.data(i);VCG.Vz.data(i)];

		 % computes the magnitude
		 dipoleMag = sqrt(dipole(1)^2 + dipole(2)^2 + dipole(3)^2);
		 
		 totalDipoleMag = totalDipoleMag + dipoleMag;
		 
		 counter = counter+1;
		 
	end
	meanWeightedDipoleMag = totalDipoleMag/counter;
	meanDipole_mag(idx) = meanWeightedDipoleMag;
	
	%% FIND MAXIMUM DIPOLE MAGNITUDE AND TIME (AND COORDINATES) (QRSmaxDipole_VCG.m)
	
	% initialises
	maxDipole = -1;
	maxDipoleMag = -1;
	maxTime = -1;

	% iterates over entire QRS loop
	for i = (QRS_on/dt):(QRS_off/dt)
		 
		 % defines the dipole explicitly from the VCG components
		 dipole = [VCG.Vx.data(i);VCG.Vy.data(i);VCG.Vz.data(i)];
		 
		 % computes the magnitude
		 dipoleMag = sqrt(dipole(1)^2 + dipole(2)^2 + dipole(3)^2);
		 
		 % stores the max
		 if dipoleMag > maxDipoleMag
		     maxDipoleMag = dipoleMag;
		     maxTime = i*dt;
		 end
		 
	end
	% outputs the found maximum dipole vector
	maxDipole = [VCG.Vx.data(maxTime/dt);VCG.Vy.data(maxTime/dt);VCG.Vz.data(maxTime/dt)];
	
	maxDipole_mag(idx) = maxDipoleMag;
	maxDipole_time(idx) = maxTime;
	
	%% CALCULATE WEIGHT-AVERAGED ELEVATION/AZIMUTH OF DIPOLE DURING QRS (QRSweightedDipoleAngles_VCG.m)
	
	WAA = 0;
	WAE = 0;
	totalDipoleMag = 0;

	% iterates over entire QRS wave
	for i = (QRS_on/dt):(QRS_off/dt)

		 % defines the dipole explicitly from the VCG components
		 dipole = [VCG.Vx.data(i);VCG.Vy.data(i);VCG.Vz.data(i)];

		 % computes the magnitude
		 dipoleMag = sqrt(dipole(1)^2 + dipole(2)^2 + dipole(3)^2);
		 
		 % elevation
		 phi = acos(VCG.Vy.data(i)/dipoleMag);

		 % azimuthal
		 theta = atan(VCG.Vz.data(i)/VCG.Vx.data(i));
		 
		 % computes the rolling weighted average angles and total dipole
		 % amplitude
		 WAA = WAA + theta*dipoleMag;
		 WAE = WAE + phi*dipoleMag;
		 totalDipoleMag = totalDipoleMag + dipoleMag;
		 
	end

	WAA = WAA/totalDipoleMag;
	WAE = WAE/totalDipoleMag; 

	unitWeightedDipole = [sin(WAE)*cos(WAA);cos(WAE);sin(WAE)*sin(WAA)];
	waa_full(idx) = WAA;
	wae_full(idx) = WAE;
	unitWeightedDipole_full(idx,:) = unitWeightedDipole;
	
end
