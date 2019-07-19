function [WAA WAE unitWeigthedDipole] = QRSweightedDipoleAngles_VCG(VCG,QRS_on,QRS_off,dt)

% initialises
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

unitWeigthedDipole = [sin(WAE)*cos(WAA);cos(WAE);sin(WAE)*sin(WAA)];