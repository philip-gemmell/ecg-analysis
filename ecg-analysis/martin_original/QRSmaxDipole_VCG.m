function [maxDipoleMag maxDipole maxTime] = QRSmaxDipole_VCG(VCG,QRS_on,QRS_off,dt)

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
    
    
    