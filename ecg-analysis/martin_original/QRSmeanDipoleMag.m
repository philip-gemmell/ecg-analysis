function [meanWeightedDipoleMag] = QRSmeanDipoleMag_VCG(VCG,QRS_on,QRS_off,dt)

% initialises
meanWeightedDipoleMag= zeros(3,1);
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
