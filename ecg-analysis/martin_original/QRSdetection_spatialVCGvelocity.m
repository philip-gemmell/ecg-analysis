function [QRS_on QRS_off SV QRSduration] = QRSdetection_spatialVCGvelocity(VCG,dt,blank)

SV = zeros(length(VCG.Vx.data),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computes spatial velocity of VCG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computes spatial derivatives of Vx, Vy, Vz first
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determines threshold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gets-rid of first part (due to tail of rep wave)
SV = SV(blank:end);

% filters it first
sampleRate = 1000/dt;
lowP = 40;
order = 2;
SVfilt = filterEGM(SV,sampleRate,lowP,order);

% finds max
maxSV = max(SV);

% determines threshold
thresh = maxSV*0.2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% looks through the newly computed SV and determines the QRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% finds start of QRS
QRS_on = -1;
QRS_off = -1;
for i = 1:length(SVfilt)-1
    if SVfilt(i) > thresh
        QRS_on = (blank+i)*dt;
        break;
    end
end
% finds end of QRS
for i = length(SVfilt)-1:-1:QRS_on/dt
    if SVfilt(i) > thresh 
        QRS_off = (blank+i)*dt;
        break;
    end
end
       
QRSduration = QRS_off - QRS_on;