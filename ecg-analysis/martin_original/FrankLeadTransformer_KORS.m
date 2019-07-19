function [Vx Vy Vz VCG] = FrankLeadTransformer_KORS(ECG)

% Defines KORS matrix for conversion
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

% makes an ECG matrix of relevant leads
ECGmatrix = [ECG.LI.data,ECG.LII.data,ECG.V1.data,ECG.V2.data,ECG.V3.data,ECG.V4.data,ECG.V5.data,ECG.V6.data];

% VCG calculation
VCGmatrix = ECGmatrix * KORS;

Vx = VCGmatrix(:,1);
Vy = VCGmatrix(:,2);
Vz = VCGmatrix(:,3);

VCG.Vx.data = Vx;
VCG.Vy.data = Vy;
VCG.Vz.data = Vz;
