%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filters EGM data (low pass)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function egmFilt = filterEGM(egm,sampleRate,lowP,order)
% inputs:
% egm data as time-series
% sample rate of time-series data in Hz
% low pass filter frequency
% order of filter

% Defines filter window
Wn = lowP/(sampleRate*0.5);

% Computes Butterworth digital filter coefficients
[B,A] = butter(order,Wn,'low');

% Filters EGM data
egmFilt = filtfilt(B,A,egm);
