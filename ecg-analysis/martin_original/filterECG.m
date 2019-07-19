function [V1f V2f V3f V4f V5f V6f LIf LIIf LIIIf aVRf aVLf aVFf ECGf] = filterECG(ECG,sampleRate,lowP,order)

% filters all signals
V1f = filterEGM(ECG.V1.data,sampleRate,lowP,order);
V2f = filterEGM(ECG.V2.data,sampleRate,lowP,order);
V3f = filterEGM(ECG.V3.data,sampleRate,lowP,order);
V4f = filterEGM(ECG.V4.data,sampleRate,lowP,order);
V5f = filterEGM(ECG.V5.data,sampleRate,lowP,order);
V6f = filterEGM(ECG.V6.data,sampleRate,lowP,order);
LIf = filterEGM(ECG.LI.data,sampleRate,lowP,order);
LIIf = filterEGM(ECG.LII.data,sampleRate,lowP,order);
LIIIf = filterEGM(ECG.LIII.data,sampleRate,lowP,order);
aVRf = filterEGM(ECG.aVR.data,sampleRate,lowP,order);
aVLf = filterEGM(ECG.aVL.data,sampleRate,lowP,order);
aVFf = filterEGM(ECG.aVF.data,sampleRate,lowP,order);

% Also creates a data structure of ECG data
ECGf.V1.data = V1f;
ECGf.V2.data = V2f;
ECGf.V3.data = V3f;
ECGf.V4.data = V4f;
ECGf.V5.data = V5f;
ECGf.V6.data = V6f;
ECGf.LI.data = LIf;
ECGf.LII.data = LIIf;
ECGf.LIII.data = LIIIf;
ECGf.aVR.data = aVRf;
ECGf.aVL.data = aVLf;
ECGf.aVF.data = aVFf;
