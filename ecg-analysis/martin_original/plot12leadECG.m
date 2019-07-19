function plot12leadECG(ECG,yMin,yMax)

subplot(2,6,1);plot(ECG.V1.data,'LineWidth',5);title('V1');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,2);plot(ECG.V2.data,'LineWidth',5);title('V2');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,3);plot(ECG.V3.data,'LineWidth',5);title('V3');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,4);plot(ECG.V4.data,'LineWidth',5);title('V4');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,5);plot(ECG.V5.data,'LineWidth',5);title('V5');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,6);plot(ECG.V6.data,'LineWidth',5);title('V6');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,7);plot(ECG.LI.data,'LineWidth',5);title('LeadI');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,8);plot(ECG.LII.data,'LineWidth',5);title('LeadII');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,9);plot(ECG.LIII.data,'LineWidth',5);title('LeadIII');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,10);plot(ECG.aVR.data,'LineWidth',5);title('aVR');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,11);plot(ECG.aVL.data,'LineWidth',5);title('aVL');
%axis([0 length(ECG.V1.data) yMin yMax]);
subplot(2,6,12);plot(ECG.aVF.data,'LineWidth',5);title('aVF');
%axis([0 length(ECG.V1.data) yMin yMax]);


