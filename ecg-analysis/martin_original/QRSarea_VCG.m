function [QRSareaX QRSareaY QRSareaZ QRSarea] = QRSarea_VCG(VCG,QRS_on,QRS_off,dt)

% initialises values to 0
QRSareaX = 0;
QRSareaY = 0;
QRSareaZ = 0;
QRSarea = 0;

% performs numerical integtration to obtain area under the curve during QRS
for i = (QRS_on/dt):(QRS_off/dt)
    
    QRSareaX = QRSareaX + 0.5*(VCG.Vx.data(i+1)+VCG.Vx.data(i))*dt;
    QRSareaY = QRSareaY + 0.5*(VCG.Vy.data(i+1)+VCG.Vy.data(i))*dt;
    QRSareaZ = QRSareaZ + 0.5*(VCG.Vz.data(i+1)+VCG.Vz.data(i))*dt;
    
end

% computes QRS area
QRSarea = sqrt(QRSareaX^2 + QRSareaY^2 + QRSareaZ^2);
    
    