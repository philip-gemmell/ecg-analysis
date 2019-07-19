function [dTtotal] = QRS_deltaDecimatedDipoles_VCG(VCG1,QRS_on1,QRS_off1,VCG2,QRS_on2,QRS_off2,dt)

dTtotal = zeros(10,1);

% iterates over the entire QRS
for i = 1:10
    
    % determines deciles
    dec1 = round((QRS_on1 + i*(QRS_off1-QRS_on1)/10)/dt);
    dec2 = round((QRS_on2 + i*(QRS_off2-QRS_on2)/10)/dt);
    
    % defines the dipole at this moment in time
    dipole1 = [VCG1.Vx.data(dec1);VCG1.Vy.data(dec1);VCG1.Vz.data(dec1)];
    dipole2 = [VCG2.Vx.data(dec2);VCG2.Vy.data(dec2);VCG2.Vz.data(dec2)];
    
    % computes angular difference between dipoles
    dTtotal(i) = acos(dot(dipole1,dipole2)/(norm(dipole1)*norm(dipole2)))*180/pi;
    
end

    
    
    