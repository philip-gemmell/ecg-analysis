function [dT] = QRS_deltaDipoleVectors(WAA1,WAE1,WAA2,WAE2)

dT = acos(sin(WAE1)*cos(WAA1)*sin(WAE2)*cos(WAA2) + cos(WAE1)*cos(WAE2) + sin(WAE1)*sin(WAA1)*sin(WAE2)*sin(WAA2));
dT = dT*180/pi;
