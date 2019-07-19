%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defines idealised scar patter in 3D torso model using UVC
% Martin Bishop
% KCL
% 23rd November 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads-in elems of current mesh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads-in elems file for torso
% torsoElems = dlmread('/data/Projects/Torso/meshes/torso_final_ref_smooth_noAir_myoFastEndo.elem',' ',1,1);
torsoElems = dlmread('meshes/torso_final_ref_smooth_noAir_myoFastEndo.elem',' ',1,1);
torsoElems(:,1:4) = torsoElems(:,1:4) + 1;
% switches back any previously-made scar
for i = 1:length(torsoElems)
    if torsoElems(i,5) == 200
        torsoElems(i,5) = 22;
    end
end

% Loads-in fibres file for torso
% torsoFibres = dlmread('/data/Projects/Torso/meshes/torso_final_ref_smooth_noAir_myoFIBRES.lon',' ',1,0);
torsoFibres = dlmread('meshes/torso_final_ref_smooth_noAir_myoFIBRES.lon',' ',1,0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads-in UVC of Mesh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% torsoUVC = dlmread('/data/Projects/Torso/meshes/UVC/BIV/UVC/COMBINED_COORDS_Z_RHO_PHI_V.pts',' ',1,0);
torsoUVC = dlmread('meshes/COMBINED_COORDS_Z_RHO_PHI_V.pts',' ',1,0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Makes centroid-based UVC coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torsoCentroidUVC = zeros(length(torsoElems),4);
for i = 1:length(torsoElems)
    
    % V
    torsoCentroidUVC(i,4) = 0.25*(torsoUVC(torsoElems(i,1),4) + torsoUVC(torsoElems(i,2),4) + torsoUVC(torsoElems(i,3),4) + torsoUVC(torsoElems(i,4),4));
    if torsoCentroidUVC(i,4) == -1
        torsoCentroidUVC(i,4) = -1;
    else 
        torsoCentroidUVC(i,4) = 1;
    end
    
    % Z
    torsoCentroidUVC(i,1) = 0.25*(torsoUVC(torsoElems(i,1),1) + torsoUVC(torsoElems(i,2),1) + torsoUVC(torsoElems(i,3),1) + torsoUVC(torsoElems(i,4),1));
    
    % RHO
    torsoCentroidUVC(i,2) = 0.25*(torsoUVC(torsoElems(i,1),2) + torsoUVC(torsoElems(i,2),2) + torsoUVC(torsoElems(i,3),2) + torsoUVC(torsoElems(i,4),2));
    if torsoCentroidUVC(i,2) > 1
        torsoCentroidUVC(i,2) = 1;
    end
    % checks for sign flip
    flag1 = 0;
    flag2 = 0;
    for j = 1:4
        if torsoUVC(torsoElems(i,j),2) > 0.9
            flag1 = 1;
        elseif  torsoUVC(torsoElems(i,j),2) < 0.1
            flag2 = 1;
        end
    end
    % for the case of a sign flip
    if flag1*flag2 ~= 0
        torsoCentroidUVC(i,2) = 1;
    end
    
    % PHI
    % checks for sign flip
    flag1 = 0;
    flag2 = 0;
    for j = 1:4
        if torsoUVC(torsoElems(i,j),3) > pi - 0.3
            flag1 = 1;
        elseif  torsoUVC(torsoElems(i,j),3) < -(pi - 0.3)
            flag2 = 1;
        end
    end
    % for no sign flip
    torsoCentroidUVC(i,3) = 0.25*(torsoUVC(torsoElems(i,1),3) + torsoUVC(torsoElems(i,2),3) + torsoUVC(torsoElems(i,3),3) + torsoUVC(torsoElems(i,4),3));
    % for the case of a sign flip
    if flag1*flag2 ~= 0
        torsoCentroidUVC(i,3) = pi;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define scar within torso model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% limits for circumferential extent
lowerPhi = -0.25;
upperPhi = 0.25;
 
% limits of transmural extent
lowerRho = 0.1;
upperRho = 0.9;
 
% limits of apex-base extent
lowerZ = 0.3;
upperZ = 0.9;

% defines scar with elements
torsoElemsScar = torsoElems;
torsoFibresScar = torsoFibres;
c = 0;
c2 = 0;

% percolation threshold (default)
plow = 0.6;
p_dense = 0.9;
p_BZ = 0.75;
tag = -1;
for i = 1:length(torsoElems)
    % checks if in LV
    if torsoCentroidUVC(i,4) == -1
        % checks between limits of Z
        if torsoCentroidUVC(i,1) < upperZ && torsoCentroidUVC(i,1) > lowerZ
            % checks between limits of Rho
            if torsoCentroidUVC(i,2) < upperRho && torsoCentroidUVC(i,2) > lowerRho
                % checks between limits of Phi
                if torsoCentroidUVC(i,3) < upperPhi && torsoCentroidUVC(i,3) > lowerPhi
                    
                    c = c+1;
                    
                    % if within main scar zone, set to lowest density
                    p = plow;
                    tag = 200;
                    
                    % does percolation in layer with 3 separate regions
                    if torsoCentroidUVC(i,3) > ((upperPhi - lowerPhi)*(1/12) + lowerPhi) && torsoCentroidUVC(i,3) < ((upperPhi - lowerPhi)*(11/12) + lowerPhi)
                        
                        if torsoCentroidUVC(i,2) > ((upperRho - lowerRho)*(1/12) + lowerRho) && torsoCentroidUVC(i,2) < ((upperRho - lowerRho)*(11/12) + lowerRho)
                            % if within second region, set to intermediate
                            % level
                            p = p_BZ;
                            tag = 201;
                            if torsoCentroidUVC(i,3) > ((upperPhi - lowerPhi)*(1/6) + lowerPhi) && torsoCentroidUVC(i,3) < ((upperPhi - lowerPhi)*(5/6) + lowerPhi)
                                
                                if torsoCentroidUVC(i,2) > ((upperRho - lowerRho)*(1/6) + lowerRho) && torsoCentroidUVC(i,2) < ((upperRho - lowerRho)*(5/6) + lowerRho)
                                    % if within central region, set to most
                                    % dense
                                    p = p_dense;
                                    tag = 202;
                                end
                            end
                        end
                    end
                  
                    % Adds in the relevant tag to reflect the layer we are
                    % in
                    torsoElemsScar(i,5) = tag;
                    % Adds percolation definition to null fibres here too
                    if rand(1) < p
                        torsoFibresScar(i,:) = 0;
                        c2 = c2+1;
                    end
                    
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writes-out data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torsoElemsScar(:,1:4) = torsoElemsScar(:,1:4) - 1;
% dlmwrite(strcat('/data/Projects/Torso/meshes/midwallFibrosis/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.elem'),length(torsoElemsScar),'precision',9,'delimiter',' ');
% dlmwrite(strcat('/data/Projects/Torso/meshes/midwallFibrosis/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.elem'),torsoElemsScar,'-append','precision',9,'delimiter',' ');
% 
% dlmwrite(strcat('/data/Projects/Torso/meshes/midwallFibrosis/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.lon'),1,'precision',9,'delimiter',' ');
% dlmwrite(strcat('/data/Projects/Torso/meshes/midwallFibrosis/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.lon'),torsoFibresScar,'-append','precision',9,'delimiter',' ');

dlmwrite(strcat('meshes/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.elem'),length(torsoElemsScar),'precision',9,'delimiter',' ');
dlmwrite(strcat('meshes/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.elem'),torsoElemsScar,'-append','precision',9,'delimiter',' ');

dlmwrite(strcat('meshes/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.lon'),1,'precision',9,'delimiter',' ');
dlmwrite(strcat('meshes/myoSCARSeptum_MIXED_B',num2str(lowerPhi),'_',num2str(upperPhi),'_',num2str(lowerRho),'_',num2str(upperRho),'_',num2str(lowerZ),'_',num2str(upperZ),'.lon'),torsoFibresScar,'-append','precision',9,'delimiter',' ');
