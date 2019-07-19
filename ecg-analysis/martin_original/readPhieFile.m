function data = readPhieFile(filepath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads-in phie data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paths
% path(path,'/home/mb12/Programmes/MatlabIGB/igb') ;
% path(path,'/home/mb12/Code/Matlab/CEPT/') ;

phiefile = strcat('phie.igb');
phiepath = strcat(filepath,phiefile);
[data_tmp,hd] = read_igb(phiepath);
data = zeros(size(data_tmp,1),size(data_tmp,4));
time = size(data_tmp,4);
for i = 1:time
    data(:,i) = data_tmp(:,1,1,i);
end
