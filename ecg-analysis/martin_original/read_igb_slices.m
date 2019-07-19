%
% [ data, hd ] = read_igb_slices( file, t_slices )
%
% reads uncompressed igb files
% Reading of compressed igb will be supported in future versions (>Matlab R6.5)
%
% file : igb filename
% data : data set reshaped according to igb header
% hd   : igb header
%
%

function [ data, hd ] = read_igb_slices( file, varargin )


% check input args
mn_args = 1;
mx_args = 2;
n_args  = nargin;

% nargchk( mn_args, mx_args, n_args );
narginchk(mn_args, mx_args);

t_slices = 0;
switch n_args
	case 2
  		t_slices = varargin{1};
end
	

% first we read out the header
[ hd, hdl ] = read_igb_header( file );

if (strcmp(hd.systeme,'big_endian') )
    fopenstr='ieee-be';
else
    fopenstr='ieee-le';
end


% check if t_slices are in the file
if t_slices 
    mx_t_slice = max( t_slices );
    % just take time slices which are in the file
	t_slices = t_slices( t_slices <= hd.t );
    % disp(sprintf('Reading only %d of %d time slices will be read.\n', mx_t_slice, hd.t))
else
	t_slices = 1 : hd.t;
    if isempty(t_slices)
        t_slices = 1;
        disp(sprintf('%c WARNING: Trying to read at least one time slice!\n', 37))
    end
end

% how many time slices we are going to read?
N_slices = length( t_slices );

% expected data size
data_in_file = hd.x * hd.y * hd.z * hd.t;

% data type
switch hd.type
    case 'float',   dbytes = 4; dtype = 'float';
    case 'double',  dbytes = 8; dtype = 'double';
    otherwise       dbytes = 4; dtype = '';
end


% check if file is big enough
d = dir( file );

if d.bytes - 1024 < data_in_file * dbytes

	% adjust data_size
	dtoks = ( d.bytes - 1024 ) / dbytes;
	possible_t_slices = floor( dtoks / hd.x / hd.y / hd.z );
	data_in_file = hd.x * hd.y * hd.z * possible_t_slices;

    msg_l1 = sprintf( 'Mismatch file size with file header: \n' );
	msg_l2 = sprintf( 'Less time slices will be read (%d instead of %d)', possible_t_slices, hd.t ); 
	warning( sprintf( '%s%s', msg_l1, msg_l2 ) );	

end

% open file in big-endian machine format
fh = fopen( file, 'r', fopenstr );

% skip 1024 header bytes 
header = fread( fh, 1024, 'char' );

% preallocate memory
switch hd.type
    case 'float'
        data = zeros( hd.x, hd.y, hd.z, N_slices, 'single');
    otherwise
        data = zeros( hd.x, hd.y, hd.z, N_slices );
end


% size of one time slice
slice_size = hd.x * hd.y * hd.z;

% read data till end of file
actual_timesteps = 0;
for i=1:N_slices
    % compute position of time slice i
    pos = (t_slices(i)-1)*slice_size*dbytes+1024;
    fseek( fh, pos, 'bof' );
    [ slice_buf, count ] = fread( fh, slice_size, dtype );
    
    if count==slice_size
        % disp(sprintf('%c READ_IGB_SLICES: Reading time step %d of %d',37,t_slices(i),hd.t))
        data(:,:,:,i)    = reshape( slice_buf, hd.x, hd.y, hd.z );
        actual_timesteps = actual_timesteps + 1;
    else
        disp( sprintf('%c READ_IGB_SLICES: Incomplete time step %d of %d', 37, i, N_slices) ) 
    end        
end

%
hd.t = actual_timesteps;
if actual_timesteps < N_slices
  data(:,:,:,actual_timesteps+1:end) = [];
end
fclose(fh);


% ---MAKE A SIMPLE IGB CHECK----------------------------------------------------
[pathstr, name] = fileparts(file);
if ~isempty(find(isnan(data(:))))
   disp(sprintf('%c WARNING NaNs were found in the data (%s.igb)!', 37, name))
elseif ~isempty(find(isinf(data(:))))
   disp(sprintf('%c WARNING INFs were found in the data (%s.igb)!', 36, name)) 
end

return
