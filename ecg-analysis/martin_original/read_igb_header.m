%
% [ hd, hdl ] = read_igb_header( file )
%
% reads igb file header
%
% Input:
%
% file : filename
%
% Output:
%
% hd   : header
% hdl  : header as a line (first 1024 bytes of igb file)
%
%
%

function [ hd, hdl ] = read_igb_header( file )

fh = fopen( file, 'r' );

if fh ~= -1
	hdl = fread( fh, 1024, 'char' );
	fclose(fh);
else
	error( sprintf( '%s does not exist.', file ) );	
end


% remove linefeeds and carriage returns

hdl = hdl( hdl ~= 10 );
hdl = hdl( hdl ~= 13 );

% transpose to line
hdl = char( hdl' );

% find delimiters: spaces between fields and colons between
% fieldname and value
%
% add a 0 to the spaces field where we store the position
% of all spaces to make the parsing easier

%hdl    = deblank( hdl );
spaces   = [ 0 findstr( hdl, ' ' ) ];
colons   = findstr( hdl, ':' );


% initialize header structure
% use NaN for numbers and '' for strings as initializer
hd.x        = NaN;
hd.y        = NaN;
hd.z        = NaN;
hd.t        = NaN;
hd.dim_x    = NaN;
hd.dim_y    = NaN;
hd.dim_z    = NaN;
hd.dim_t    = NaN;
hd.unites_x = '';
hd.unites_y = '';
hd.unites_z = '';
hd.unites_t = '';
hd.unites   = '';
hd.type     = '';
hd.systeme  = '';
hd.facteur  = NaN;
hd.fac_t    = NaN;
hd.zero     = NaN;

header    = hdl;
numTokens = length( colons );

for i=1:numTokens
    [token, header] = strtok( header );
    colon           = findstr( token, ':' );
	value           = token( colon+1:end );
    token           = strtrim(token( 1:colon-1 ));  % Remove all blanks    
	
	switch token
		case 'x',           hd.x        = str2num( value );
		case 'y',           hd.y        = str2num( value );
		case 'z',           hd.z        = str2num( value );
		case 't',           hd.t        = str2num( value );
		case 'dim_x',       hd.dim_x    = str2num( value );
		case 'dim_y',       hd.dim_y    = str2num( value );
		case 'dim_z',       hd.dim_z    = str2num( value );
		case 'dim_t',       hd.dim_t    = str2num( value );
		case 'unites_x',    hd.unites_x = value;
		case 'unites_y',    hd.unites_y = value;
		case 'unites_z',    hd.unites_z = value;
		case 'unites_t',    hd.unites_t = value;
		case 'unites',  	hd.unites   = value;
		case 'type',        hd.type     = value;
		case 'systeme',     hd.systeme  = value;
        case 'fac_t',       hd.fac_t    = str2num( value );
		case 'facteur',     hd.facteur  = str2num( value );
    	case 'org_t',       hd.org_t    = str2num( value );
		case 'zero',        hd.zero     = str2num( value );
        otherwise
            disp(sprintf('%c READ_IGB_HEADER: Unrecognized token %s', 37, token))
    end
end

% ---WANT TO SEE IF THE DATA WAS READ CORRECTLY---------------------------------
pause(1)
