%
% [ data, hd ] = read_igb( file )
%
% reads uncompressed igb files
% Reading of compressed igb will be supported in future versions (>Matlab R6.5)
%
% file : igb filename
% data : data set reshaped according to igb header
% hd   : igb header
%
%

function [ data, hd ] = read_igb( file )


[ data, hd ] = read_igb_slices( file );