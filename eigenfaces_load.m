function [ I A ] = eigenfaces_load( basepath, extension, varargin )
% LOAD_IMAGES Loads images from all sub-folders
%   I = LOAD_IMAGES(path, ext) loads all images with the given extension
%       from all subfolders of path.
%   [I A] = LOAD_IMAGES(path, ext) as above, but also returns a vector
%       specifying the index of the sub directory for each image.

p = inputParser;

addRequired(p, 'basepath', @ischar);
addRequired(p, 'extension', @ischar);
% addParameter(p, 'ShowImages', false);

parse(p, basepath, extension, varargin{:});

% list directory content
dirs = dir(p.Results.basepath);
dirs(~[dirs.isdir]) = [];  %remove non-directories
dir_names = { dirs.name }';
dir_names(ismember(dir_names, {'.', '..'})) = [];

% result matrices
I = [];
A = [];

%% iterate all sub-folders
for i = 1:length(dir_names)
    dir_path = fullfile(basepath, dir_names{i});
    files = dir(fullfile(dir_path, p.Results.extension));
    file_names = { files.name }';
    
    %% iterate files within a certain sub-folder
    for j = 1:length(file_names)
        file_path = fullfile(dir_path, file_names{j});
        img = imread(file_path);
        I = cat(3, I, img);
    end
    A = [A repelem(i, length(file_names))];
end
end
