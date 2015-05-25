function [ weights ] = eigenfaces_weights( efm, image )
%EIGENFACES_WEIGHTS Computes the eigenface weights of the given image
%   Detailed explanation goes here

    % computation: W = V' * (I - M);
    % W ... weights
    % V ... eigenfaces
    % I ... image to classify
    % M ... mean face
    
    % convert input image to vector
    img = im2double(reshape(image, numel(image), 1));
    
    % compute diff image
    diff = img - efm.meanface';
    
    weights = (efm.eigenfaces' * diff)';
end

