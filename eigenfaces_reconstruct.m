function [ img ] = eigenfaces_reconstruct( efm, weights )
%EIGENFACES_RECONSTRUCT Reconstructs a face specified by a set of weights
%   Given a trained Eigenfaces model and a set of weights, this function
%   reconstructs the original face.
%   Math: F = Mean + V * W';

    img = efm.eigenfaces * weights' + efm.meanface';
    img = reshape(img, efm.imagesize);
end
