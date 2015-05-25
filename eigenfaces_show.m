function [ ] = eigenfaces_show( efm, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

addRequired(p, 'efm');
addParameter(p, 'Eigenfaces', false);
addParameter(p, 'Weights', false);

parse(p, efm, varargin{:});

if p.Results.Eigenfaces
    %% show Eigenfaces
    V = efm.eigenfaces;
    if p.Results.Eigenfaces == 1
        idx = (1:min(15, size(V, 2)));
    else
        idx = (p.Results.Eigenfaces);
    end
    
    show_eigenfaces( efm, idx );
end

if p.Results.Weights
    %% show weight plots
    W = efm.weights;
    if p.Results.Weights == 1
        idx = (1:min(16, size(W, 1)));
    else
        idx = (p.Results.Weights);
    end
    
    show_weights( efm.weights, idx );
end

end

function [] = show_weights( weights, idx )
s = size(idx, 2);
d = ceil(sqrt(s));

figure('name', 'Weights of input images')
colormap gray
h = zeros(s);
for i = 1:s
    h(i) = subplot(d, d, i);
    stem(weights(idx(i),:))
    title(sprintf('weights(%d,:)', idx(i)))
end
linkaxes(h, 'xy');
%toc
end

function [] = show_eigenfaces( efm, idx )
s = size(idx, 2);
d = ceil(sqrt(s + 1));
eig0 = reshape(efm.meanface, efm.imagesize);

figure('name', 'Eigenfaces')
h = zeros(s + 1);
h(1) = subplot(d, d, 1);
imagesc(eig0)
title('mean');
colormap gray;
axis off;
for i = 1:s
    h(i+1) = subplot(d, d, i+1);
    imagesc(reshape(efm.eigenfaces(:,i), efm.imagesize))
    title(sprintf('face %d', idx(i)))
    axis off;
end
linkaxes(h, 'xy');
%toc
end
