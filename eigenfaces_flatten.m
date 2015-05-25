function [ I_flat ] = eigenfaces_flatten( I )
%EIGENFACES_FLATTEN flattens the image and classifier data
%   Detailed explanation goes here

    [w,h,n] = size(I);
    d = w*h;
    I_flat = double(reshape(I, [d, n]))';
end
