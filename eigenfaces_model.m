% EIGENFACES_MODEL Trains a eigenfaces model based on the given input
%   I ... input images; h*w*n matrix (h=height, w=width, n=number of images)
%   C ... class of input; 1*n vector/matrix (n=number of images)
%
%   Core algorithm from
%   http://en.wikipedia.org/wiki/Eigenface#Matlab_example_code.
%   Improvements by Absenger, Salomon, Winterer according to
%   http://en.wikipedia.org/wiki/Eigenface#Computing_the_eigenvectors.
function [ efm ] = eigenfaces_model( I, C, varargin )

p = inputParser;
addRequired(p, 'I'); % TODO: add verification function
addOptional(p, 'C', []); % TODO: add verification function
addParameter(p, 'EigenfacesLimit', false);
addParameter(p, 'Optimize', true);
addParameter(p, 'Show', { 'Eigenfaces', false, 'Weights', false });
addParameter(p, 'Variance', 0.95);

parse(p, I, C, varargin{:});

% validate dimensions of I and C
if nargin >= 2 && size(I, ndims(I)) ~= size(C, 2)
    error('dimensions of images I and class C must match!');
end

%% compute eigenfaces
if ndims(I) == 3
    % matrix structure: h*w*n (2d image data)
    [h,w,n] = size(I);
    d = h*w;
elseif ismatrix(I)
    % matrix structure: d*n (linear image data)
    [d,n] = size(I);
    h = 1; w = d; % we don't know the size
else
    error('Unsupported number of dimensions of input image matrix: %d', ndims(I));
end

% vectorize images
x = reshape(I,[d n]);
x = im2double(x)';

% compute mean image
mean_img = mean(x);

% subtract mean
T = bsxfun(@minus, x, mean_img)';

%% calculate eigenvectors and eigenvalues
%tic
if p.Results.Optimize
    %% optimization to avoid huge covariance matrix
    % see http://en.wikipedia.org/wiki/Eigenface#Computing_the_eigenvectors
    
    % General: v is a eigenvector of S, if: S*v == a*v.
    % Covariance matrix S = T*T' => T*T'*v == a*v.
    % We choose u so that T*u = v => T*T'*T*u == a*T*u.
    %
    % instead of TT' (for cov. matrix), we compute T'T, which is much smaller:
    S = T' * T;
    
    % obtain eigenvalue & eigenvector
    [v,D] = eig(S);
    V = T * v;
    
    % normalize eigenvectors
    V_norm = diag(sqrt(V'*V)); % norm of eigenvectors
    V = bsxfun(@times, V', 1 ./ V_norm)';
    
    eigval = diag(D);
else
    %% non-optimized
    % compute covariance matrix
    S = cov(T');    
    
    % obtain eigenvalue & eigenvector
    [V,D] = eig(S);
    eigval = diag(D);
end
%toc

%% sort eigenvalues in descending order
% eigenvalues are already sorted (why?)

%tic
if ~issorted(eigval)
    error('eigenvalues are not sorted!');
end

eigval = eigval(end:-1:1);
V = fliplr(V);

if p.Results.EigenfacesLimit
    M = p.Results.EigenfacesLimit;
    if strcmpi(M, 'auto')
        M = computeNumberOfComponents(p.Results.Variance);
    end
    if length(M) > 1
        % use M as list to define which elements to choose
        V = V(:,M);
        eigval = eigval(M);
    else
        % take highest m values
        V = V(:,1:M);
        eigval = eigval(1:M);
    end
end
%toc

%% compute weights of input faces
W = (V' * T)'; % transform W to have one weight-vector per row

efm = struct('meanface', mean_img, ...
    'eigenfaces', V, ...
    'eigenvalues', eigval, ...
    'weights', W, ...
    'class', p.Results.C, ...
    'imagesize', [h, w]);

eigenfaces_show(efm, p.Results.Show{:});
%showFigures();

    function [ k ] = computeNumberOfComponents( var )
        % evaluate the number of principal components needed to represent var % total variance.
        eigsum = sum(eigval);
        csum = 0;
        k = length(eigval);
        for c_i = 1:length(eigval)
            csum = csum + eigval(c_i);
            tv = csum / eigsum;
            if tv > var
                k = c_i;
                break
            end
        end
    end
end