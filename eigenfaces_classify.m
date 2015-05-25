
function [ best_face, dist, nn_idx ] = eigenfaces_classify( efm, image, varargin )

p = inputParser;

addRequired(p, 'efm');
addRequired(p, 'image', @ismatrix);
addParameter(p, 'K', 1);
addParameter(p, 'ShowDistances', false);
addParameter(p, 'ShowWeights', false);

parse(p, efm, image, varargin{:});

% compute weights
w = eigenfaces_weights( efm, image );

% use kNN (k=1) to get nearest neighbour
% of course we could also sort the euclidean distances and take the first
% element, but using knnsearch makes it easier to experiement with different
% settings
[nn_idx, dist] = knnsearch( efm.weights, w, 'K', p.Results.K );
best_face = efm.class(nn_idx);

%fprintf('nearest neighbour: #%d (face #%d)\n', nn_idx, best_face_idx);

if p.Results.ShowDistances
    % euclidean distance
    % not necessary, as we use kNN (see below).
    % we keep this to be able to plot some results.
    D = pdist2( efm.weights, w );
    
    % plot euclidean distances
    figure()
    stem(D, 'Marker', 'o', 'LineStyle', ':')
    hold on
    stem(nn_idx, D(nn_idx), 'filled')
    legend('Eigenface', 'Best match')
    title('Euclidean distances');
end

if p.Results.ShowWeights
    % plot weights comparison chart
    figure()
    stem(w)
    hold on
    stem(efm.weights(nn_idx,:), 'Marker', 'x', 'LineStyle', '--')
    legend('Input image', sprintf('Best match #%d', nn_idx));
    title(sprintf('Weights of input image vs. best match (#%d)', nn_idx))
end
end