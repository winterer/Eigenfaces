%% init variables
path = './att_faces';
ext  = '*.pgm';
imageToClassify = './att_faces/s1/1.pgm';

%% load images
[I, C] = eigenfaces_load(path, ext);

%% train eigenfaces model
%efm = eigenfaces_model(I, C, 'EigenfacesLimit', 'auto', 'ShowEigenfaces', 1:8, 'ShowWeights', 1:9);
efm = eigenfaces_model(I, C, 'EigenfacesLimit', 'auto', 'Show', { 'Eigenfaces', true, 'Weights', 1:9});
disp(efm);

%% show figures
% mean + first 24 eigenfaces
eigenfaces_show(efm, 'Eigenfaces', 1:24);

% weights of the first 16 input faces
eigenfaces_show(efm, 'Weights', 1:16);

%% classify new image
image = imread(imageToClassify);
[faceId, dist, idx] = eigenfaces_classify(efm, image, 'ShowDistances', true, 'ShowWeights', true);
fprintf('best match: image #%d; face #%d (distance: %f)\n', idx, faceId, dist);

%% perform manual classification using knnsearch
image = imread(imageToClassify);
weights = eigenfaces_weights(efm, image);
[idx, dist] = knnsearch(efm.weights, weights);
faceId = efm.class(idx);
fprintf('best match: image #%d; face #%d (distance: %f)\n', idx, faceId, dist);

%% flatten images and classes for validation
I_flat = eigenfaces_flatten(I);

% create proper input matrix for cross validation
% append classification as last column
data = [I_flat C'];

%% test the first 10 samples against the entire set
eigenfaces_test(data, data(1:10,:), 'ModelParams', {'EigenfacesLimit', 1:50});

%% test 10 random samples against the entire set
eigenfaces_test(data, datasample(data, 10), 'ModelParams', {'EigenfacesLimit', 'auto'});

%% test 10 random samples against 200 random training samples
eigenfaces_test(datasample(data, 200), datasample(data, 10))

%% use 70% random samples for training; test with the rest
f = 0.70; % 70% of the data used for training
n = size(data, 1);
idx = randperm(n);
xtrain = data(idx(1:f*n),:);    % training data
xtest = data(idx(f*n+1:end),:); % test data
eigenfaces_test(xtrain, xtest)

%% perform a 10-fold cross validation
result = crossval(@eigenfaces_test, data);
mean(result)

%% perform a 10-fold cross validation using the first 50 eigenvectors only
result = crossval(@(x,y) eigenfaces_test(x,y, 'ModelParams', { 'EigenfacesLimit', 1:50 } ), data);
mean(result)

%% perform a 10-fold cross validation using kNN (k=3)
result = crossval(@(x,y) eigenfaces_test(x,y, 'ClassifyParams', { 'K', 3 } ), data);
mean(result)

%% perform 10-fold cross validations for all models with number of eigenvectors between 1 and 50
result = zeros(50, 10);
for i = 1:50
    fprintf('10-fold cross validation using %d eigenfaces, starting with %d:\n', 50, i);
    result(i,:) = crossval(@(x,y) eigenfaces_test(x,y, 'ModelParams', { 'EigenfacesLimit', 1:50 } ), data);
    fprintf('mean classifcation rate: %f\n', mean(result(i,:)));
end
disp('done');