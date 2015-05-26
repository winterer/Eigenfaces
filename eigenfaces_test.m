%EIGENFACES_TEST function for testing the method. Can be passed to crossval
%   xtrain ... the training subset - used to build the eigenfaces model.
%              Structure: n*(m+1) matrix
%                  n...number of observations
%                  m...flat image data
%              last column contains the class of the observation
%   ytrain ... the test subset - used to test the trained model
%              Structure: n*(m+1) matrix
%                  n...number of observations
%                  m...flat image data
%              last column contains the class of the observation
%   returns the classification (success) rate: correct / total number.

function [ value ] = eigenfaces_test( xtrain, xtest, varargin )

p = inputParser;

addRequired(p, 'xtrain');
addRequired(p, 'xtest');
addParameter(p, 'ModelParams', {});
addParameter(p, 'ClassifyParams', {});

parse(p, xtrain, xtest, varargin{:});

% last column is class
train_img = xtrain(:,1:end-1);
train_class = xtrain(:,end:end);

% last column is class
test_img = xtest(:,1:end-1);
test_class = xtest(:,end:end);

% train model
efm = eigenfaces_model(train_img', train_class', p.Results.ModelParams{:} );

tests = size(test_img, 1);
hits = 0;
for i = 1:tests
    % classify i-ith image from test set
    [ face, dist, nn ] = eigenfaces_classify( efm, test_img(i,:), p.Results.ClassifyParams{:} );
    best_face = mode(face);
    if (best_face == test_class(i))
        %fprintf('success (face %d)\n', face);
        hits = hits + 1;
    else
        %fprintf('failed (classified: %d, correct: %d)\n', best_face, test_class(i));
    end
end

value = hits / tests;
fprintf('classification rate: %f\n', value);
end

