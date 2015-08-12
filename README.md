# Eigenfaces
MATLAB Eigenfaces Toolbox

This MATLAB Toolbox implements the 'Eigenface' algorithm for feature extraction and classification of faces.
For details see http://en.wikipedia.org/wiki/Eigenface.

This MATLAB scripts are the results of a project for the lecture course 'Data Fusion in Sensor Systems' of the study program 'Human-Centered Computing' at the University of Applied Sciences Upper Austria (http://www-en.fh-ooe.at).

Project members: Christoph Absenger, Christian Salomon, Mario Winterer.

## Usage

See [eigenfaces.m](eigenfaces.m) for examples of how to load images, train a model and use it for classification. It also contains several sections for validation.

For testing we suggest [The Database of Faces (formerly 'ORL')](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

### Loading images

The function [`eigenfaces_load(path, ext)`](eigenfaces_load.m) loads all images with extension `ext` from all subfolders of a folder specified by `path`. Every subfolder is treated as separate class (= face). So the image folder structured should be as follows:

````
path
+-- face1           (class 1)
|   +-- img1.ext    (image 1 / class 1)
|   +-- img2.ext    (image 2 / class 1)
|   \-- img3.ext    (image 3 / class 1)
|
+-- face2           (class 2)
|   +-- img1.ext    (image 4 / class 2)
|   +-- img2.ext    (image 5 / class 2)
|   \-- img3.ext    (image 6 / class 2)
...
````

(Note that the names of subfolders and files do not matter, but all images must have exactly the same size).

The result of calling `eigenfaces_load(...)` is a _h x w x n_ matrix (_h_ = height of an image, _w_ = width of an image, _n_ = number of images) as well as a _1 x n_ vector containing the classes (target values) of all faces as numbers. These classes (target values) are numbered in ascending order according to the order in which the faces-subfolders are listed:

````matlab
% load images and their classification
[I, C] = eigenfaces_load('<path>', '*.jpg');

% get 5th image
img5 = I(:,:,5);

% get class (=face) of 5th image
class5 = C(5)
````

This results can be used directly for training the eigenface model using [`eigenfaces_model(I, C)`](eigenfaces_model.m).

#### Alternative data format (if required)

The function [`eigenfaces_flatten(I)`](eigenfaces_flatten.m) rearranges the image matrix into a matrix of size _n x (h*w)_ as follows:

* Every row contains one observation (= image)
* The image-data is flattened into a one dimension array (row)

The results can easily be combined into one big matrix to be used in MATLAB functions like `datasample` or `crossval`:
 
 ````matlab
 % flatten images
 I_flat = eigenfaces_flatten(I);
 
 % combine into big matrix
 data = [I_flat C'];
 
 ````

### Train Eigenface model

In order to perform classifications, it is required to train an Eigenface model using [`eigenfaces_model(I, C)`](eigenfaces_model.m). The result is a _1x1_ structure array containing all required information about the trainied model (but not the original images themselves):

````matlab
efm = eigenfaces_model(I, C);
disp(efm);
% result:
       meanface: [1x10304 double]
     eigenfaces: [10304x400 double]
    eigenvalues: [400x1 double]
        weights: [400x400 double]
          class: [1x400 double]
````

| field       | format    | description               |
|-------------|-----------|---------------------------|
| meanface    | 1 x (h*w) | mean face                 |
| eigenfaces  | (h*w) x n | eigenvectors              |
| eigenvalues | n x 1     | eigenvalues               |
| weights     | n x M     | weights of trained images |
| class       | 1 x n     | class of each image       |

### Classification

To classify an image, either use [`eigenfaces_classify(efm, image)`](eigenfaces_classify.m) directly, or compute the weights of the image to classify via [`eigenfaces_weights(efm, image)`](eigenfaces_weights.m) first and use any classifier (e.g. kNN) then to find the best match of the resulting weights within `efm.weights` - the weights of the trained eigenface model.

Internally, `eigenfaces_classify` uses kNN(k=1, euclidean distance) to find the best match. The results are the index and the class (target value) of the best match:

````matlab
[face, dist, idx] = eigenfaces_classify(efm, image);
fprintf('best match: image #%d; face #%d (distance: %f)\n', idx, faceId, dist);
% result
  best match: image #68 face #33 (distance: 834.564822)
````

### Validation

The function [`eigenfaces_test(xtrain, xtest)`](eigenfaces_test.m) is thought for being used by different validation algorithms like `crossval`. It uses the given training data set `xtrain` to train an Eigenface model (using `eigenface_model`) and then tries to classify the given test set `xtest`, returning the classification rate (number of correctly classified samples / total number of samples).

The two data sets `xtrain` and `xtest` must be _n x (h x w + 1)_ matrices, where every row contains the image data of an observation and the classification (= face id) of that image in the last column. This structure can easily be established by using the alternative date format ([`eigenfaces_flatten(I)`](eigenfaces_flatten.m)):

````matlab
I_flat = eigenfaces_flatten(I); % flatten images
data = [I_flat C'];              % append class column
````

Using this test data, it is now easy to perform a 10-fold cross-validation by calling `crossval`:

````matlab
result = crossval(@eigenfaces_test, data);
disp(mean(result));
````

### Visualization

For visualization, some functions accept optional parameters:

````matlab
%% create model and display Eigenfaces and weights
efm = eigenfaces_model(I, C, 'Show', { 'Eigenfaces', true, 'Weights', true});

%% classify image and show distances to all training faces and weights of input image
[face, dist, idx] = eigenfaces_classify(efm, image, 'ShowDistances', true, 'ShowWeights', true);
````

See the two [examples](examples) for more info.

## Limitations

* All images must be grayscale.
* All images must have equal size.
* All images are loaded into memory at once, so this toolbox may not be suitable for large data sets.

