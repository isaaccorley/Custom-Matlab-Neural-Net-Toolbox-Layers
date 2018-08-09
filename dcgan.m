%% Load mnist
clearvars; clc;
imgShape = [28, 28, 1];
latentDim = 100;

trainFile = 'train-images.idx3-ubyte';
testFile = 't10k-images.idx3-ubyte';

imgs = cat(4, loadMNISTImages(trainFile), loadMNISTImages(testFile));
numImgs = size(imgs, 4);

%% G and D architectures
gModel = [ ...
        imageInputLayer([latentDim, 1, 1], 'Normalization', 'None', 'Name', 'g_input')
        fullyConnectedLayer(128*7*7, 'name', 'g_fc_1')
        leakyReluLayer('name', 'g_lrelu_1')
        reshapeLayer(128*7*7, [7, 7, 128], 'g_reshape_1')
        batchNormalizationLayer('name', 'g_bnorm_1')
        upsampling2dLayer([2, 2], 'g_upsample_1')
        convolution2dLayer(5, 128, 'Padding', 'same', 'name', 'g_conv_1')
        leakyReluLayer('name', 'g_lrelu_2')
        batchNormalizationLayer('name', 'g_bnorm_2')
        upsampling2dLayer([2, 2], 'g_upsample_2')
        convolution2dLayer(5, 64, 'Padding', 'same', 'name', 'g_conv_2')
        leakyReluLayer('name', 'g_lrelu_3')
        batchNormalizationLayer('name', 'g_bnorm_3')
        convolution2dLayer(5, 1, 'Padding', 'same', 'name', 'g_conv_3')
        sigmoidLayer('g_sigmoid_1')
        regressionLayer('name', 'g_out')];

dModel = [
        imageInputLayer(imgShape, 'Normalization', 'None', 'Name', 'd_input')
        convolution2dLayer(5, 128, 'Stride', 2, 'Padding', 'same', 'name', 'd_conv_1')
        leakyReluLayer('name', 'd_lrelu_1')
        batchNormalizationLayer('name', 'd_bnorm_1')
        convolution2dLayer(5, 128, 'Stride', 2, 'Padding', 'same', 'name', 'd_conv_2')
        leakyReluLayer('name', 'd_lrelu_2')
        batchNormalizationLayer('name', 'd_bnorm_2')
        fullyConnectedLayer(256, 'name', 'd_fc_1')
        leakyReluLayer('name', 'd_lrelu_3')
        batchNormalizationLayer('name', 'd_bnorm_3')
        fullyConnectedLayer(1, 'name', 'd_fc_2')
        regressionLayer('name', 'd_out')];
  
gModel = layerGraph(gModel);
dModel = layerGraph(dModel);

%% Freeze D layers before combining with G
dLayers = removeLayers(dModel, dModel.Layers(1).Name);
dLayers = makeNotTrainable(dLayers);

% Create combined model to update G using D
gLayers = removeLayers(gModel, gModel.Layers(end).Name);
combinedModel = addLayers(gLayers, dLayers.Layers);
combinedModel = connectLayers(combinedModel, gLayers.Layers(end).Name,  dLayers.Layers(1).Name);

%% Train ops
batchSize = 2;

noise = randn(latentDim, 1, 1, batchSize); 
realImgs = imgs(:, :, :, randi([0, numImgs], batchSize, 1));

options = trainingOptions(...
    'adam', ...
    'InitialLearnRate', 1E-100, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', batchSize, ...
    'Verbose', false, ...
    'Shuffle', 'never');

dModel = trainNetwork(realImgs, ones(batchSize,1), dModel, options);
combinedModel = trainNetwork(noise, ones(batchSize,1), combinedModel, options);
gModel = trainNetwork(noise, realImgs, gModel, options);

%% Train
batchSize = 128;
iters = 10000;

realLabel = ones(batchSize, 1);
fakeLabel = zeros(batchSize, 1);

options = trainingOptions(...
    'adam', ...
    'InitialLearnRate', 1E-3, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', batchSize, ...
    'Verbose', false, ...
    'Shuffle', 'never');

f = waitbar(0, '', 'Name','Training GAN');

%%
for iter = 1:iters
    
    % sample images at uniform randomly
    realImgs = imgs(:, :, :, randi([1, numImgs], batchSize, 1));
    
    % sample a latent vector
    noise = randn(latentDim, 1, 1, batchSize); 
    
    % generate fake imgages
    fakeImgs = predict(gModel, noise);
    
    % Update D on real
    [dModel, dLossReal] = trainNetwork(realImgs, realLabel, dModel, options);

    % Update D on fake
    [dModel, dLossFake] = trainNetwork(fakeImgs, fakeLabel, dModel, options);

    % Update D weights in G
    combinedModel = updateWeights(combinedModel, dModel);
    
    % Update G on 
    [combinedModel, gLoss] = trainNetwork(noise, realLabel, combinedModel, options);
    
    % Update G weights
    gModel = updateWeights(gModel, combinedModel);
    
    % Print loss to progressbar
    gLoss = gLoss.TrainingLoss;
    dLoss = (dLossReal.TrainingLoss + dLossFake.TrainingLoss) / 2;
    waitbar(iter/iters, f, sprintf('Iter: %d; G Loss: %.4f; D Loss: %.4f', iter, gLoss, dLoss));
    
    if mod(iter, 100) == 0
        plotImages(gModel, iter, latentDim)
    end
    
end

close(f)

%% Functions
function plotImages(model, iter, latentDim)
    noise = randn(latentDim, 1, 1, 4);
    y_pred = predict(model, noise);
    y_pred(y_pred > 1) = 1;
    y_pred(y_pred < 0) = 0;
    
    close all
    figure;
    for i = 1:size(y_pred, 4)
        subplot(2, 2, i);
        imshow(y_pred(:, :, :, i), [0, 1]);
    end
    
    print(strcat('images/', 'LSGAN_', 'Iter_', num2str(iter)), '-dpng');
end


function model = updateWeights(modelToUpdate, updatedModel)
    model = modelToUpdate.saveobj;

    % Search through layers
    for i = 1:numel(model.Layers)

        % Skip if layer has no learnable parameters
        if ~isprop(model.Layers(i), 'Weights')
            continue
        end

        % Search for the equivalent layer in updated model
        for j = 1:numel(updatedModel.Layers)

            % Once found copy the layer
            % skip layers with no learnable parameters
            if strcmp(model.Layers(i).Name, updatedModel.Layers(j).Name)
                model.Layers(i) = updatedModel.Layers(j);
                model.Layers(i).WeightLearnRateFactor = 0;
                model.Layers(i).WeightL2Factor = 0;
                model.Layers(i).BiasLearnRateFactor = 0;
                model.Layers(i).BiasL2Factor = 0;
            end
        end
    end
    
    model = modelToUpdate.loadobj(model);
end


function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);

    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);

    fclose(fp);

    images = reshape(images, numRows, numCols, 1, numImages);
    
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;

end


function graph = makeNotTrainable(graph)

    layers = graph.Layers;

    for i = 1:numel(layers)
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'WeightL2Factor')
            layers(i).WeightL2Factor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasL2Factor')
            layers(i).BiasL2Factor = 0;
        end
    end

    graph = layerGraph(layers);                     

end
