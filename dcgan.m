%% Load mnist
clearvars; clc;
imgShape = [28, 28, 1];
latentDim = 100;

imgs = readMNIST();
numImgs = size(imgs, 4);

% Normalize images to [0, 1]
imgs = imgs ./ 255;

%% G and D architectures
gModel = layerGraph(createG(latentDim));
dModel = layerGraph(createD(imgShape));

% Freeze D layers before combining with G
dLayers = removeLayers(dModel, 'd_input');
dLayers = makeNotTrainable(dLayers);

% Create combined model to update G using D
gLayers = removeLayers(gModel, 'g_out');
combinedModel = addLayers(dLayers, gLayers.Layers);
combinedModel = connectLayers(combinedModel, 'g_conv_1', 'd_conv_1');

%% Train ops
batchSize = 2;

realLabel = categorical(ones(batchSize, 1));
noise = randn(latentDim, 1, 1, batchSize); 
realImgs = imgs(:, :, :, randi([0, numImgs], batchSize, 1));
gOutMask = zeros(28, 28, 1, batchSize);

options = trainingOptions(...
    'adam', ...
    'InitialLearnRate', 1E-100, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'never');

dModel = trainNetwork(realImgs, realLabel, dModel, options);
combinedModel = trainNetwork(noise, realLabel, combinedModel, options);
gModel = trainNetwork(noise, gOutMask, gModel, options);

%% Train
batchSize = 32;
iters = 1000;

realLabel = categorical(ones(batchSize, 1));
fakeLabel = categorical(zeros(batchSize, 1));

options = trainingOptions(...
    'adam', ...
    'InitialLearnRate', 1E-4, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'never');

for i = 1:iters
    
    % sample images at uniform randomly
    realImgs = imgs(:, :, :, randi([0, numImgs], batchSize, 1));
    
    % sample a latent vector
    noise = randn(latentDim, 1, 1, batchSize); 
    
    % generate fake imgages
    fakeImgs = predict(gModel, noise);
    
    % Update D on real
    dModel = trainNetwork(realImgs, realLabel, dModel, options);

    % Update D on fake
    dModel = trainNetwork(fakeImgs, fakeLabel, dModel, options);

    % Update D weights in G
    gModel = updateWeights(gModel, dModel);
    
    % Update G on 
    gUpdate = trainNetwork(noise, realLabel, combinedModel, options);
    
    % Update G weights
    gModel = updateWeights(gModel, combinedModel);
    
    if mod(iter, 100) == 0
        plotImages(gModel, iter)
    end
    
end

%% Functions
function plotImages(model, iter)
    noise = randn(latentDim, 1, 1, 4);
    y_pred = predict(model, noise);
    
    for i = 1:size(y_pred, 4)
        subplot(4,5,i);
        imshow(squeeze(y_pred(:, :, :, i)));
        title(num2str(iter));
    end

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

            % Once found link the layer weights and biases
            % skip layers with no learnable parameters
            if model.Layers(i).Name == updatedModel.Layers(j).Name
                model.Layers(i).Weights = updatedModel.Layers(j).Weights;
                model.Layers(i).Bias = updatedModel.Layers(j).Bias;
            end        
        end
    end
    
    model = modelToUpdate.loadobj(model);
end


function imgs = readMNIST()
    % Read MNIST files from Kaggle (skip headers)
    trainImages = csvread('train.csv', 1, 1); % don't read labels column
    testImages = csvread('test.csv', 1, 0);
    imgs = [trainImages; testImages];
    imgs = permute(imgs, [2, 1]);
    imgs = reshape(imgs, 28, 28, 1, []);
end


function layers = createD(imgShape)

    layers = [
        imageInputLayer(imgShape, 'Normalization', 'None', 'Name', 'd_input')
        convolution2dLayer([5 5],128, 'Stride', [2, 2], 'Padding', 'same', 'name', 'd_conv_1')
        leakyReluLayer('name', 'd_lrelu_1')
        convolution2dLayer([5 5], 128, 'Stride', [2, 2], 'Padding', 'same', 'name', 'd_conv_2')
        leakyReluLayer('name', 'd_lrelu_2')
        fullyConnectedLayer(1, 'name', 'd_fc_1')
        softmaxLayer('name', 'd_softmax_1')
        classificationLayer('name', 'd_out')];
    
end


function layers = createG(latentDim)

    layers = [ ...
        imageInputLayer([latentDim, 1, 1], 'Normalization', 'None', 'Name', 'g_input')
        fullyConnectedLayer(128*7*7, 'name', 'g_fc_1')
        reluLayer('name', 'g_relu_1')
        reshapeLayer(128*7*7, [7, 7, 128], 'g_reshape_1')
        upsampling2dLayer([2, 2], 'g_upsample_1')
        convolution2dLayer([5 5], 128, 'Padding', 'same', 'name', 'g_conv_1')
        reluLayer('name', 'g_relu_2')
        upsampling2dLayer([2, 2], 'g_upsample_2')
        convolution2dLayer([5 5], 64, 'Padding', 'same', 'name', 'g_conv_3')
        reluLayer('name', 'g_relu_3')
        convolution2dLayer([5 5], 1, 'Padding', 'same', 'name', 'g_conv_3')
        regressionLayer('name', 'g_out')];
    
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
