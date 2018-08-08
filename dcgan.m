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
layers = removeLayers(dModel, 'd_input');
layers = makeNotTrainable(layers);

% Create full generator model
combinedModel = addLayers(layers, gModel.Layers);
combinedModel = connectLayers(combinedModel, 'g_out', 'd_conv_1');

%% Get layer names (makes for easier updating)
dLayers = {};

for i = 2:numel(dModel.Layers) % skip the input layer
    dLayers{i-1} = dModel.Layers(i).Name;
end

clear layers

%%



%% Train ops
batchSize = 32;
iters = 1000;

realLabel = categorical(ones(1, 1));
fakeLabel = categorical(zeros(1, 1));
gOutMask = zeros(28, 28, 1, 1);
options = trainingOptions(...
    'adam', ...
    'InitialLearnRate', 1E-100, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'never');

%% Update D, G, and combined once to initialize weights and create DAGNetwork objects
% latent vector
noise = randn(latentDim, 1, 1, batchSize); 

% random sample of images
realImgs = imgs(:, :, :, randi([0, numImgs], batchSize, 1));

dModel = trainNetwork(realImgs, realLabel, dModel, options);
combinedModel = trainNetwork(noise, realLabel, combinedModel, options);

%% Create gModel to output images
gModel = removeLayers(combinedModel, dLayers);


%% Train
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
    gModel = update_D_weights(dModel, gModel);
    
    % Update G on 
    gUpdate = trainNetwork(noise, realLabel, combinedModel, options);
    
    % Update G weights
    gModel = update_G_weights(combinedModel, gModel);
    
end

%% Functions
function gModel = update_G_weights(combinedModel, gModel)
    model = gModel.saveobj;

    % Search through layers in G
    for i = 1:numel(model.Layers)

        % Skip if layer has no learnable parameters
        if ~isprop(model.Layers(i), 'Weights')
            continue
        end

        % Search for the equivalent layer in combined
        for j = 1:numel(combinedModel.Layers)

            % Once found link the layer weights and biases
            % skip layers with no learnable parameters
            if model.Layers(i).Name == combinedModel.Layers(j).Name
                model.Layers(i).Weights = combinedModel.Layers(j).Weights;
                model.Layers(i).Bias = combinedModel.Layers(j).Bias;
            end        
        end
    end
    
    gModel = gModel.loadobj(model);
end

function combinedModel = update_D_weights(dModel, combinedModel)
    model = combinedModel.saveobj;
    
    % Search through layers in G
    for i = 1:numel(model.Layers)

        % Skip if layer has no learnable parameters
        if ~isprop(model.Layers(i), 'Weights')
            continue
        end

        % Search for the equivalent layer in D
        for j = 1:numel(model.Layers)

            % Once found link the layer weights and biases
            % skip layers with no learnable parameters
            if model.Layers(i).Name == dModel.Layers(j).Name
                model.Layers(i).Weights = dModel.Layers(j).Weights;
                model.Layers(i).Bias = dModel.Layers(j).Bias;
            end        
        end
    end
    
    combinedModel = combinedModel.loadobj(model);
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
        transposedConv2dLayer([5 5], 512, 'name', 'g_tconv_1')
        reluLayer('name', 'g_relu_2')
        transposedConv2dLayer([5 5], 256, 'name', 'g_tconv_2')
        reluLayer('name', 'g_relu_3')
        transposedConv2dLayer([5 5], 1, 'name', 'g_out')];
    
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
