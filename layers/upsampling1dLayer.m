classdef upsampling1dLayer < nnet.layer.Layer

    properties
        upSize
    end
   
    methods
        function layer = upsampling1dLayer(upSize, name)
            % Layer constructor function goes here
            if nargin < 1
                error('upSize must be defined.');
            end
            
            upSize = num2cell(upSize);
            layer.upSize = upSize;
            
            if nargin > 1
                layer.Name = name;
            end
            
            layer.Description = 'UpSampling1D Layer';
        end
        
        function Z = predict(layer, X)
            Z = repelem(X, layer.upSize);
        end

        function dLdX = backward(layer, ~, ~, dLdZ, ~)
            dLdX = dLdZ(1:layer.upSize:end, :, :);
        end
    end
end

