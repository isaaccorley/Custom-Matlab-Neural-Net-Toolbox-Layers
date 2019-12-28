classdef upsampling2dLayer < nnet.layer.Layer

    properties
        rows
        cols
    end
   
    methods
        function layer = upsampling2dLayer(upSize, name)
            % Layer constructor function goes here
            if nargin < 1
                error('upSize must be defined.');
            end
            
            upSize = num2cell(upSize);
            [layer.rows, layer.cols] = upSize{:};
            
            if nargin > 1
                layer.Name = name;
            end
            
            layer.Description = 'UpSampling2D Layer';
        end
        
        function Z = predict(layer, X)
            Z = repelem(X, layer.rows, layer.cols);
        end

        function dLdX = backward(layer, ~, ~, dLdZ, ~)
            dLdX = dLdZ(1:layer.rows:end, 1:layer.cols:end, :, :);
        end
    end
end

