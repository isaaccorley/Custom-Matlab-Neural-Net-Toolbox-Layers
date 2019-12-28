classdef sigmoidLayer < nnet.layer.Layer

    methods
        function layer = sigmoidLayer(name)
             % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            
            layer.Description = 'Sigmoid Activation Function Layer';
        end
        
        function Z = predict(layer, X)
            Z = 1 ./ (1 + exp(-X));
        end

        function dLdX = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* (Z .* (1 - Z));
        end
    end
end

