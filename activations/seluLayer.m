classdef seluLayer < nnet.layer.Layer

    properties
        lambda = 1.0507
        alpha = 1.67326
    end
    
    methods
        function layer = seluLayer(name)
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            
            layer.Description = 'Scaled Exponential Linear Unit (SELU) Activation Function Layer';
        end        

        function Z = predict(layer,X)
            Z = layer.lambda .* (X .* (X > 0)) + (layer.alpha.*(exp(min(X,0)) - 1) .* (X <= 0));
        end

        function [dLdX] = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* layer.lambda .* ((X > 0) + ((layer.alpha + Z) .* (X <= 0)));            
        end
    end
end

