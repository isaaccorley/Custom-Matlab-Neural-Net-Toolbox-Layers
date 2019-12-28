classdef eluLayer < nnet.layer.Layer

    properties
        alpha
    end
    
    methods
        function layer = eluLayer(alpha, name)
            % Set layer name
            if nargin > 1
                layer.Name = name;
            end
            
            layer.alpha = alpha;
            
            layer.Description = 'Exponential Linear Unit (ELU) Activation Function Layer';
        end        

        function Z = predict(layer,X)
            Z = (X .* (X > 0)) + (layer.alpha .* (exp(min(X,0)) - 1) .* (X <= 0));
        end

        function [dLdX] = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* ((X > 0) + ((layer.alpha + Z) .* (X <= 0)));            
        end
    end
end

