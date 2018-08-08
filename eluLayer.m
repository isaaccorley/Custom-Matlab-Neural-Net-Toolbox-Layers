classdef eluLayer < nnet.layer.Layer

    properties
        alpha = 1.0;
    end
    
    methods
        function layer = eluLayer()
            layer.Description = 'Exponential Linear Unit (ELU) Activation Function Layer';
        end        

        function Z = predict(layer,X)
            Z = (X .* (X > 0)) + (layer.alpha.*(exp(min(X,0)) - 1) .* (X <= 0));
        end

        function [dLdX] = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* ((X > 0) + ((layer.alpha + Z) .* (X <= 0)));            
        end
    end
end

