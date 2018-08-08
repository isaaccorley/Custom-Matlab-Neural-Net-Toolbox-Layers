classdef tanhLayer < nnet.layer.Layer

    methods
        function layer = tanhLayer()
            layer.Description = 'Hyperbolic Tangent (Tanh) Activation Function Layer';
        end
        
        function Z = predict(layer, X)
            Z = (exp(X) - exp(-X)) ./ (exp(X) + exp(-X));
        end

        function dLdX = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* (1 - Z.^2);
        end
    end
end

