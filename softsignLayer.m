classdef softsignLayer < nnet.layer.Layer

    methods
        function layer = softsignLayer()
            layer.Description = 'Softsign Activation Function Layer';
        end
        
        function Z = predict(layer, X)
            Z = X ./ (1 + abs(X));
        end

        function dLdX = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* (1 ./ (1 + abs(X)).^2);
        end
    end
end

