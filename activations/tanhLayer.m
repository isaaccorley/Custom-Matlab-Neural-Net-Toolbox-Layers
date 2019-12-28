classdef tanhLayer < nnet.layer.Layer

    methods
        function layer = tanhLayer(name)
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            
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

