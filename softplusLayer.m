classdef softplusLayer < nnet.layer.Layer

    methods
        function layer = softplusLayer(name)
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            
            layer.Description = 'Softplus Activation Function Layer';
        end
        
        function Z = predict(layer, X)
            Z = log(exp(X) + 1);
        end

        function dLdX = backward(layer, X, Z, dLdZ, ~)
            dLdX = dLdZ .* (1 ./ (1 + exp(-X)));
        end
    end
end

