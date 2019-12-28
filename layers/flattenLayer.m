classdef flattenLayer < nnet.layer.Layer

    methods
        function layer = flattenLayer(name)
            
            if nargin == 1
                layer.Name = name;
            end
            
            layer.Description = 'Flatten Layer';
        end
        
        function Z = predict(layer, X)
            shape = size(X);
            Z = reshape(X, shape(end));
        end

        function dLdX = backward(~, ~, ~, dLdZ, ~)
            dLdX = reshape(dLdZ, size(X));
        end
    end
end

