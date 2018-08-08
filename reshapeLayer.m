classdef reshapeLayer < nnet.layer.Layer

    properties
        inputShape
        outputShape
    end
   
    methods
        function layer = reshapeLayer(inputShape, outputShape)
            % Layer constructor function goes here
            if nargin < 2
                error('inputShape and outputShape must be defined.');
            end
            
            layer.inputShape = inputShape;
            layer.outputShape = outputShape;
            layer.Description = ...
                ['RehapeLayer from shape [', num2str(inputShape), '] to shape[', ...
                num2str(outputShape), ']'];
        end
        
        function Z = predict(layer, X)
            % Get batch size (if any)
            if isequal(ndims(X), 4)
                batchSize = size(X, 4);
            else
                batchSize = 1;
            end
            
            % Reshape inputs to outputShape
            Z = reshape(X, [layer.outputShape, batchSize]);
            %disp(size(X))
            %disp(size(Z))
        end

        function dLdX = backward(~, X, ~, dLdZ, ~)
           
            % Reshape gradients to inputShape
            disp(size(dLdZ))
            dLdX = reshape(dLdZ, size(X));
            disp(size(dLdX))
        end
    end
end
