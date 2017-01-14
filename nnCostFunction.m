function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%   Feedforward the neural network and return the cost in the
%   variable J. 
%

X = [ones(size(X, 1) ,1) X];
z2 = Theta1*X';
Z2 = sigmoid(z2');
Z2 = [ones(size(Z2, 1) ,1) Z2];
z3 = Theta2*Z2';
Z3 = sigmoid(z3');


YPractical = zeros(size(Z3));
for i =1:m
YPractical(i,y(i)) = 1;
end


%   Implement the backpropagation algorithm to compute the gradients
%   Theta1_grad and Theta2_grad.
%

for i = 1:m
J = J + (-1)*(log(Z3(i,:))*YPractical(i,:)' + log(1-Z3(i,:))*(1-YPractical(i,:)'));
end
J = (1/m)*J + (lambda/(2*m))*(sum((sum(Theta1(:, 2:end).^2))) + sum((sum(Theta2(:, 2:end).^2))));

del3 = Z3 - YPractical;
del2 = (del3*Theta2(:, 2:end)).*sigmoidGradient(z2');

Theta1_grad = del2'*X/m + (lambda/m)*Theta1;
Theta2_grad = del3'*Z2/m + (lambda/m)*Theta2;

Theta1_grad(:, 1) = del2'*X(:,1)/m;
Theta2_grad(:, 1) = del3'*Z2(:,1)/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
