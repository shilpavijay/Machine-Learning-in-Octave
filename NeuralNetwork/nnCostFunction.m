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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%computing h:
x = [ones(m, 1) X];
z2 = sigmoid(x * Theta1');
z3 = [ones(m, 1) z2];
h = sigmoid(z3 * Theta2');

I = eye(num_labels);
for i=1:m,
  Y(i, :)= I(y(i), :);
endfor;

%Regularization:
theta_one = Theta1(:,2:input_layer_size+1);
theta_two = Theta2(:,2:hidden_layer_size+1);

Reg = lambda * (1/m) * (1/2) * (sum(sum(theta_one.^2)) + sum(sum(theta_two.^2)));


%computing J (Cost):
J = (1/m)*sum(sum((-Y .* log(h)) - ((1-Y) .* log(1-h)))) + Reg;

%-------------------------------------------------------------

%Implementing Backpropogation Algorithm here:

%Step 1:
x_0 = X;
x_0 = [ones(m, 1) x_0];
a_1 = x_0;
z_2= a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);


%Step 2:
%Y1 = zeros(5000,10);
%for u = 1:m,
%  Y1(u,y(u)) = 1;
%endfor;

delta_3 = a_3 - Y;

%Step 3:
delta_2 = (delta_3 * Theta2) .* sigmoidGradient([ones(m,1) z_2]);


%Step 4:

sumdelta2 = delta_3' * a_2;
sumdelta1 = delta_2(:,2:end)' * a_1;  

%step 5:
%Regularization:
theta_1 = [zeros(size(theta_one,1), 1) theta_one];
theta_2 = [zeros(size(theta_two,1), 1) theta_two];
regpar = lambda * (1/m); 
Theta1_grad = ((1/m) * sumdelta1) + (regpar * theta_1);
Theta2_grad = ((1/m) * sumdelta2) + (regpar * theta_2);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
