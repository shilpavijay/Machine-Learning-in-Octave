function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
num = (-y' * log(h)) - ((1-y)' * log(1-h));
num = num * (1/m);
reg = (1/m) * (1/2) * (lambda) * (sum(theta([2:size(theta)],:).^2));
J = num + reg;

GD = X' * (h - y);
init = (1/m)*(GD([2:size(GD)],:));
regulr = (1/m) * (lambda) * theta([2:size(theta)],:);
grad(1) = (1/m)*(GD(1));
grad([2:size(grad)],:) = init + regulr;


% =============================================================

end
