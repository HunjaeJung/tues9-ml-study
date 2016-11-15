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
lambda_m = zeros(size(theta));
lambda_m(:)=lambda; lambda_m(1)=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sig_ret = sigmoid(X*theta)

J = sum((-y.*log(sig_ret) - (1-y).*log(1-sig_ret))/m) + sum((lambda_m / (2*m)).*(theta.^2))

grad = (X'*(sig_ret - y))/m + (lambda_m / m) .*theta

% =============================================================

end
