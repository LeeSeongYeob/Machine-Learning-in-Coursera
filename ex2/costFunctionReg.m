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
hypo = sigmoid(X*theta);
reg = (lambda/(2*m)) * sum(theta(2:end).^2);  % 첫번째 세타 값은 포함x bias
J = (1/m)*sum(-y.*log(hypo) -((1-y).*log(1-hypo))) + reg;


grad(1) = (1/m) * (X(:,1)'*(hypo-y)); % X0 번째 값
grad(2:end) = (1/m) * (X(:,2:end)' * (hypo-y)) + (lambda / m * theta(2:end));





% =============================================================

end
