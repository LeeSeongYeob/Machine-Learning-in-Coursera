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
% X = 5000x400
% Theta1 = 25 x 401
% Theta2 = 10 x 26
% y = 5000 x 1
%display(Theta1(:,end))
X = [ones(m , 1), X];  % 5000 x 401
z2 = X * Theta1';   % 5000x26
a2 = sigmoid(z2);

a2 = [ones(m , 1), a2];
z3 = a2 * Theta2'; % 5000x10
hypo = sigmoid(z3); % 5000x10


%y_new = zeros(m,num_labels);
y_new  = (1:num_labels) == y;  % y_new 5000x10 y의 label값과 동일시, 1로 만듬
%disp(y_new([1:20],:)) %20행까지만 결과 보기 디버그용
%cost 함수
J = (1/m)*sum(sum((-y_new.*log(hypo)) - ((1-y_new).*log(1-hypo)))); % scalar 결과값

%----------------------------------------------------------------------------
%cost 함수에 Regularize 추가
% bias의 값은 regularization에 포함하지 않아야 함.
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);

reg = (lambda/(2*m))*(sum(sum(t1.^2)) + sum(sum(t2.^2)));
J = J + reg;

%-----------------------------------------------------------------------------
%역전파 알고리즘 실행하기
X2 = X;  % 5000x 401
Z2 = X2 * Theta1';   % 5000x25
A2 = sigmoid(z2); 

A2 = [ones(m , 1), A2];
Z3 = A2 * Theta2'; % 5000x10
hypo_bp = sigmoid(Z3); % 5000x10

y_new  = (1:num_labels) == y; 

delta3 = hypo_bp - y_new; %5000x10
Z2 = [ones(size(Z2),1),Z2]; 

delta2 = (delta3 * Theta2) .* sigmoidGradient(Z2); %5000x26
delta2 = delta2(:,2:end);

Theta1_grad = (1/m) * (delta2' * X2); % 25x401
Theta2_grad = (1/m) * (delta3' * A2); % 10x26


%----------------------------------------------------------------------------------
% Regularized NN

reg_theta1 = Theta1(:,2:end);
reg_theta2 = Theta2(:,2:end);
reg1 = (lambda/m) * reg_theta1;
reg1 = [zeros(size(Theta1,1),1),reg1];
%size(reg1)
Theta1_grad = Theta1_grad + reg1; %25 x 401
%Theta1_grad = [ones(size(Theta1_grad,1),1),Theta1_grad];

reg2 = (lambda/m) * reg_theta2;
reg2 = [zeros(size(reg2,1),1),reg2];
%reg2 = [Theta2(:,1),reg2];
Theta2_grad = Theta2_grad + reg2;
%Theta2_grad = [ones(size(Theta2_grad,1),1),Theta2_grad];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
