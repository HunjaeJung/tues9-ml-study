function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
%     sum_for_theta_1 = 0;
%     sum_for_theta_2 = 0;
%     for i=1:size(X,1)
%         sum_for_theta_1 = sum_for_theta_1 + ( X(i,:)*theta - y(i,:) ) ;
%         sum_for_theta_2 = sum_for_theta_2 + ( X(i,:)*theta - y(i,:) ) * X(i,2);
%     end

    temp_theta = theta % to update simultaneously
    theta(1) = theta(1) - alpha/m * (sum(X*temp_theta -y));
    theta(2) = theta(2) - alpha/m * (sum((X*temp_theta -y).*X(:,2)));
    
%     theta(1) = theta(1) - alpha/m * (sum_for_theta_1);
%     theta(2) = theta(2) - alpha/m * (sum_for_theta_2);
    % ============================================================
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
%     if iter > 1 && (J_history(iter-1) - J_history(iter)) < 0.001
%         disp('Hello Cost function!')
%         disp(iter)
%         disp(J_history(iter-1))
%         disp(J_history(iter))
%         break
%     end

end

end
