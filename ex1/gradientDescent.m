function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    
%     tem = X*theta-y;
%     tem1 = sum(tem.*X(:,1));
%     delta_theta1 = alpha/m*tem1;
%     
%     tem2 = sum(tem.*X(:,2));
%     delta_theta2 = alpha/m*tem2;
%     
%     delta_theta = [delta_theta1;delta_theta2];
%     
%     theta = theta-delta_theta;

      theta = theta - alpha/m*(X'*(X*theta-y));
      % X*theta calculuates the output of hypothesis in each training example, this is a (n+1)*1 matrix
      % i.e., h_{theta}(x)
      % (X*theta-y) corresponds to the h_{theta}(x)-y
      % X'
      


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
