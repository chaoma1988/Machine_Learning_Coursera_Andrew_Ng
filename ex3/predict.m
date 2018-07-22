function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% if there is more than one layers, we can use a loop to implemented the update,
% with input of every Thetas, and first A
X = [ones(m, 1) X];

Z2 = X*Theta1'; % 5000 * 25

A2 = sigmoid(Z2); % 5000 * 25

A2 = [ones(m,1) A2]; % (5000 * 26)

Z3 = A2*Theta2'; % 5000 * 10

A3 = sigmoid(Z3); 

[dummy,p] = max(A3,[],2);






% =========================================================================


end
