function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly

centroids = zeros(K, size(X, 2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% centroids를 무작위 example 로 시작함

% exaple을 랜덤으로 섞음
randidx = randperm(size(X, 1));
% ++ rannperm은 1~n 까지의 난수 순열을 만들어 주는 함수 입니다.!
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);


% =============================================================

end

