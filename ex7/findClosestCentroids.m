function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
%size(centroids)  % 3 x 2
% K 의 값은 3
%size(X(1,:))
for i = 1:size(X,1)
  tmp = zeros(1,K);  % x값에서 K값을 각각 뺀 거리 담을 리스트트
  for j = 1:K
    tmp(1,j) = sqrt(sum((X(i,:) .- centroids(j,:)).^2)); % 두 점사이의 거리리
  end
  [val,index] = min(tmp);  % tmp리스트에서 최소인 값과 index 리턴턴
  idx(i) = index;
end






% =============================================================

end

