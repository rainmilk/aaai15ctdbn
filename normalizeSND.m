function [ batchdata, means, stds ] = normalizeSND( batchdata )
%NORMALIZESND Summary of this function goes here
% make it zero-mean
means = mean(batchdata, 1);
batchdata = bsxfun(@minus, batchdata, means);

% make it unit-variance
stds = std(batchdata, [], 1);
batchdata = bsxfun(@rdivide, batchdata, stds);
end

