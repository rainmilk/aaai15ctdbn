function minibatch = genMiniBatch(norder, nframes, batchsize)
% genMiniBatch
% norder    -- order of CRBM
% batchsize = 100; %size of mini-batches

if nargin < 3, batchsize = 100; end
batchdataindex = norder+1:nframes;
%now that we know all the valid starting frames, we can randomly permute
%the order, such that we have a balanced training set
permindex = batchdataindex(randperm(length(batchdataindex)));

%fit all minibatches of size batchsize
batches = floor(length(permindex)/batchsize);
minibatchindex = reshape(permindex(1:batchsize*batches),batches,batchsize);
%Not all minibatches will be the same length ...
%must use a cell array (the last batch is a different size)
minibatch = num2cell(minibatchindex,2);
%tack on the leftover frames (smaller last batch)
leftover = permindex(batchsize*batches+1:end);
minibatch = [minibatch;num2cell(leftover,2)];
end