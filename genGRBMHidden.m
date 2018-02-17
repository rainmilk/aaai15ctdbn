function [ output ] = genGRBMHidden( batchdata, nt, W, B, bj, gsd )
%GENUPDATA Summary of this function goes here
batchdataindex = nt+1:size(batchdata,1); 
numcases = length(batchdataindex);
numhid = size(bj,1);
%Calculate contributions from directed visible-to-hidden connections
bjstar = zeros(numhid,numcases);
for hh = 1:nt
  bjstar = bjstar + B(:,:,hh)*batchdata(batchdataindex-hh,:)';
end

%Calculate "posterior" probability -- hidden state being on
%Note that it isn't a true posterior
bottomup = W*(batchdata(batchdataindex,:)./gsd)';

eta =  bottomup + ...                  %bottom-up connections
  repmat(bj, 1, numcases) + ...        %static biases on unit
  bjstar;                              %dynamic biases

output = 1./(1 + exp(-eta'));   %logistic

end

