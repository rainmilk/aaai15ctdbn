% Version 1.01 
%
% Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This is the "main" demo
% It trains two CRBM models, one on top of the other, and then
% demonstrates data generation

clear; close all;
more off;   %turn off paging

%initialize RAND,RANDN to a different state
rand('state',sum(100*clock))
randn('state',sum(100*clock))

%Our important Motion routines are in a subdirectory
addpath('./Motion')

%Load the supplied training data
%Motion is a cell array containing 3 sequences of walking motion (120fps)
%skel is struct array which describes the person's joint hierarchy
load Data/data.mat

%Downsample (to 30 fps) simply by dropping frames
%We really should low-pass-filter before dropping frames
%See Matlab's DECIMATE function
dropRate=4;
dropframes;

fprintf(1,'Preprocessing data \n');

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%how-many timesteps do we look back for directed connections
%this is what we call the "order" of the model 
n1 = 3; %first layer
n2 = 3; %second layer
        
%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2
numdims = size(batchdata,2); %data (visible) dimension
nModel = 1;

%save some frames of our pre-processed data for later
%we need an initialization to generate 
initdata = batchdata(1:100,:);
initdata = repmat(initdata,[1,1,nModel]);

%Set network properties
numhid1 = 150; numhid2 = 150; numepochs=50;
gsd=1;          %fixed standard deviation for Gaussian units
nt = n1;        %crbm "order"
numhid=numhid1;

fprintf(1,'Training Layer 1 CRBM, order %d: %d-%d \n',nt,numdims,numhid);
for i=1:nModel
    [W1(:,:,i), bi1(:,i), bj1(:,i), A1(:,:,:,i), B1(:,:,:,i)] =...
        gaussiancrbmSM(batchdata, numhid1, n1, 50,...
        'lr', 1e-3, 'echoEpoch', 10, 'useState', 1);
    hiddenData(:,:,i) = ...
        genGRBMHidden( batchdata, n1, W1(:,:,i), B1(:,:,:,i), bj1(:,i), gsd);
end

%save Results/layer1.mat W1 bj1 bi1 A1 B1

fprintf(1,'Training Layer 2 CRBM, order %d: %d-%d \n',n2,numhid1,numhid2);
[W2, bi2, bj2, A2, B2] = binarycrbmMM(hiddenData, numhid2, n2, 50, ...
    'lr', 1e-3, 'useState', 1, 'echoEpoch', 10);
%save Results/layer2.mat W2 bj2 bi2 A2 B2

nT = 400;
meanfield = false;

%Now use the 2-layer CRBM to generate a sequence of data
fprintf(1,'Generating %d-frame sequence of data from 2-layer CRBM ... \n',nT);
visible3D = predictMM(initdata, W1, [], bi1, bj1, A1, B1, ...
    W2, bi2, bj2, A2, B2, gsd, ...
    nT, meanfield);

fprintf(1,'Playing generated sequence\n');

for i=1:size(visible3D,3)
    %We must postprocess the generated data before playing
    %It is in the normalized angle space
    visible = visible3D(:,:,i);
    postprocess;
    figure(2); expPlayData(skel, newdata, 1/30)
end

