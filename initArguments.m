numModels = size(batchdata,3);

numdims = size(batchdata,2); %visible dimension

p = inputParser;
p.addParamValue('miniBatch',{}, @iscell);
p.addParamValue('batchSize',100, @isnumeric);
p.addParamValue('initParam',{}, @isstruct);
% standard deviation
p.addParamValue('sd',1, @isnumeric);
p.addParamValue('lr',1e-4, @(x) isstruct(x)||isnumeric(x));
p.addParamValue('lrDecay',0, @isnumeric);
p.addParamValue('L1WDecay',0, @(x) isnumeric(x)||islogical(x));
%currently we use the same weight decay for w, A, B
p.addParamValue('wtDecay',2e-4, @isnumeric);
p.addParamValue('initmomentum',0.5, @isnumeric);
p.addParamValue('momentum',0.9, @isnumeric);
p.addParamValue('initEpoch',10, @isnumeric);
p.addParamValue('echoEpoch',inf, @isnumeric);
p.addParamValue('useState', 0, @(x) isnumeric(x)||islogical(x));

p.addParamValue('CDk',1, @isnumeric);
p.addParamValue('PCD',0, @(x) isnumeric(x)||islogical(x));
p.parse(varargin{:});
args = p.Results;

gsd = args.sd;
CDk = args.CDk;
PCD = (args.PCD ~= 0);
lrDecay = args.lrDecay ~= 0;
initmom = args.initmomentum;
mom = args.momentum;
initEpoch = args.initEpoch;
wdecay = args.wtDecay; 
L1WDecay = (args.L1WDecay~=0);
echoEpoch = args.echoEpoch;
useState = args.useState;

minibatch = args.miniBatch;
if isempty(minibatch)
    minibatch = genMiniBatch(nt, size(batchdata,1), args.batchSize);
end

batchSize = length(minibatch{1});

if isempty(args.initParam)
    %Randomly initialize weights
    W = 0.01*randn(numhid,numdims,numModels);
    bi = 0.01*randn(numdims,numModels);
    bj = -1+0.01*randn(numhid,1); %set to favor units being "off"
    
    %The autoregressive weights; A(:,:,j) is the weight from t-j to the vis
    A = 0.01*randn(numdims, numdims, nt, numModels);
    
    %The weights from previous time-steps to the hiddens; B(:,:,j) is the
    %weight from t-j to the hidden layer
    B = 0.01*randn(numhid, numdims, nt, numModels);
else
    W = args.initParam.W;
    bi = args.initParam.bi;
    bj = args.initParam.bj;
    A = args.initParam.A;
    B = args.initParam.B;
    if isfield(args.initParam,'R'), R = args.initParam.R; end
end

%Setting learning rates
if isstruct(args.lr)
    epsilonW0=args.lr.W;   %undirected
    epsilonbi0=args.lr.bi; %visibles
    epsilonbj0=args.lr.bj; %hidden units
    epsilonA0=args.lr.A;   %autoregressive
    epsilonB0=args.lr.B;   %prev visibles to hidden
else
    epsilonW0=args.lr;  %undirected
    epsilonbi0=args.lr; %visibles
    epsilonbj0=args.lr; %hidden units
    epsilonA0=args.lr;  %autoregressive
    epsilonB0=args.lr;  %prev visibles to hidden
end
epsilonW=epsilonW0;
epsilonbi=epsilonbi0;
epsilonbj=epsilonbj0;
epsilonA=epsilonA0;
epsilonB=epsilonB0;

%keep previous updates around for momentum
wupdate = zeros(size(W));
biupdate = zeros(size(bi));
bjupdate = zeros(size(bj));
Aupdate = zeros(size(A));
Bupdate = zeros(size(B));
if isfield(args.initParam,'R'), Rupdate = zeros(size(R)); end

wgrad = zeros(size(W));
negwgrad = zeros(size(W));
bigrad = zeros(size(bi));
negbigrad = zeros(size(bi));
Agrad = zeros(size(A));
Bgrad = zeros(size(B));
negAgrad = zeros(size(A));
negBgrad = zeros(size(B));

numbatches = length(minibatch);

hidstatesPCD = [];