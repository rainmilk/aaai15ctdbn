function visible = predictSM(initdata, W1, R1, bi1, bj1, A1, B1, gsd, ...
    nT, meanfield, nIter)

% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

if nargin<10, meanfield = false; end
if nargin < 11, nIter = 50; end %number of alternating Gibbs iterations
if isempty(R1), R1 = W1; end

n1 = size(A1,3);
numdims = size(initdata,2);
numhid1 = size(bj1,1);
numframes = n1 + nT;

%initialize visible layer
visible = zeros(numframes,numdims);
visible(1:n1,:) = initdata(end-n1+1:end,:);
%initialize hidden layer
hidden1 = ones(numframes,numhid1);

for tt=n1+1:numframes
  
  %initialize using the last frame + noise
  visible(tt,:) = visible(tt-1,:) + 0.01*randn(1,numdims);
  
  %Dynamic biases aren't re-calculated during Alternating Gibbs
  %First, add contributions from autoregressive connections 
  bistar = zeros(numdims,1);
  for hh=1:n1
    %should modify to data * A'
    bistar = bistar +  A1(:,:,hh)*visible(tt-hh,:)' ;
  end

  %Next, add contributions to hidden units from previous time steps
  bjstar = zeros(numhid1,1);
  for hh = 1:n1
    bjstar = bjstar + B1(:,:,hh)*visible(tt-hh,:)';
  end

  %Gibbs sampling
  for gg = 1:nIter
    %Calculate posterior probability -- hidden state being on (estimate)
    %add in bias
    bottomup =  R1*(visible(tt,:)./gsd)';
    eta = bottomup + ...                   %bottom-up connections
      bj1 + ...                            %static biases on unit
      bjstar;                              %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));      %logistic
    
    if meanfield
        hidden1(tt,:) = hposteriors;
    else
        hidden1(tt,:) = double(hposteriors' > rand(1,numhid1));
    end
    
    %Downward pass; visibles are Gaussian units
    %So find the mean of the Gaussian    
    topdown = gsd.*(hidden1(tt,:)*W1);
    
    %Mean-field approx
    visible(tt,:) = topdown + ...            %top down connections
      bi1' + ...                             %static biases
      bistar';                               %dynamic biases     
  
  end

  %If we are done Gibbs sampling, then do a mean-field sample
  %(otherwise very noisy)  
  topdown = gsd.*(hposteriors'*W1);                 

  visible(tt,:) = topdown + ...              %top down connections
      bi1' + ...                             %static biases
      bistar';                               %dynamic biases
end

visible = visible(end-nT+1:end,:);