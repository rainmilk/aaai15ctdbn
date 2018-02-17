function visible = predictMM(initdata, W1, R1, bi1, bj1, A1, B1, ...
    W2, bi2, bj2, A2, B2, gsd, ...
    nT, meanfield, nIter)

% This program uses the 2-level CRBM to generate data
% More efficient version than the original
%
% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

if nargin<15, meanfield = false; end
if nargin<16, nIter = 50; end %number of alternating Gibbs/Mean-filed iterations
if isempty(R1), R1 = W1; end

%We have saved some initialization data in "initdata"
%How many frames of it do we need to make a prediction for the first h2
%frame? we need n1 + n2 frames of data
n1 = size(A1,3);
n2 = size(A2,3);
numhid1 = size(bj1,1);
numhid2 = size(bj2,1);
max_clamped = n1 + n2;
numframes = max_clamped + nT;

%use this data to get the posteriors at layer 1
numcases = n2; %number of hidden units to generate
numdims = size(initdata,2);
numModels = size(initdata,3);

%initialize visible data
visible = zeros(numframes,numdims,numModels);
visible(1:max_clamped,:,:) = initdata(end-max_clamped+1:end,:,:);

data = zeros(numcases,numdims,n1+1,numModels);
dataindex = n1+1:max_clamped;

data(:,:,1,:) = visible(dataindex,:,:); %store current data
%store delayed data
for hh=1:n1
  data(:,:,hh+1,:) = visible(dataindex-hh,:,:);
end

hidden1 = ones(numhid1,numframes,numModels);
past = zeros(n2*numhid1,numModels);
for m=1:numModels
    %Calculate contributions from directed visible-to-hidden connections
    bjstar = zeros(numhid1,numcases);
    for hh = 1:n1
        bjstar = bjstar + B1(:,:,hh,m)*data(:,:,hh+1,m)';
    end
    
    %Calculate "posterior" probability -- hidden state being on
    %Note that it isn't a true posterior
    eta =  R1(:,:,m)*(data(:,:,1,m)./gsd)' + ...       %bottom-up connections
        repmat(bj1(:,m), 1, numcases) + ...         %static biases on unit
        bjstar;                                      %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));     %logistic
    
    %initialize hidden layer 1
    %first n1 frames are just padded
    hidden1(:,n1+1:n1+n2,m) = hposteriors;
    
    %keep the recent past in vector form
    %for input to directed links
    %it's slightly faster to pre-compute this vector and update it (by
    %shifting) at each time frame, rather than to fully recompute each time
    %frame
    pastframe = hidden1(:,max_clamped:-1:max_clamped+1-n2,m);
    past(:,m) = pastframe(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First generate a hidden sequence (top layer)
%Then go down through the first CRBM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize second layer (first n1+n2 frames padded)
hidden2 = zeros(numhid2,numframes);
bistar = zeros(numhid1,numModels);
W2cont = reshape(W2,numhid2,numModels*numhid1);
for tt=max_clamped+1:numframes  
  %initialize using the last frame
  %noise is not added for binary units
  hidden1tt = hidden1(:,tt-1,:);
  bjstar = zeros(numhid2,1);
  
  for m=1:numModels
      %Dynamic biases aren't re-calculated during Alternating Gibbs
      A2flat = reshape(A2(:,:,:,m),numhid1,n2*numhid1);
      B2flat = reshape(B2(:,:,:,m),numhid2,n2*numhid1);
      bistar(:,m) = A2flat*past(:,m);
      bjstar = bjstar + B2flat*past(:,m);
  end

  hidden1tt = hidden1tt(:);
  %Gibbs sampling
  for gg = 1:nIter
    %Calculate posterior probability -- hidden state being on (estimate)
    %add in bias
    bottomup = W2cont*hidden1tt;
    eta = bottomup + ...                   %bottom-up connections
      bj2 + ...                            %static biases on unit
      bjstar;                              %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));      %logistic
    
    if meanfield || gg >= nIter
        hidden2(:,tt) = hposteriors;
    else
        hidden2(:,tt) = double(hposteriors > rand(numhid2,1)); %Activating hiddens
    end
    
    %Downward pass; visibles are binary logistic units     
    topdown = hidden2(:,tt)'*W2cont;
        
    eta = topdown' + ...                    %top down connections
      bi2(:) + ...                             %static biases
      bistar(:);                               %dynamic biases   
    hidden1tt = 1./(1 + exp(-eta));         %logistic 
  end
  hidden1(:,tt,:) = reshape(hidden1tt, numhid1, numModels);
  %update the past
  past(numhid1+1:end,:) = past(1:end-numhid1,:);                %shift older history down
  past(1:numhid1,:) = reshape(hidden1tt, numhid1, numModels);   %place most recent frame at top
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Now that we've decided on the "filtering distribution", generate visible
%data through CRBM 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for tt=max_clamped+1:numframes
    for m=1:numModels
        %Add contributions from autoregressive connections
        bistar = zeros(numdims,1);
        for hh=1:n1
            bistar = bistar +  A1(:,:,hh,m)*visible(tt-hh,:,m)' ;
        end

        %Mean-field approx; visible units are Gaussian
        %(filtering distribution is the data we just generated)
        topdown = gsd.*(hidden1(:,tt,m)'*W1(:,:,m));
        visible(tt,:,m) = topdown + ...        %top down connections
            bi1(:,m)' + ...                       %static biases
            bistar';                              %dynamic biases
    end
end

visible = visible(end-nT+1:end,:,:);