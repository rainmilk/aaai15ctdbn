function [W, bi, bj, A, B, recData] = binarycrbmMM(batchdata, numhid, nt, epochs, varargin)

% The program assumes that the following variables are set externally:
% nt        -- order of the model
% numepochs -- maximum number of epochs
% numhid    --  number of hidden units
% batchdata --  a matrix of data (numcases,numdims)
% minibatch -- a cell array of dimension batchsize, indexing the valid

%% Set parameters
initArguments;

outputData = false;
if nargout > 5
    recData = batchdata;
    outputData = true;
end

%% Main loop
for epoch = 1:epochs
    errsum=0; %keep a running total of the difference between data and recon
    calErr = mod(epoch,echoEpoch) ==0;
    for batch = 1:numbatches
        numcases = length(minibatch{batch});
        mb = minibatch{batch}; %caches the indices
        data = zeros(numcases,numdims,nt+1,numModels);
        for hh=1:nt
            data(:,:,hh+1,:) = batchdata(mb-hh,:,:);
        end
        data(:,:,1,:) = batchdata(mb,:,:);
        %data is a nt+1-d array with current and delayed data
        %corresponding to this mini-batch
        
        bistar = zeros(numdims,numcases,numModels);
        bjstar = 0;
        bottomup = 0;
        %% START POSITIVE PHASE
        for m = 1:numModels   
            for hh=1:nt
                %Calculate contributions from directed autoregressive connections
                bistar(:,:,m) = bistar(:,:,m) +  A(:,:,hh,m)*data(:,:,hh+1,m)';
                
                %Calculate contributions from directed visible-to-hidden connections
                bjstar = bjstar + B(:,:,hh,m)*data(:,:,hh+1,m)';
            end
            
            bottomup = bottomup + W(:,:,m)*data(:,:,1,m)';
        end
        
        %Calculate "posterior" probability -- hidden state being on
        %Note that it isn't a true posterior
        eta =  bottomup + ...                %bottom-up connections
            repmat(bj, 1, numcases) + ...    %static biases on unit
            bjstar;                          %dynamic biases
        hposteriors = 1./(1 + exp(-eta));    %logistic
        
        %Activate the hidden units
        hidstates = double(hposteriors > rand(numhid, numcases));
        if useState, hposteriors = hidstates; end
        
        bjgrad = sum(hposteriors,2);
        for m = 1:numModels
            %Calculate positive gradients (note w.r.t. neg energy)
            wgrad(:,:,m) = hposteriors*data(:,:,1,m);
            bigrad(:,m) = sum(data(:,:,1,m)' - repmat(bi(:,m),1,numcases) - bistar(:,:,m), 2);
            for hh=1:nt
                Agrad(:,:,hh,m) = (data(:,:,1,m)' - repmat(bi(:,m),1,numcases) - bistar(:,:,m)) * data(:,:,hh+1,m);
                Bgrad(:,:,hh,m) = hposteriors*data(:,:,hh+1,m);
            end
        end          
        %END OF POSITIVE PHASE
        
      %% START NEGATIVE PHASE        
        negdata = zeros(numcases,numdims,numModels);
        for k=1:CDk
            bottomup = 0;
            if k > 1 % k==1 having been sampled
                %Activate the hidden units
                hidstates = double(hposteriors > rand(numhid,numcases));
            end
            for m = 1:numModels
                %Activate the visible units
                topdown = hidstates'*W(:,:,m);
            
                eta =  topdown + ...                    %top down connections
                    repmat(bi(:,m),1,numcases)' + ...   %static biases
                    bistar(:,:,m)';                     %dynamic biases
                
                negdata(:,:,m) = 1./(1+exp(-eta));
                
                %Now conditional on negdata, calculate "posterior" probability
                %for hiddens
                bottomup = bottomup + W(:,:,m)*negdata(:,:,m)';
            end
            
            eta =  bottomup + ...              %bottom-up connections
                repmat(bj, 1, numcases) + ...  %static biases on unit (no change)
                bjstar;                        %dynamic biases (no change)
            hposteriors = 1./(1 + exp(-eta));   %logistic
        end
        
        negbjgrad = sum(hposteriors, 2);
        for m = 1:numModels
            %Calculate negative gradients
            negwgrad(:,:,m) = hposteriors*negdata(:,:,m); %not using activations
            negbigrad(:,m) = sum(negdata(:,:,m)' - repmat(bi(:,m),1,numcases) - bistar(:,:,m), 2);
            
            for hh=1:nt
                negAgrad(:,:,hh,m) = (negdata(:,:,m)' - repmat(bi(:,m),1,numcases) - bistar(:,:,m)) * data(:,:,hh+1,m);
                negBgrad(:,:,hh,m) = hposteriors*data(:,:,hh+1,m);
            end
            
            if calErr
                errsum = errsum + sum(sum((data(:,:,1,m)-negdata(:,:,m)).^2));
            end
        end
        %%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if epoch > initEpoch %use momentum
            momentum=mom;
        else %no momentum
            momentum=initmom;
        end
        
        if L1WDecay
            decayW = wdecay*sign(W);
            decayA = wdecay*sign(A);
            decayB = wdecay*sign(B);
        else
            decayW = wdecay*W;
            decayA = wdecay*A;
            decayB = wdecay*B;
        end
        
        wupdate =  momentum*wupdate + ...
            epsilonW*( (wgrad - negwgrad)/numcases - decayW );
        
        biupdate = momentum*biupdate + ...
            epsilonbi*( (bigrad - negbigrad)/numcases );
        bjupdate = momentum*bjupdate + ...
            epsilonbj*( (bjgrad - negbjgrad)/numcases );
        
        Aupdate = momentum*Aupdate + ...
            epsilonA*( (Agrad - negAgrad)/numcases - decayA );
        
        Bupdate = momentum*Bupdate + ...
            epsilonB*( (Bgrad - negBgrad)/numcases - decayB );
        
        
        W = W +  wupdate;
        bi = bi + biupdate;
        bj = bj + bjupdate;
        A = A + Aupdate;
        B = B + Bupdate;
        
        %END OF UPDATES
        
        %Save recstructed data
        if outputData && epoch == epochs
            recData(mb,:,:) = negdata;
        end
        
    end
    
    decRate = (1 + lrDecay*epoch/epochs);
    epsilonW = epsilonW0 / decRate;
    epsilonbi = epsilonbi0 / decRate;
    epsilonbj = epsilonbj0 / decRate;
    epsilonA = epsilonA0 / decRate;
    epsilonB = epsilonB0 / decRate;
        
    %every 10 epochs, show output
    if calErr
        fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
    end
    
end

