function [W, R, bi, bj, A, B] = gcrbmFT( batchdata, wakedata, sleepdata, numhid, nt, epochs, varargin)
% nt        -- order of the model
% gsd       -- fixed standard deviation of Gaussian visible units
% numepochs -- maximum number of epochs
% numhid    --  number of hidden units
% batchdata --  a matrix of data (numcases,numdims)
% minibatch -- a cell array of dimension batchsize, indexing the valid

%% Set parameters
initArguments;

if useState
    rnddata = rand(size(wakedata));
    wakedata = wakedata > rnddata;
    sleepdata = sleepdata > rnddata;
    clear rnddata;
end
%% Main loop
for epoch = 1:epochs
    errsum=0; %keep a running total of the difference between data and recon
    calErr = mod(epoch,echoEpoch) ==0;
    for batch = 1:numbatches
        numcases = length(minibatch{batch});
        mb = minibatch{batch}; %caches the indices
        
        %data is a nt+1-d array with current and delayed data
        %corresponding to this mini-batch
        data = zeros(numcases,numdims,nt+1);
        data(:,:,1) = batchdata(mb,:);
        for hh=1:nt
            data(:,:,hh+1) = batchdata(mb-hh,:);
        end
        wkstates = wakedata(mb-nt,:); 
        slstates = sleepdata(mb-nt,:);
        
        %Calculate contributions from directed autoregressive connections
        bistar = zeros(numdims,numcases);
        for hh=1:nt
            bistar = bistar +  A(:,:,hh)*data(:,:,hh+1)' ;
        end
        
        %Calculate contributions from directed visible-to-hidden connections
        bjstar = zeros(numhid,numcases);
        for hh = 1:nt
            bjstar = bjstar + B(:,:,hh)*data(:,:,hh+1)';
        end
        
        %This is the mean of the Gaussian
        %Instead of properly sampling, negdata is just the mean
        %If we want to sample from the Gaussian, we would add in
        %gsd.*randn(numcases,numdims);
        wakevis = gsd.*(wkstates*W) + ...   %top down connections
            repmat(bi',numcases,1) + ...    %static biases
            bistar';                        %dynamic biases
        
        %Calculate generative parameter gradients
        wgrad = wkstates'*(data(:,:,1)./gsd);
        negwgrad = wkstates'*(wakevis./gsd); %not using activations
        
        bigrad = sum(data(:,:,1)' - repmat(bi,1,numcases) - bistar, 2)./gsd.^2;
        negbigrad = sum(wakevis' - repmat(bi,1,numcases) - bistar, 2)./gsd.^2;
        
        for hh=1:nt
            Agrad(:,:,hh) = (data(:,:,1)' -  ...
                repmat(bi,1,numcases) - bistar)./gsd.^2 * data(:,:,hh+1);
            negAgrad(:,:,hh) = (wakevis' - ...
                repmat(bi,1,numcases) - bistar)./gsd.^2 * data(:,:,hh+1);
        end
        
        %Calculate bottom-up prediction
        sleepvis = gsd.*(slstates*W) + ...  %top down connections
            repmat(bi',numcases,1) + ...    %static biases
            bistar';                        %dynamic biases
        
        eta =  R*(sleepvis./gsd)' + ...     %bottom-up connections
            repmat(bj, 1, numcases) + ...   %static biases on unit
            bjstar;                         %dynamic biases
        
        sleephid = 1./(1 + exp(-eta));    %logistic
        
        %Calculate recognition parameter gradients
        Rgrad = slstates'*(sleepvis./gsd);
        negRgrad = sleephid*(sleepvis./gsd);
        
        bjgrad = sum(slstates,1)';
        negbjgrad = sum(sleephid,2);
        
        for hh=1:nt
            Bgrad(:,:,hh) = slstates'*data(:,:,hh+1);
            negBgrad(:,:,hh) = sleephid*data(:,:,hh+1);
        end
        
        %%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if calErr, errsum = errsum + sum(sum( (data(:,:,1)-wakevis).^2 )); end
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if epoch > initEpoch %use momentum
            momentum=mom;
        else %no momentum
            momentum=initmom;
        end
        
        if L1WDecay
            decayW = wdecay*sign(W);
            decayR = wdecay*sign(R);
            decayA = wdecay*sign(A);
            decayB = wdecay*sign(B);
        else
            decayW = wdecay*W;
            decayR = wdecay*R;
            decayA = wdecay*A;
            decayB = wdecay*B;
        end
        
        wupdate =  momentum*wupdate + ...
            epsilonW*( (wgrad - negwgrad)/numcases - decayW );
        Rupdate = momentum*Rupdate + ...
            epsilonW*( (Rgrad - negRgrad)/numcases - decayR );
        
        biupdate = momentum*biupdate + ...
            epsilonbi*( (bigrad - negbigrad)/numcases );
        bjupdate = momentum*bjupdate + ...
            epsilonbj*( (bjgrad - negbjgrad)/numcases );
        
        Aupdate = momentum*Aupdate + ...
            epsilonA*( (Agrad - negAgrad)/numcases - decayA );
        
        Bupdate = momentum*Bupdate + ...
            epsilonB*( (Bgrad - negBgrad)/numcases - decayB );
        
        
        W = W +  wupdate;
        R = R + Rupdate;
        bi = bi + biupdate;
        bj = bj + bjupdate;
        A = A + Aupdate;
        B = B + Bupdate;
        %%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    decRate = (1 + lrDecay*epoch/epochs);
    epsilonW = epsilonW0 / decRate;
    epsilonbi = epsilonbi0 / decRate;
    epsilonbj = epsilonbj0 / decRate;
    epsilonA = epsilonA0 / decRate;
    epsilonB = epsilonB0 / decRate;
    
    % show output
    if calErr
        fprintf(1, 'fine tune, error %6.1f  \n', errsum);
    end
    
end

end

