function [W, bi, bj, A, B] = gaussiancrbmSM(batchdata, numhid, nt, numItr, varargin)
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

% This program trains a Conditional Restricted Boltzmann Machine in which
% visible, Gaussian-distributed inputs are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% Directed connections are present, from the past nt configurations of the
% visible units to the current visible units (A), and the past nt
% configurations of the visible units to the current hidden units (B)

% The program assumes that the following variables are set externally:
% nt        -- order of the model
% gsd       -- fixed standard deviation of Gaussian visible units
% numepochs -- maximum number of epochs
% numhid    --  number of hidden units
% batchdata --  a matrix of data (numcases,numdims)
% minibatch -- a cell array of dimension batchsize, indexing the valid

%% Set parameters
initArguments;

%% Main loop
for epoch = 1:numItr
    errsum=0; %keep a running total of the difference between data and recon
    calErr = mod(epoch,echoItrs) ==0;
    for batch = 1:numbatches
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        numcases = length(minibatch{batch});
        mb = minibatch{batch}; %caches the indices
        
        %data is a nt+1-d array with current and delayed data
        %corresponding to this mini-batch
        data = zeros(numcases,numdims,nt+1);
        data(:,:,1) = batchdata(mb,:);
        for hh=1:nt
            data(:,:,hh+1) = batchdata(mb-hh,:);
        end
        
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
        
        %Calculate "posterior" probability -- hidden state being on
        %Note that it isn't a true posterior
        eta =  W*(data(:,:,1)./gsd)' + ...   %bottom-up connections
            repmat(bj, 1, numcases) + ...      %static biases on unit
            bjstar;                            %dynamic biases
        
        hposteriors = 1./(1 + exp(-eta));    %logistic
        
        %Calculate positive gradients (note w.r.t. neg energy)
        wgrad = hposteriors*(data(:,:,1)./gsd);
        bigrad = sum(data(:,:,1)' - ...
            repmat(bi,1,numcases) - bistar,2)./gsd.^2;
        bjgrad = sum(hposteriors,2);
        
        for hh=1:nt
            Agrad(:,:,hh) = (data(:,:,1)' -  ...
                repmat(bi,1,numcases) - bistar)./gsd.^2 * data(:,:,hh+1);
            Bgrad(:,:,hh) = hposteriors*data(:,:,hh+1);
        end
        
        %%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Start CD_k
        for k=1:CDk
            %Activate the hidden units
            hidstates = double(hposteriors' > rand(numcases,numhid));
            
            %Activate the visible units
            %Find the mean of the Gaussian
            topdown = gsd.*(hidstates*W);
            
            %This is the mean of the Gaussian
            %Instead of properly sampling, negdata is just the mean
            %If we want to sample from the Gaussian, we would add in
            %gsd.*randn(numcases,numdims);
            negdata = topdown + ...             %top down connections
                repmat(bi',numcases,1) + ...   %static biases
                bistar';                        %dynamic biases
            
            %Now conditional on negdata, calculate "posterior" probability
            %for hiddens
            eta = W*(negdata./gsd)' + ...      %bottom-up connections
                repmat(bj, 1, numcases) + ... %static biases on unit (no change)
                bjstar;                        %dynamic biases (no change)
            
            hposteriors = 1./(1 + exp(-eta));   %logistic
        end
            
        %Calculate negative gradients
        negwgrad = hposteriors*(negdata./gsd); %not using activations
        negbigrad = sum( negdata' - repmat(bi,1,numcases) - bistar, 2 )./gsd.^2;
        negbjgrad = sum(hposteriors,2);
        
        for hh=1:nt
            negAgrad(:,:,hh) = (negdata' - repmat(bi,1,numcases) - bistar)./gsd.^2 * data(:,:,hh+1);
            negBgrad(:,:,hh) = hposteriors*data(:,:,hh+1);
        end
        
        %%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if calErr, errsum = errsum + sum(sum( (data(:,:,1)-negdata).^2 )); end
       
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if epoch > usemomatItr %use momentum
            momentum=mom;
        else %no momentum
            momentum=0;
        end
        
        if L1WDecay
            regW = wdecay*sign(W); regA = wdecay*sign(A); regB = wdecay*sign(B);
        else
            regW = wdecay*W; regA = wdecay*A; regB = wdecay*B;
        end
        
        wupdate =  momentum*wupdate + epsilonW* ...
            ( (wgrad - negwgrad)/numcases - regW );
        
        biupdate = momentum*biupdate + ...
            epsilonbi*( (bigrad - negbigrad)/numcases );
        bjupdate = momentum*bjupdate + ...
            epsilonbj*( (bjgrad - negbjgrad)/numcases );
        
        Aupdate = momentum*Aupdate + ...
            epsilonA*( (Agrad - negAgrad)/numcases - regA );
        
        Bupdate = momentum*Bupdate + ...
            epsilonB*( (Bgrad - negBgrad)/numcases - regB );
        
        
        W = W +  wupdate;
        bi = bi + biupdate;
        bj = bj + bjupdate;
        A = A + Aupdate;
        B = B + Bupdate;
        %%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    decRate = (1 + lrDecay*epoch/numItr);
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

end

