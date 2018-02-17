clear
batchdata = xlsread('DICH_pre.xlsx');
updownratio = (batchdata(2:end,:) - batchdata(1:end-1,:))./batchdata(1:end-1,:);
%batchdata = batchdata(:,[1,3,5,2,4,6]); % Cross-market
[ batchdata, means, stds ] = normalizeSND( updownratio );

gsd = 1;            % standard deviation
numdims = 5;        % number of visible units for layer1 RBM
nModel = size(batchdata,2)/numdims;
numhid1 = 60;       % number of hidden units for layer1 RBM
numhid2 = 60;

cty = 5; % target country
mkt = 1; % target market;

batchdata = reshape(batchdata, size(batchdata,1), numdims, nModel); % 3D array time-features-models
updownratio = reshape(updownratio, size(batchdata));

startP = 301;       % start prediction from t=startP
numTs= 100;         % number of testing size
batchPre = 100;      % Predict T+batchPre per training
numTr = 300;        % number of training size
ts_data = batchdata(startP:startP+numTs-1,cty,mkt);
ts_updownratio = updownratio(startP:startP+numTs-1,cty,mkt);

nT = 1;             % Predict next nT steps.
meanfield = true;   % using meanfield or Gibbs sampling
useState = false;    % use states or probabilities to construct hidden units
L1WDecay = false;   % use L1 (sparse) or L2 (small) regularization

epochs1 = 3000;     % iterations for layer 1 RBM
wtDecay1 = 2e-3;       % regularization parameter for layer 1 RBM
lr1 = 1e-4;         % learning rate for layer 1 RBM
lr1f = 1e-4;        % learning rate for fine tuning layer 1 RBM

epochs2 = 3000;     % iterations for layer 2 RBM
wtDecay2 = 2e-3;       % regularization parameter for layer 2 RBM
lr2 = 1e-3;         % learning rate for layer 2 RBM
lr2f = 1e-4;        % learning rate for fine tuning layer 2 RBM

finetune = true;    % if fine tuning
ftEpoch = 100;      % fine tuning epoches
ftsubEpoch = 20;     % fine tuning epoches for RBM

n1s = [1,2,3,4];          % orders for layer 1 model
n2s = [1,2,3,4];          % orders for layer 2 model
parfor o=1:length(n1s)	% parallel learning for models with different orders
    n1 = n1s(o);	% 1st layer order
    n2 = n2s(o);	% 2nd layer order
    
    %% Prediction
    predicts1 = zeros(numTs,1);
    predicts2 = zeros(numTs,1);
    fprintf('Starting prediction.\n');
    offset = 0;
    
    startTime = tic;
    for n=1:numTs
        %% Training
        if mod(n-1,batchPre) == 0   % only re-training model per batchPre predictions
            interval = max(1,(startP+offset-numTr)):(startP+offset-1);
            tr_data = batchdata(interval,:,:);
            offset = offset + batchPre;
            
            W1 = zeros(numhid1,numdims,nModel);
            bi1 = zeros(numdims,nModel);
            bj1 = zeros(numhid1,nModel);
            A1 = zeros(numdims, numdims, n1, nModel);
            B1 = zeros(numhid1, numdims, n1, nModel);
            wakeData = zeros(numTr-n1,numhid1,nModel);
            for i=1:nModel
                fprintf(1,'Training Layer 1 CRBM(%d), order %d: %d-%d \n',i,n1,numdims,numhid1);
                [W1(:,:,i), bi1(:,i), bj1(:,i), A1(:,:,:,i), B1(:,:,:,i)] =...
                    gaussiancrbmSM(tr_data(:,:,i), numhid1, n1, epochs1,...
                    'lr',lr1, 'wtDecay', wtDecay1, 'L1WDecay', L1WDecay, ...
                    'useState', useState, 'echoEpoch',500);
                wakeData(:,:,i) = ...
                    genGRBMHidden( tr_data(:,:,i), n1, W1(:,:,i), B1(:,:,:,i), bj1(:,i), gsd );
            end
            R1 = W1; % recognition parameter
            
            fprintf(1,'Training Layer 2 CRBM, order %d: %d-%d \n',n2,numhid1,numhid2);
            [W2, bi2, bj2, A2, B2, sleepdata] = binarycrbmMM(wakeData, numhid2, n2, epochs2,...
                'lr', lr2,'wtDecay', wtDecay2, 'L1WDecay', L1WDecay, ...
                'useState', useState, 'echoEpoch',500);
            
            %% Fine tuning (optional)
            if finetune
                fprintf(1,'Fine tuning CTDBN, order %d-%d\n',n1,n2);
                param1 = cell(nModel,1);
                param2 = struct;
                for epoch=1:ftEpoch
                    param2.W = W2; param2.bi = bi2; param2.bj = bj2;
                    param2.A = A2; param2.B = B2;
                    [W2, bi2, bj2, A2, B2, sleepdata] = ...
                        binarycrbmMM(wakeData, numhid2, n2, ftsubEpoch,...
                        'lr', lr2f, 'wtDecay', wtDecay2, 'L1WDecay', L1WDecay, ...
                        'useState', useState, 'echoEpoch', ftsubEpoch, 'initParam', param2);
                    
                    for i=1:nModel
                        param1{i}.W = W1(:,:,i); param1{i}.R = R1(:,:,i); param1{i}.bi = bi1(:,i);
                        param1{i}.bj = bj1(:,i); param1{i}.A = A1(:,:,:,i); param1{i}.B =  B1(:,:,:,i);
                        [W1(:,:,i), R1(:,:,i), bi1(:,i), bj1(:,i), A1(:,:,:,i), B1(:,:,:,i)] = ...
                            gcrbmFT( tr_data(:,:,i), wakeData(:,:,i), sleepdata(:,:,i),...
                            numhid1, n1, ftsubEpoch, 'lr', lr1f, 'wtDecay', wtDecay1, 'L1WDecay', L1WDecay,...
                            'useState', useState, 'echoEpoch', ftsubEpoch, 'initParam', param1{i});
                        wakeData(:,:,i) = ...
                            genGRBMHidden( tr_data(:,:,i), n1, param1{i}.R, param1{i}.B, param1{i}.bj, gsd );
                    end
                end
            end
        end
        
        initdata = batchdata(1:startP+n-2,:,:);
        % Only use 1st layer CRBM for prediction
        predictV = predictSM(initdata(:,:,mkt), W1(:,:,mkt), R1(:,:,mkt), ...
            bi1(:,mkt), bj1(:,mkt), A1(:,:,:,mkt), B1(:,:,:,mkt), gsd, ...
            nT, meanfield, 200);
        predicts1(n) = predictV(1,cty);
        
        % Use Two-layer CTDBN for prediction
        predictV = predictMM(initdata, W1, R1, bi1, bj1, A1, B1, ...
            W2, bi2, bj2, A2, B2, gsd, ...
            nT, meanfield, 200);
        predicts2(n) = predictV(1,cty,mkt);
        
        if mod(n,batchPre) == 0
            deltaTime = toc(startTime);
            fprintf('Finished prediction for the model %d-%d: %g%%, Est. %g secs remaining\n', ...
                n1, n2, n/numTs*100, deltaTime/n*(numTs-n));
        end
    end
    fprintf('Finished all prediction for the model %d-%d.\n', n1, n2);
    
    rmse1(o) = sqrt(mean( (predicts1 - ts_data).^2 ));
    rmse2(o) = sqrt(mean( (predicts2 - ts_data).^2 ));
    
    %updown = (predicts(2:end)-predicts(1:end-1))>0;
    %updownTrue = (ts_data(2:end)-ts_data(1:end-1))>0;
    
    updownPredict1 = predicts1>0;
    updownPredict2 = predicts2>0;
    
    updownTrue = ts_data>0;
    
    hits1(:,o) = ( updownPredict1 == updownTrue);
    hits2(:,o) = ( updownPredict2 == updownTrue);
    
    profit1(o) = prod( 1 + ts_updownratio(updownPredict1) );
    profit2(o) = prod( 1 + ts_updownratio(updownPredict2) );
end
acc1 = mean(hits1);
cuacc1 = bsxfun(@times, cumsum(hits1,1), 1./(1:length(hits1))');



acc2 = mean(hits2);
cuacc2 = bsxfun(@times, cumsum(hits2,1), 1./(1:length(hits2))');

fprintf('Single Layer CRBM RMSE: %g\n', rmse1);
fprintf('Single Layer CRBM Accuracy: %g\n', acc1);
fprintf('Single Layer CRBM Profit: %g\n', profit1);



fprintf('Two Layer CTDBN RMSE: %g\n', rmse2);
fprintf('Two Layer CTDBN Accuracy: %g\n', acc2);
fprintf('Two Layer CTDBN Profit: %g\n', profit2);