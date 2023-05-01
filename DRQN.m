function [u_lstm,K,info] = DRQN(TRAIN,TEST,N)
% N = 100;
tic
input_train = TRAIN.input();
input_test = TEST.input();
target_train = TRAIN.target();
target_test = TEST.target();
input_train = input_train(1,1:N);
% input_train = z_bar(:,1:N);
input_test = input_test(1,1:N);
numFeatures = size(input_train,1);
numResponses = size(target_train,1);
numHiddenUnits = 10;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'StateActivationFunction','tanh')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','none');
%% Train LSTM Net
net = trainNetwork(input_train,target_train,layers,options);
time_elapsed = toc
status = StabilityLSTM(net,target_test)
%% Predict 
net = predictAndUpdateState(net,input_test);
[~,K] = predictAndUpdateState(net,input_test);
u_lstm = K.*target_test;
info.time = time_elapsed; 
info.status = status;;