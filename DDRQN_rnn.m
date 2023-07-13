function [u_dlstm,K,info] = DDRQN_rnn(TRAIN,TEST,N)
% tic
input_train = TRAIN.input;
input_test = TEST.input;
target_train = TRAIN.target*1E-06;
target_test = TEST.target*1E-06;
input_train = input_train(:,1:N);
input_test = input_test(:,1:N);
% km = TRAIN.k;
numFeatures = size(input_train,1);
numResponses = size(target_train,1);
% dsTrain = combine(input_train,target_train);
% dsValidation =  combine(input_test,target_test);
%%
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",1000,...
    "Shuffle","every-epoch",...
    "Plots","none");
%% 
lgraph = layerGraph();
%%
tempLayers = sequenceInputLayer(numFeatures,"Name","sequence");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(64,"Name","lstm");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(64,"Name","lstm_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    fullyConnectedLayer(numResponses,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%%
lgraph = connectLayers(lgraph,"sequence","lstm");
lgraph = connectLayers(lgraph,"sequence","lstm_1");
lgraph = connectLayers(lgraph,"lstm","addition/in1");
lgraph = connectLayers(lgraph,"lstm_1","addition/in2");
%%
[net, ~] = trainNetwork(input_train,target_train,lgraph,opts);
time_elapsed = toc
status = StabilityLSTM(net,target_test)
%% Predict 
net = predictAndUpdateState(net,input_test);
[~,Qhat] = predictAndUpdateState(net,input_test);

%% Estimate the K via RNN
xx = con2seq(Qhat);
yy = con2seq(target_train);
lrn_net = layrecnet(1:2,10);
lrn_net.trainFcn = 'trainlm';
% lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 100;
lrn_net = train(lrn_net,xx,yy);
K = lrn_net(target_test);
%% Evaluate Performance 
predictionError = target_test-Qhat; 
thr = 0.001;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(target_test);
accuracy = numCorrect/numValidationImages;
u_dlstm = K.*TRAIN.y;
%%
info.time = time_elapsed; 
info.status = status;
info.net = net;
info.netK = lrn_net;
info.numCorrect = numCorrect;
info.Qhat = Qhat;