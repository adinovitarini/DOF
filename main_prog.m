%% Main Program for DOF framework 
clear all;
clc
Ry = 10;
Ru = .1;
N = 100; 
df = .1;
% Set the model parameter :
% Ry : weight state matrices
% Ru : weight consig matrices 
% df : discount factor 
% N : number of sample-data
model.Ry = Ry;
model.Ru = Ru;
model.df = df;
model.N = N;
% Choose the system :
% 1. Cart-Pole System 
% 2. Distillate System 
% 3. Unstable System 
% cartpole = sysmdl_cartpole(N,1);
cartpole = sysmdl_distillate(N,1);
% cartpole = sysmdl_unstable(N,df);
tic
dataset_cp = GenerateSeq(cartpole.sys,N,.00001,1,Ry,Ru);

time_model = toc
dataset_cp_test = GenerateSeq(cartpole.sys,N,1,1,Ry,Ru);
TRAIN.input = [dataset_cp.u_nw(1:N);dataset_cp.y_nw];
TRAIN.y = dataset_cp.y_nw;
TRAIN.k = dataset_cp.K;
% TRAIN.target = dataset_cp.y;
TRAIN.target = dataset_cp.reward;
TEST.input = [dataset_cp_test.u_nw(1:N);dataset_cp_test.y_nw];
TEST.k = dataset_cp_test.K;
% TEST.target = dataset_cp_test.y;
TEST.target = dataset_cp.reward;
TEST.y = dataset_cp_test.y_nw;

%% 
[UN,VN,TN,u_bar,y_bar] = InputOutputSeq(cartpole,TRAIN,N,df);
u = dataset_cp.u_nw(1:N);
y = TRAIN.y;
Q = q_func(model,dataset_cp.reward);
[u_q,z_bar,info_q] = q_learn_io(N,u,y,Ry,Ru,df,u_bar,y_bar);
%% Data Test 
[UN_t,VN_t,TN_t,u_bar_t,y_bar_t] = InputOutputSeq(cartpole,TEST,N,df);
u_t = dataset_cp_test.u_nw(1:N);
y_t = TEST.target;
% Q_t = q_func(model,dataset_cp);
[u_q_t,z_bar_t,info_t] = q_learn_io(N,u_t,y_t,Ry,Ru,df,u_bar_t,y_bar_t);
%% DRQN 
TRAIN.target = Q;
% [u_lstm,K_lstm,info_drqn] = DRQN(TRAIN,TEST,N);
[u_lstm,K,info_drqn] = DRQN_rnn(TRAIN,TEST,N);
% [u_dlstm,Kd,info_ddrqn] = DDRQN(TRAIN,TEST,N);
[u_dlstm,K,info_ddrqn] = DDRQN_rnn(TRAIN,TEST,N);
u_lstm = u_lstm;
%% Check Convergence Q-Learn
% phi = kron(z_bar',z_bar');
% rank_phi = rank(phi)
%% 
% net = info_drqn.net;
% u_bar_lstm = net.Layers(2).HiddenState;
% cc = net.Layers(2).CellState;
% qq = hh*cc';
