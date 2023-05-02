%% Main Program for DOF framework 
clear all;clc
Q = 0.1;
R = 0.1;
N = 100; 
df = 0.9;
cartpole = sysmdl_cartpole(N,df);
% cartpole = sysmdl_distillate(N,df);
dataset_cp = GenerateSeq(cartpole.sys,N,0,0,Q,R);
dataset_cp_test = GenerateSeq(cartpole.sys,N,0,0,Q,R);
TRAIN.input = dataset_cp.u;
TRAIN.target = dataset_cp.y_nw;
TEST.input = dataset_cp_test.u;
TEST.target = dataset_cp_test.y_nw;
%% 
[UN,VN,TN,u_bar,y_bar] = InputOutputSeq(cartpole,TRAIN,N,df);
u = TRAIN.input;
y = TRAIN.target;
[u_q,z_bar,info] = q_learn_io(N,u,y,Q,R,df,u_bar,y_bar);
%% Data Test 
[UN_t,VN_t,TN_t,u_bar_t,y_bar_t] = InputOutputSeq(cartpole,TEST,N,df);
u_t = TEST.input;
y_t = TEST.target;
[u_q_t,z_bar_t,info_t] = q_learn_io(N,u_t,y_t,Q,R,df,u_bar_t,y_bar_t);
%% DRQN 
[u_lstm,K,info] = DRQN(TRAIN,TEST,N);