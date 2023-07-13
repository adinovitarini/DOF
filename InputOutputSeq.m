function [UN,VN,TN,u_bar,y_bar] = InputOutputSeq(cartpole,TRAIN,N,df)
%%
y = TRAIN.target;
u = TRAIN.input;
%% Define the augmented variable
% N = 100;

%% Model-based LQ Control
A = df*cartpole.sys.A;
B = df*cartpole.sys.B;
C = df*cartpole.sys.C;
m = size(B,2);
p = size(A,1);
n = size(C,2);
for k = 1:N
    if k~=1
    UN(:,k) = df*A^(k-1)*B;
    end
    VN(k,:) = df*C*A^(N-k);
end
% UN = ctrb(A,B);
% VN = obsv(A,C);
%% Toeplitz Matrix 
c = zeros(N,1);
for i =1:N-1
    r(:,i) = df*C*A^(i-1)*B;
end
r = [zeros(1,1) r];
TN = toeplitz(c,r);
TN = TN(1:N,1:N);
%% Toeplitz Matrix 
% 
% c = zeros(p,m);
% % for i = 1:p-1
% %     r(:,i+1) = C*A^(i-1)*B;
% % end
% % r(:,p) = C*A^(N-2)*B;
% % r = c;
% for i =1:N-1
%     r(:,i) = df*C*A^(i-1)*B;
% end
% % r = df*C*A^(N-1)*B;
% r = [zeros(1,1) r];
% TN = toeplitz(c,r);
% % TN = TN(1:N,1:N);
%% Iterate input-output data
N = size(u,2);
u_bar = zeros(N,1);
y_bar = zeros(N,1); 
for i = 2:N
    if i~=1
        u_bar(i) = u(i-1);
        y_bar(i) = y(i-1);
    end 
end
% for i=1:N-1
%     z_bar(:,i) = [u(i);u_bar(i);y_bar(i)];
% end
% VNplus = inv(VN'*VN)*VN';
% Mo = A^N*VNplus;
% My = Mo;
% Mu = UN-Mo*TN;