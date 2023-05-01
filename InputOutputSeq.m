function [UN,VN,TN,u_bar,y_bar] = InputOutputSeq(cartpole,TRAIN,N,df)
y = TRAIN.target;
u = TRAIN.input;
%% Define the augmented variable
% N = 100;
%% Model-based LQ Control
A = cartpole.sys.A;
B = cartpole.sys.B;
C = cartpole.sys.C;
UN(:,1) = B;
for k = 1:N
    if k~=1
    UN(:,k) = df*A^(k-1)*B;
    end
    VN(k,:) = df*C*A^(N-k);
end
%% Toeplitz Matrix 
c = zeros(N,1);
for i =1:N-1
    r(:,i) = df*C*A^(i-1)*B;
end
r = [zeros(1,1) r];
TN = toeplitz(c,r);
TN = TN(1:N,1:N);
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
VNplus = inv(VN'*VN)*VN';
Mo = A^N*VNplus;
My = Mo;
Mu = UN-Mo*TN;