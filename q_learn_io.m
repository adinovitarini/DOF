function [u_q,info] = q_learn_io(N,u,y,Q,R,df,u_bar,y_bar)
 
for i =1:N
    z_bar(:,i) = [u(i);u_bar(1:N-1);y_bar];
end
tic 
% Q = 1*eye(3);
% R = 0.1;
iter = 1;
u_bar = u_bar(1:N-1,1);
y_bar = y_bar(1:N,1);
P = zeros(202,202);
m = size(u,1); %dimensi input 
p = size(y,1); %dimensi output
p0 = zeros(m,m);
pu = zeros(m,m*(N-1));
py = zeros(m,p*N);
p22 = zeros(m*(N-1),m*(N-1));
p23 = zeros(m*(N-1),p*N); 
p32 = zeros(p*N,m*(N-1));
p33 = zeros(p*N,p*N);
P = [p0 pu py;pu' p22 p23;py' p32 p33];
u = zeros(iter,N);
    for i = 1:N-1
        if i~=1
            %% Policy Evaluation 
            P(:,:,i) = y(i)^2*Q+u(i)^2*R+df*z_bar(:,i)*z_bar(:,i)'*P(:,:,i-1);
            %% Policy Improvement
            p0(i) = P(1,1,i);
%             py = P(1,p*N,j);
            pu(i,:) = P(1,2:N,i);
            py(i,:) = P(1,N+1:end,i);
%             pu = P(1,m*(N-1),j);
            u(i+1) = -inv(R+p0(i))*(pu(i,:)*u_bar+py(i,:)*y_bar);
        end
    end
for i = 1:N-1
    P_norm(i) = norm(P(:,:,i));
end
time_elapsed = toc;
info.time = time_elapsed;
info.kernelP = P;
info.normP = P_norm;
u_q = u;