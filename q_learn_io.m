function [u_q,z_bar,info] = q_learn_io(N,u,y,Q,R,df,u_bar,y_bar)
 %%
m = size(u,1); %dimensi input 
p = size(y,1); %dimensi output
% for i =1:N
%     z_bar(:,i) = [u(i);y(i)];
% end
for i =1:N
    Z_bar(:,i) = [u(i);u_bar(1:N-1);y_bar];
end
% z_bar = z_bar(:,1);
z_bar = reshape(Z_bar(:,1),[N,m+p])';
%%
tic 
% Q = 1*eye(3);
% R = 0.1;
iter = 1;
u_bar = u_bar(1:N-1,1);
y_bar = y_bar(1:N,1);
P = eye(202,202);
p0 = ones(m,m);
pu = ones(m,m*(N-1));
py = ones(m,p*N);
p22 = ones(m*(N-1),m*(N-1));
p23 = ones(m*(N-1),p*N); 
p32 = ones(p*N,m*(N-1));
p33 = ones(p*N,p*N);
P = [p0 pu py;pu' p22 p23;py' p32 p33];
u = zeros(iter,N);
    for i = 1:N-1
        if i~=1
            % Policy Evaluation 
            P(:,:,i) = y(i)^2*Q+u(i)^2*R+df*Z_bar(:,i)*Z_bar(:,i)'*P(:,:,i-1);
            % Policy Improvement
            p0(i) = P(1,1,i);
%             py = P(1,p*N,j);
            pu(i,:) = P(1,2:N,i);
            py(i,:) = P(1,N+1:end,i);
%             pu = P(1,m*(N-1),j);
            u(i+1) = -df*inv(R+p0(i))*(pu(i,:)*u_bar+py(i,:)*y_bar);
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