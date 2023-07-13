function [iter,perfIndex] = PerfAnalysis(u_q,YN,Yref,Q,R)
perfIndex = sum(YN.^2*Q+u_q.^2*R);
iter = 0;
N = size(YN,1);
for i = 1:N 
    if (YN(i)-Yref<=1e-20)
        iter = i;
        break
    end
end