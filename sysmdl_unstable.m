function UnstableSys = sysmdl_unstable(N,df)
close all;clc
AA = [1.8 -0.77;1 0];
BB = [1;0];
CC = [1 -0.5];
D = 0;
A = sqrt(df)*AA;
B = sqrt(df)*BB;
C = sqrt(df)*CC;
w = wgn(N,4,0.01);
v = wgn(N,1,0.01);
sys = ss(A,B,C,D,0.01);
UnstableSys.sys = sys;
end