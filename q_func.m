%% Compute Q-Function 
function Q = q_func(model,reward)
Ry = model.Ry;
Ru = model.Ru;
df = model.df;
N = model.N;
r = reward;
for k = 1:N
    for i = k:N
        temp(i) = sum(df^(i-k)*r(i));
    end
    Q(k) = r(k)+temp(i);
end