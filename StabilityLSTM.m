%%  Check Stability LSTM 
% if norm(Wi) and norm(Wo) < (1-norm(f)_\infty), ...
% norm(Wz)<0.25*1-norm(f)_\infty,...
% norm(f)<(1-norm(f)_\infty)^2,...
% r = O(log(d)) then \phi_lstm is stable
function status = StabilityLSTM(net,y)
nh = net.Layers(2).NumHiddenUnits;
Wi = net.Layers(2).InputWeights(1:nh,1);
Wf = net.Layers(2).InputWeights(nh+1:2*nh,1);
Wc = net.Layers(2).InputWeights(2*nh+1:3*nh,1);
Wo = net.Layers(2).InputWeights(3*nh+1:4*nh,1);
Ri = net.Layers(2).RecurrentWeights(1:nh,1);
Rf = net.Layers(2).RecurrentWeights(nh+1:2*nh,1);
Rc = net.Layers(2).RecurrentWeights(2*nh+1:3*nh,1);
Ro = net.Layers(2).RecurrentWeights(3*nh+1:4*nh,1);
h = zeros(nh,1);
% for i = 1:N
    inputGate = 1./(1+exp((Wi.*y+Ri.*h)));
    forgetGate = 1./(1+exp((Wf.*y+Rf.*h)));
    cellGate = tanh(Wc.*y+Rc.*h);
    outGate = 1./(1+exp((Wo.*y+Ro.*h)));
    cellGate = inputGate.*forgetGate.*cellGate.*outGate;
    h = outGate.*tanh(cellGate);
% end
f_inf = abs(norm(forgetGate,inf));
if norm(Wi,inf)<f_inf&&norm(Wo,inf)<f_inf&&norm(Wc)<0.25*f_inf
    status = 1;
else
    status = 0;
end