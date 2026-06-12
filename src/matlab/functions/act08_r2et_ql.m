function y = act08_r2et_ql(x)
%ACT08_R2ET_QL Politica greedy de Q-Learning tabular para R2ET.
% Entrada: x = [e1; e2; d]. Salida: y = [u1; u2; r1].

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end
e1 = x(1);
e2 = x(2);
d = x(3);

persistent Q actions
if isempty(Q)
    actions = [
        0.40 0.50
        0.50 0.50
        0.62 0.42
        0.72 0.36
        0.82 0.30
        0.92 0.24
        0.76 0.24
        0.88 0.20];
    Q = local_build_q_table(actions);
end

s = local_encode_state(e1, e2, d);
[~, a] = max(Q(s, :));
speed = actions(a, 1);
splitMagnitude = actions(a, 2);

u1 = local_clip(speed + 0.16*max(e1, 0), 0.05, 1.0);
u2 = local_clip(speed + 0.16*max(e2, 0), 0.05, 1.0);

balance = e1 - e2;
if balance > 0.12
    r1 = splitMagnitude;
elseif balance < -0.12
    r1 = 1 - splitMagnitude;
else
    r1 = 0.50;
end

y = [u1; u2; local_clip(r1, 0.20, 0.80)];
end

function Q = local_build_q_table(actions)
nStates = 4 * 5 * 2;
Q = zeros(nStates, size(actions, 1));
for s = 1:nStates
    [riskBin, balanceBin, distBin] = local_decode_state(s);
    for a = 1:size(actions, 1)
        speed = actions(a, 1);
        splitMag = actions(a, 2);
        riskCost = abs(speed - (0.42 + 0.12*riskBin + 0.08*(distBin - 1)));
        balanceCost = abs(splitMag - (0.50 - 0.06*(balanceBin - 3)));
        energyCost = 0.05*speed;
        Q(s, a) = -(riskCost + balanceCost + energyCost);
    end
end
end

function s = local_encode_state(e1, e2, d)
risk = max(e1, e2) + 1.5;
balance = e1 - e2;
if risk < 1.65
    riskBin = 1;
elseif risk < 2.40
    riskBin = 2;
elseif risk < 2.70
    riskBin = 3;
else
    riskBin = 4;
end
if balance < -0.75
    balanceBin = 1;
elseif balance < -0.20
    balanceBin = 2;
elseif balance <= 0.20
    balanceBin = 3;
elseif balance <= 0.75
    balanceBin = 4;
else
    balanceBin = 5;
end
distBin = 1 + double(d > 0.5);
s = riskBin + 4*(balanceBin - 1) + 4*5*(distBin - 1);
end

function [riskBin, balanceBin, distBin] = local_decode_state(s)
s0 = s - 1;
riskBin = mod(s0, 4) + 1;
balanceBin = mod(floor(s0 / 4), 5) + 1;
distBin = floor(s0 / (4*5)) + 1;
end

function y = local_clip(x, lo, hi)
y = min(max(x, lo), hi);
end
