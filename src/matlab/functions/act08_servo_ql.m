function u = act08_servo_ql(x)
%ACT08_SERVO_QL Politica greedy de Q-Learning tabular para el lazo servo.
% Entrada: x = [e; de; d]. Salida: accion de velocidad u.

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end

persistent Q actions
if isempty(Q)
    actions = linspace(0.08, 1.0, 8);
    Q = local_build_q_table(actions);
end

s = local_encode_state(x(1), x(2), x(3));
[~, a] = max(Q(s, :));
u = actions(a);
end

function Q = local_build_q_table(actions)
nStates = 7 * 5 * 2;
Q = zeros(nStates, numel(actions));
for s = 1:nStates
    [ie, ide, id] = local_decode_state(s);
    eCenter = linspace(-0.6, 0.6, 7);
    deCenter = linspace(-0.25, 0.25, 5);
    target = 0.48 + 0.75*eCenter(ie) + 0.20*deCenter(ide) + 0.12*(id - 1);
    for a = 1:numel(actions)
        Q(s, a) = -abs(actions(a) - target) - 0.08*actions(a);
    end
end
end

function s = local_encode_state(e, de, d)
eEdges = [-inf -0.45 -0.25 -0.08 0.08 0.25 0.45 inf];
deEdges = [-inf -0.16 -0.04 0.04 0.16 inf];
ie = find(e < eEdges(2:end), 1, 'first');
ide = find(de < deEdges(2:end), 1, 'first');
id = 1 + double(d > 0.5);
s = ie + 7*(ide - 1) + 7*5*(id - 1);
end

function [ie, ide, id] = local_decode_state(s)
s0 = s - 1;
ie = mod(s0, 7) + 1;
ide = mod(floor(s0 / 7), 5) + 1;
id = floor(s0 / (7*5)) + 1;
end
