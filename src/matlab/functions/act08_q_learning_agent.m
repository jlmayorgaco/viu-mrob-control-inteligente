function agent = act08_q_learning_agent(cfg)
%ACT08_Q_LEARNING_AGENT Trains a small tabular Q-learning policy for R2ET.
% The state uses risk, balance, and disturbance bins. Actions combine belt
% speed, robot split intensity, and admission of incoming pieces.

rng(cfg.randomSeed + 17);

actions = [
    0.40, 0.50, 1.00;  % low smooth speed, balanced, full admission
    0.50, 0.50, 1.00;  % nominal
    0.62, 0.42, 0.98;  % moderate acceleration
    0.72, 0.36, 0.94;  % high speed, soft diversion
    0.82, 0.30, 0.90;  % high speed, stronger diversion
    0.92, 0.24, 0.84;  % emergency bounded speed
    0.76, 0.24, 0.92;  % balanced safety action
    0.88, 0.20, 0.86   % hard diversion without full saturation
];

nRisk = 4;
nBalance = 5;
nDist = 2;
nStates = nRisk * nBalance * nDist;
nActions = size(actions, 1);
Q = zeros(nStates, nActions);

for episode = 1:cfg.qlearning.episodes
    n = [0.8 + 1.2 * rand(), 0.8 + 1.2 * rand()];
    prevN = n;
    prevU = [cfg.baseSpeed, cfg.baseSpeed];
    prevAdmission = 1.0;
    epsilon = max(cfg.qlearning.epsilonMin, ...
        cfg.qlearning.epsilon0 * (1 - episode / cfg.qlearning.episodes));

    for k = 1:cfg.qlearning.horizon
        disturbance = double(rand() > 0.82 || (k >= 42 && k <= 54 && rand() > 0.35));
        demand = cfg.inflowBase + cfg.inflowPulse * double(k >= 25 && k <= 42);

        s = encode_state(n, disturbance);
        if rand() < epsilon
            a = randi(nActions);
        else
            [~, a] = max(Q(s, :));
        end

        [nNext, reward, stepU, stepAdmission] = q_step( ...
            cfg, n, prevN, actions(a, :), demand, disturbance, prevU, prevAdmission);
        sNext = encode_state(nNext, disturbance);
        Q(s, a) = Q(s, a) + cfg.qlearning.alpha * ...
            (reward + cfg.qlearning.gamma * max(Q(sNext, :)) - Q(s, a));

        prevN = n;
        prevU = stepU;
        prevAdmission = stepAdmission;
        n = nNext;
    end
end

agent.Q = Q;
agent.actions = actions;
agent.encodeState = @encode_state;
end

function [nNext, reward, u, admission] = q_step(cfg, n, prevN, action, demand, disturbance, prevU, prevAdmission)
err = n - cfg.targetOccupancy;
derr = n - prevN;
[u, split, admission] = q_control_from_action(cfg, action, n, err, derr, disturbance, prevU, prevAdmission);

feed = demand .* admission .* split;
loss = [cfg.disturbanceLoss * disturbance, 0.50 * cfg.disturbanceLoss * disturbance];
commandedOutflow = cfg.exitGain .* u .* (1 - loss);
outflow = min(commandedOutflow, n + feed);
nNext = n + feed - outflow;
nNext = min(max(nNext, 0), cfg.capacity + 0.4);

violation = sum(max(0, nNext - cfg.capacity));
risk = max(0, max(nNext) - cfg.supervisor.riskThreshold);
balancePenalty = abs(nNext(1) - nNext(2));
energy = sum(u);
throughput = sum(outflow);
targetError = sum(abs(nNext - cfg.targetOccupancy));
targetErrorL2 = sum((nNext - cfg.targetOccupancy) .^ 2);
overshoot = sum(max(0, nNext - (cfg.targetOccupancy + 0.22)) .^ 2);
undershoot = sum(max(0, (cfg.targetOccupancy - 0.18) - nNext) .^ 2);
du = sum(abs(u - prevU));
duExcess = sum(max(0, abs(u - prevU) - 0.09) .^ 2);
saturation = sum(max(0, u - 0.92) .^ 2);

reward = 1.35 * throughput ...
    - 18.0 * violation ...
    - 5.0 * risk ...
    - 1.30 * targetError ...
    - 1.15 * targetErrorL2 ...
    - 7.0 * overshoot ...
    - 0.70 * undershoot ...
    - 0.30 * energy ...
    - 0.55 * balancePenalty ...
    - 0.90 * max(0, 1 - admission) ...
    - 2.10 * du ...
    - 8.0 * duExcess ...
    - 4.50 * saturation;
end

function [u, split, admission] = q_control_from_action(cfg, action, n, err, derr, disturbance, prevU, prevAdmission)
speed = action(1);
splitMagnitude = action(2);
admission = action(3);

balance = n(1) - n(2);
if balance > 0.12
    split = [splitMagnitude, 1 - splitMagnitude];
elseif balance < -0.12
    split = [1 - splitMagnitude, splitMagnitude];
else
    split = [0.50, 0.50];
end

risk = max(n);
u = [speed, speed] + 0.16 .* max(err, 0) + 0.025 .* max(derr, 0);
if risk > 2.25
    u = u + 0.04;
end
if risk > 2.55
    u = u + 0.05;
    admission = min(admission, 0.88);
end
if disturbance > 0
    u = u + 0.04;
    admission = min(admission, 0.90);
end
u = min(max(u, cfg.speedMin), cfg.qlearning.speedMax);

if max(prevU) > 0.01
    rateUp = cfg.qlearning.rateUp;
    if risk > 2.35 || disturbance > 0
        rateUp = rateUp + 0.035;
    end
    u = min(u, prevU + rateUp);
    u = max(u, prevU - cfg.qlearning.rateDown);
end
admission = max(prevAdmission - 0.08, min(prevAdmission + 0.05, admission));
u = min(max(u, cfg.speedMin), cfg.qlearning.speedMax);
end

function s = encode_state(n, disturbance)
risk = max(n);
balance = n(1) - n(2);

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

distBin = 1 + double(disturbance > 0);
s = riskBin + 4 * (balanceBin - 1) + 4 * 5 * (distBin - 1);
end
