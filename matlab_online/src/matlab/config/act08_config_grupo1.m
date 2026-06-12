% act08_config_grupo1.m
% Configuration for Activity 08 MROB, Group 1, R2ET case
% Four methods: PID, Fuzzy, Deep Learning, Q-Learning
function cfg = act08_config_grupo1()
    cfg.capacity    = 3.0;
    cfg.target      = 1.5;
    cfg.n_steps     = 160;
    cfg.inflow_base = 0.42;
    cfg.exit_gain   = 0.46;
    cfg.loss_e1     = 0.35;
    cfg.loss_e2     = 0.175;
    cfg.risk        = 2.4;
    cfg.alarm       = 2.7;
    cfg.seed        = 8;
    cfg.u_ss        = 0.457;
    cfg.speed_min   = 0.05;
    cfg.speed_max   = 1.0;
    cfg.split_min   = 0.20;
    cfg.split_max   = 0.80;
    % PID supervisado de Fase 2
    cfg.pid.kp = 1.30;  cfg.pid.ki = 0.18;  cfg.pid.kd = 0.10;

    % Campos de compatibilidad para las funciones auxiliares.
    cfg.randomSeed = cfg.seed;
    cfg.targetOccupancy = cfg.target;
    cfg.inflowBase = cfg.inflow_base;
    cfg.inflowPulse = 0.35;
    cfg.exitGain = cfg.exit_gain;
    cfg.disturbanceLoss = cfg.loss_e1;
    cfg.baseSpeed = cfg.u_ss;
    cfg.speedMin = cfg.speed_min;
    cfg.speedMax = cfg.speed_max;
    cfg.supervisor.riskThreshold = cfg.risk;

    % Q-Learning
    cfg.ql.n_states_servo = 100;
    cfg.ql.n_states_aware = 98;
    cfg.ql.n_actions = 8;
    cfg.ql.alpha = 0.15;  cfg.ql.gamma = 0.92;

    cfg.qlearning.episodes = 3000;
    cfg.qlearning.horizon = 160;
    cfg.qlearning.epsilon0 = 0.35;
    cfg.qlearning.epsilonMin = 0.03;
    cfg.qlearning.alpha = cfg.ql.alpha;
    cfg.qlearning.gamma = cfg.ql.gamma;
    cfg.qlearning.speedMax = cfg.speed_max;
    cfg.qlearning.rateUp = 0.08;
    cfg.qlearning.rateDown = 0.12;
end
