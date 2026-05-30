% act08_config_grupo1.m
% Configuration for Activity 08 MROB, Group 1, R2ET case
% Four methods: PID, Fuzzy, Deep Learning, Q-Learning
function cfg = act08_config_grupo1()
    cfg.capacity    = 3.0;
    cfg.target      = 1.5;
    cfg.n_steps     = 160;
    cfg.inflow_base = 0.42;
    cfg.exit_gain   = 0.46;
    cfg.risk        = 2.4;
    cfg.alarm       = 2.7;
    cfg.seed        = 8;
    % PID
    cfg.pid.kp = 0.52;  cfg.pid.ki = 0.028;  cfg.pid.kd = 0.10;
    % Q-Learning
    cfg.ql.n_states_servo = 100;
    cfg.ql.n_states_aware = 98;
    cfg.ql.n_actions = 8;
    cfg.ql.alpha = 0.15;  cfg.ql.gamma = 0.92;
end
