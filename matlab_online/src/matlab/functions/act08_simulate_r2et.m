function result = act08_simulate_r2et(cfg)
%ACT08_SIMULATE_R2ET Simula la celda R2ET supervisada en MATLAB.
% Devuelve ocupaciones, mandos y metricas basicas del escenario nominal.

if nargin < 1 || isempty(cfg)
    cfg = act08_config_grupo1();
end

N = cfg.n_steps;
n = zeros(N, 2);
u = zeros(N, 2);
r1 = zeros(N, 1);
adm = ones(N, 1);
n(1, :) = [1.25, 1.25];

for k = 2:N
    lambda = cfg.inflow_base + 0.35 * double(k >= 55 && k <= 82);
    d = double(k >= 95 && k <= 118);
    err = n(k-1, :) - cfg.target;
    primary = act08_r2et_fuzzy([err(1); err(2); d]);
    supervisor = act08_supervisor_fuzzy([n(k-1, 1); n(k-1, 2); d]);

    u(k, :) = min(max(primary(1:2)' .* supervisor(1), cfg.speed_min), cfg.speed_max);
    r1(k) = supervisor(2);
    adm(k) = supervisor(3);

    loss = [cfg.loss_e1, cfg.loss_e2] * d;
    feed = lambda * adm(k) * [r1(k), 1-r1(k)];
    outflow = cfg.exit_gain .* u(k, :) .* (1 - loss);
    n(k, :) = max(0, n(k-1, :) + feed - outflow);
end

result.t = (0:N-1)';
result.n = n;
result.u = u;
result.r1 = r1;
result.admission = adm;
result.iae = sum(abs(n(:,1) - cfg.target) + abs(n(:,2) - cfg.target));
result.max_occupancy = max(n, [], 'all');
result.violations = sum(n(:,1) > cfg.capacity) + sum(n(:,2) > cfg.capacity);
result.alarm_samples = sum(n(:,1) > cfg.alarm) + sum(n(:,2) > cfg.alarm);
end
