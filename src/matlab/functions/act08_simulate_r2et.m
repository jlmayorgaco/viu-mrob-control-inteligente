% act08_simulate_r2et.m
% R2ET simulation for Activity 08 MROB, Group 1
% Methods: PID, Fuzzy, Deep Learning, Q-Learning
% Main simulation is in src/python/v2_simulate.py
function result = act08_simulate_r2et(cfg)
    if nargin < 1
        cfg = struct();
    end

    repoRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
    currentDir = pwd;
    cleanupObj = onCleanup(@() cd(currentDir));
    cd(repoRoot);
    repoRoot = pwd;
    clear cleanupObj;
    pyScript = fullfile(repoRoot, 'src', 'python', 'v2_simulate.py');

    if ~isfile(pyScript)
        error('No se encontro el generador Python: %s', pyScript);
    end

    cmd = sprintf('python "%s"', pyScript);
    [status, output] = system(cmd);
    if status ~= 0
        error('La simulacion R2ET fallo:\n%s', output);
    end

    result = struct();
    result.config = cfg;
    result.generator = pyScript;
    result.output = output;
    result.artifacts = fullfile(repoRoot, 'artifacts');
end
