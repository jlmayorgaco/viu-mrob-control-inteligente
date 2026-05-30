function generar_simulink_act08_grupo1()
% generar_simulink_act08_grupo1.m
% =========================================================================
% Actividad 08 MROB - Grupo 1 - Caso R2ET
% Genera AUTOMÁTICAMENTE los modelos Simulink de la propuesta de control y
% exporta una captura PNG de cada uno a  figures/simulink/.
%
% USO (en MATLAB, con Simulink instalado):
%   >> cd <raíz del proyecto>/simulations
%   >> generar_simulink_act08_grupo1
%
% Resultado: PNGs en  <raíz>/figures/simulink/  que el informe LaTeX ya
% referencia mediante \artifactfigure (placeholders hasta que existan):
%   sl_pid_servo.png        Lazo de control servo digital con PID (Fase 1)
%   sl_control_directo.png  Arquitectura de control digital directo inteligente (4.2.c)
%   sl_supervisor_pid.png   Supervisor inteligente que optimiza el PID local (4.2.d)
%   sl_r2et_aware.png       Coordinación R2ET de dos esteras + supervisor (Fase 2)
%
% Parámetros de planta coherentes con src/python/v2_simulate.py:
%   Gp(s) asignada = (0.174 s + 0.3744)/(s^2 + 0.785 s + 0.3744)  (guía, Equipo 1)
%   Modelo discreto de 1er orden equivalente (ZOH, Ts = tau/12 = 0.2125 s):
%       G(z) = b z^-1 / (1 - a z^-1),  a = 0.92, b = 0.08, perturbación bd = 0.024
% =========================================================================

    if isempty(ver('simulink'))
        error('Simulink no está instalado. Este script requiere Simulink.');
    end

    % --- Rutas ---
    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);
    outDir  = fullfile(rootDir, 'figures', 'simulink');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    % --- Parámetros de planta (coherentes con el script Python) ---
    P.a   = 0.92;     % polo discreto
    P.b   = 0.08;     % ganancia de control (K_DC = 1.0)
    P.bd  = 0.024;    % ganancia de perturbación de carga
    P.Ts  = 0.2125;   % período de muestreo (s)
    P.ref = 0.60;     % referencia de operación normalizada
    P.num = sprintf('[0 %g]', P.b);
    P.den = sprintf('[1 %g]', -P.a);

    fprintf('Generando modelos Simulink para Act08 - Grupo 1 (R2ET)...\n');

    construir_pid_servo(outDir, P);
    construir_control_directo(outDir, P);
    construir_supervisor_pid(outDir, P);
    construir_r2et_aware(outDir, P);

    fprintf('\nListo. Capturas exportadas a:\n  %s\n', outDir);
    fprintf('Súbelas al repositorio; el informe LaTeX las incrusta automáticamente.\n');
end


% =========================================================================
%  MODELO 1 — Lazo servo digital con PID (Fase 1)
% =========================================================================
function construir_pid_servo(outDir, P)
    name = 'sl_pid_servo';
    nuevo_modelo(name);

    add_block('simulink/Sources/Step',                 [name '/Referencia'], ...
        'Time', '2', 'After', num2str(P.ref), 'Before', '0', 'SampleTime', num2str(P.Ts));
    add_block('simulink/Math Operations/Sum',          [name '/Error'], 'Inputs', '+-');
    add_block('simulink/Discrete/Discrete PID Controller', [name '/PID']);
    add_block('simulink/Discontinuities/Saturation',   [name '/Actuador'], ...
        'UpperLimit', '1.0', 'LowerLimit', '0.05');
    add_block('simulink/Discrete/Discrete Transfer Fcn', [name '/Planta (estera)'], ...
        'Numerator', P.num, 'Denominator', P.den, 'SampleTime', num2str(P.Ts));
    add_block('simulink/Sources/Step',                 [name '/Perturbacion d'], ...
        'Time', '18', 'After', '1', 'Before', '0', 'SampleTime', num2str(P.Ts));
    add_block('simulink/Math Operations/Gain',         [name '/Carga (-bd)'], ...
        'Gain', num2str(-P.bd));
    add_block('simulink/Math Operations/Sum',          [name '/Suma planta'], 'Inputs', '++');
    add_block('simulink/Sinks/Scope',                  [name '/Nivel y(k)']);

    conectar(name, {
        'Referencia/1',     'Error/1'
        'Error/1',          'PID/1'
        'PID/1',            'Actuador/1'
        'Actuador/1',       'Suma planta/1'
        'Perturbacion d/1', 'Carga (-bd)/1'
        'Carga (-bd)/1',    'Suma planta/2'
        'Suma planta/1',    'Planta (estera)/1'
        'Planta (estera)/1','Nivel y(k)/1'
        'Planta (estera)/1','Error/2'
    });

    anotar(name, 'Lazo servo digital (Fase 1): PID discreto + planta ZOH de Gp(s) + perturbacion de carga.');
    exportar(name, outDir);
end


% =========================================================================
%  MODELO 2 — Arquitectura de control digital directo inteligente (4.2.c)
% =========================================================================
function construir_control_directo(outDir, P)
    name = 'sl_control_directo';
    nuevo_modelo(name);

    add_block('simulink/Sources/Step',               [name '/Referencia'], ...
        'Time', '2', 'After', num2str(P.ref), 'Before', '0', 'SampleTime', num2str(P.Ts));
    add_block('simulink/Math Operations/Sum',        [name '/Error'], 'Inputs', '+-');
    % Controlador inteligente directo como subsistema (Fuzzy/DL/QL intercambiable)
    add_block('simulink/Ports & Subsystems/Subsystem', [name '/Controlador inteligente directo']);
    poblar_controlador_directo([name '/Controlador inteligente directo']);
    add_block('simulink/Discontinuities/Saturation', [name '/Actuador'], ...
        'UpperLimit', '1.0', 'LowerLimit', '0.05');
    add_block('simulink/Discrete/Discrete Transfer Fcn', [name '/Planta estera'], ...
        'Numerator', P.num, 'Denominator', P.den, 'SampleTime', num2str(P.Ts));
    add_block('simulink/Sinks/Scope',                [name '/Salida']);

    conectar(name, {
        'Referencia/1', 'Error/1'
        'Error/1',      'Controlador inteligente directo/1'
        'Controlador inteligente directo/1', 'Actuador/1'
        'Actuador/1', 'Planta estera/1'
        'Planta estera/1', 'Salida/1'
        'Planta estera/1', 'Error/2'
    });

    anotar(name, ['Control digital directo (4.2.c): fuzzificacion -> inferencia AND-min -> ' ...
                  'defuzzificacion WA singletons. Bloque intercambiable Fuzzy/DL/QL.']);
    exportar(name, outDir);
end

function poblar_controlador_directo(sub)
    % Interior del subsistema: fuzzificación -> inferencia -> defuzzificación.
    delete_line_all(sub);
    blk = get_param(sub, 'Blocks');
    for i = 1:numel(blk); delete_block([sub '/' blk{i}]); end
    add_block('simulink/Ports & Subsystems/In1',  [sub '/e(k)']);
    add_block('simulink/Signal Routing/Demux',    [sub '/Estado']);  % e, de
    add_block('simulink/User-Defined Functions/Fcn', [sub '/Fuzzificacion'], 'Expr', 'u(1)');
    add_block('simulink/User-Defined Functions/Fcn', [sub '/Inferencia (AND=min)'], 'Expr', 'u(1)');
    add_block('simulink/User-Defined Functions/Fcn', [sub '/Defuzzificacion (WA)'], 'Expr', 'u(1)');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u(k)']);
    try
        conectar_raw(sub, {
            'e(k)/1', 'Fuzzificacion/1'
            'Fuzzificacion/1', 'Inferencia (AND=min)/1'
            'Inferencia (AND=min)/1', 'Defuzzificacion (WA)/1'
            'Defuzzificacion (WA)/1', 'u(k)/1'
        });
        delete_block([sub '/Estado']);
    catch
    end
end


% =========================================================================
%  MODELO 3 — Supervisor inteligente que optimiza el PID local (4.2.d)
% =========================================================================
function construir_supervisor_pid(outDir, P)
    name = 'sl_supervisor_pid';
    nuevo_modelo(name);

    add_block('simulink/Sources/Constant',           [name '/Nivel objetivo n*'], 'Value', '1.5');
    add_block('simulink/Math Operations/Sum',        [name '/Error ocupacion'], 'Inputs', '+-');
    add_block('simulink/Discrete/Discrete PID Controller', [name '/PID local']);
    % Supervisor (Nivel 2): ajusta ganancias + admision + reparto
    add_block('simulink/Ports & Subsystems/Subsystem', [name '/Supervisor inteligente']);
    poblar_supervisor([name '/Supervisor inteligente']);
    add_block('simulink/Discontinuities/Saturation', [name '/Velocidad estera'], ...
        'UpperLimit', '1.0', 'LowerLimit', '0.05');
    add_block('simulink/Discrete/Discrete Transfer Fcn', [name '/Estera R2ET'], ...
        'Numerator', P.num, 'Denominator', P.den, 'SampleTime', num2str(P.Ts));
    add_block('simulink/Sinks/Scope',                [name '/Ocupacion n(k)']);

    conectar(name, {
        'Nivel objetivo n*/1', 'Error ocupacion/1'
        'Error ocupacion/1',   'PID local/1'
        'PID local/1',         'Supervisor inteligente/1'
        'Supervisor inteligente/1', 'Velocidad estera/1'
        'Velocidad estera/1',  'Estera R2ET/1'
        'Estera R2ET/1',       'Ocupacion n(k)/1'
        'Estera R2ET/1',       'Error ocupacion/2'
    });

    anotar(name, ['Supervisor inteligente (4.2.d): ajusta Kp/Ki del PID por regimen ' ...
                  '(normal/riesgo/alarma), reduce admision y redirige el robot. ' ...
                  'Capa dura de capacidad siempre activa.']);
    exportar(name, outDir);
end

function poblar_supervisor(sub)
    blk = get_param(sub, 'Blocks');
    for i = 1:numel(blk); delete_block([sub '/' blk{i}]); end
    add_block('simulink/Ports & Subsystems/In1',  [sub '/u_PID']);
    add_block('simulink/User-Defined Functions/Fcn', [sub '/Regimen nmax d'], 'Expr', 'u(1)');
    add_block('simulink/User-Defined Functions/Fcn', [sub '/Ajuste Kp Ki admision'], 'Expr', 'u(1)');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u_r1_a']);
    try
        conectar_raw(sub, {
            'u_PID/1', 'Regimen nmax d/1'
            'Regimen nmax d/1', 'Ajuste Kp Ki admision/1'
            'Ajuste Kp Ki admision/1', 'u_r1_a/1'
        });
    catch
    end
end


% =========================================================================
%  MODELO 4 — Coordinación R2ET de dos esteras + supervisor (Fase 2)
% =========================================================================
function construir_r2et_aware(outDir, P)
    name = 'sl_r2et_aware';
    nuevo_modelo(name);

    add_block('simulink/Sources/Constant',  [name '/Objetivo n 1.5'], 'Value', '1.5');
    add_block('simulink/Ports & Subsystems/Subsystem', [name '/Controlador R2ET']);
    add_block('simulink/Ports & Subsystems/Subsystem', [name '/Supervisor capacidad']);
    add_block('simulink/Discrete/Discrete Transfer Fcn', [name '/Estera E1'], ...
        'Numerator', P.num, 'Denominator', P.den, 'SampleTime', num2str(P.Ts));
    add_block('simulink/Discrete/Discrete Transfer Fcn', [name '/Estera E2'], ...
        'Numerator', P.num, 'Denominator', P.den, 'SampleTime', num2str(P.Ts));
    add_block('simulink/Sinks/Scope',        [name '/Ocupacion n1 n2']);

    conectar(name, {
        'Objetivo n 1.5/1',      'Controlador R2ET/1'
        'Controlador R2ET/1',    'Supervisor capacidad/1'
        'Supervisor capacidad/1','Estera E1/1'
        'Supervisor capacidad/1','Estera E2/1'
        'Estera E1/1',           'Ocupacion n1 n2/1'
    });

    anotar(name, ['Coordinacion R2ET (Fase 2): controlador genera u1,u2,reparto r1; ' ...
                  'supervisor de capacidad (barrera de seguridad) sobre dos esteras (cap. 3 piezas).']);
    exportar(name, outDir);
end


% =========================================================================
%  UTILIDADES
% =========================================================================
function nuevo_modelo(name)
    if bdIsLoaded(name); close_system(name, 0); end
    new_system(name);
    load_system(name);
end

function conectar(name, pares)
    for i = 1:size(pares, 1)
        try
            add_line(name, pares{i, 1}, pares{i, 2}, 'autorouting', 'on');
        catch err
            warning('No se pudo conectar %s -> %s (%s)', pares{i,1}, pares{i,2}, err.message);
        end
    end
end

function conectar_raw(sub, pares)
    for i = 1:size(pares, 1)
        add_line(sub, pares{i, 1}, pares{i, 2}, 'autorouting', 'on');
    end
end

function delete_line_all(sys)
    try
        lh = find_system(sys, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
        for i = 1:numel(lh); delete_line(lh(i)); end
    catch
    end
end

function anotar(name, txt)
    try
        add_block('built-in/Note', [name '/nota'], 'Position', [30 20]);
        set_param([name '/nota'], 'Text', txt);
    catch
        % Las anotaciones difieren entre versiones; no es crítico.
    end
end

function exportar(name, outDir)
    % Auto-organiza el diagrama y exporta una captura PNG.
    try
        Simulink.BlockDiagram.arrangeSystem(name);   % R2018b+
    catch
    end
    out = fullfile(outDir, [name '.png']);
    ok = false;
    try
        print(['-s' name], '-dpng', '-r150', out); ok = true;   % captura del diagrama
    catch
    end
    if ~ok
        try
            saveas(get_param(name, 'Handle'), out); ok = true;
        catch err
            warning('No se pudo exportar %s: %s', name, err.message);
        end
    end
    if ok; fprintf('  [OK] %s\n', out); end
    close_system(name, 0);
end
