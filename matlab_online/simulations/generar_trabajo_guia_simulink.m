function generar_trabajo_guia_simulink()
%GENERAR_TRABAJO_GUIA_SIMULINK
% Genera los modelos Simulink y las capturas usadas en la Actividad 08
% MROB, Grupo 1, caso R2ET. El programa es autocontenido: define los
% parametros de la planta, construye los diagramas de Fase 1, Fase 2 y
% supervisor, y exporta las imagenes con nombres neutros para el informe.
%
% Uso en MATLAB:
%   cd <raiz_del_proyecto>
%   addpath(genpath('simulations'))
%   generar_trabajo_guia_simulink

    close all; clc;
    assert(~isempty(ver('simulink')), 'Este generador requiere Simulink.');

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(rootDir, 'src', 'matlab')));
    addpath(genpath(fullfile(rootDir, 'simulations')));

    outDir = fullfile(rootDir, 'figures', 'simulink');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    P = parametros_r2et();

    fprintf('Generando modelos y capturas MATLAB/Simulink para R2ET...\n');
    generar_modelo_servo(P, outDir);
    generar_modelo_control_directo(P, outDir);
    generar_modelo_coordinacion(P, outDir);
    generar_modelo_supervisor(P, outDir);
    generar_graficas_de_validacion(P, outDir);

    fprintf('Capturas exportadas en: %s\n', outDir);
end

function P = parametros_r2et()
    P.a = 0.92;
    P.b = 0.08;
    P.bd = 0.024;
    P.Ts_servo = 0.2125;
    P.Ts_r2et = 1.0;
    P.ref_servo = 0.60;
    P.n_ref = 1.5;
    P.n_max = 3.0;
    P.lambda0 = 0.42;
    P.lambda_pulso = 0.35;
    P.eta = 0.46;
    P.loss1 = 0.35;
    P.loss2 = 0.175;
    P.risk = 2.4;
    P.alarm = 2.7;
    P.u_min = 0.05;
    P.u_max = 1.0;
    P.r_min = 0.20;
    P.r_max = 0.80;
end

function generar_modelo_servo(P, outDir)
    mdl = nuevo_modelo('act08_fase1_servo_pid');

    add_block('simulink/Sources/Step', [mdl '/Referencia'], ...
        'Time', '10', 'Before', '0', 'After', num2str(P.ref_servo), ...
        'SampleTime', num2str(P.Ts_servo));
    add_block('simulink/Math Operations/Sum', [mdl '/Error e = r - y'], 'Inputs', '+-');
    add_block('simulink/Discrete/Discrete PID Controller', [mdl '/PID discreto']);
    add_block('simulink/Discontinuities/Saturation', [mdl '/Saturacion actuador'], ...
        'LowerLimit', num2str(P.u_min), 'UpperLimit', num2str(P.u_max));
    add_block('simulink/Sources/Step', [mdl '/Perturbacion d'], ...
        'Time', '85', 'Before', '0', 'After', '1', 'SampleTime', num2str(P.Ts_servo));
    add_block('simulink/Math Operations/Gain', [mdl '/Carga -bd'], 'Gain', num2str(-P.bd));
    add_block('simulink/Math Operations/Sum', [mdl '/Entrada planta'], 'Inputs', '++');
    add_block('simulink/Discrete/Discrete Transfer Fcn', [mdl '/Planta servo ZOH'], ...
        'Numerator', sprintf('[0 %.12g]', P.b), ...
        'Denominator', sprintf('[1 %.12g]', -P.a), ...
        'SampleTime', num2str(P.Ts_servo));
    add_block('simulink/Sinks/Scope', [mdl '/Scope salida']);

    conectar(mdl, {
        'Referencia/1', 'Error e = r - y/1'
        'Error e = r - y/1', 'PID discreto/1'
        'PID discreto/1', 'Saturacion actuador/1'
        'Saturacion actuador/1', 'Entrada planta/1'
        'Perturbacion d/1', 'Carga -bd/1'
        'Carga -bd/1', 'Entrada planta/2'
        'Entrada planta/1', 'Planta servo ZOH/1'
        'Planta servo ZOH/1', 'Scope salida/1'
        'Planta servo ZOH/1', 'Error e = r - y/2'
    });

    nota(mdl, 'Fase 1: lazo servo con PID discreto, saturacion de actuador, planta ZOH y perturbacion de carga.');
    exportar_diagrama(mdl, outDir, 'sl_pid_servo.png');

    mdlPlant = nuevo_modelo('act08_planta_servo_zoh');
    add_block('simulink/Sources/In1', [mdlPlant '/u(k)']);
    add_block('simulink/Sources/In1', [mdlPlant '/d(k)']);
    add_block('simulink/Math Operations/Gain', [mdlPlant '/b'], 'Gain', num2str(P.b));
    add_block('simulink/Math Operations/Gain', [mdlPlant '/-bd'], 'Gain', num2str(-P.bd));
    add_block('simulink/Discrete/Unit Delay', [mdlPlant '/z^-1'], 'SampleTime', num2str(P.Ts_servo));
    add_block('simulink/Math Operations/Gain', [mdlPlant '/a'], 'Gain', num2str(P.a));
    add_block('simulink/Math Operations/Sum', [mdlPlant '/y(k+1)'], 'Inputs', '+++');
    add_block('simulink/Ports & Subsystems/Out1', [mdlPlant '/y(k)']);
    conectar(mdlPlant, {
        'u(k)/1', 'b/1'
        'd(k)/1', '-bd/1'
        'b/1', 'y(k+1)/1'
        '-bd/1', 'y(k+1)/2'
        'z^-1/1', 'a/1'
        'a/1', 'y(k+1)/3'
        'y(k+1)/1', 'z^-1/1'
        'z^-1/1', 'y(k)/1'
    });
    nota(mdlPlant, 'Modelo discreto: y(k+1)=0.92 y(k)+0.08 u(k)-0.024 d(k).');
    exportar_diagrama(mdlPlant, outDir, 'sl_planta_servo.png');
end

function generar_modelo_control_directo(P, outDir)
    mdl = nuevo_modelo('act08_control_directo_inteligente');

    add_block('simulink/Sources/In1', [mdl '/Referencia r']);
    add_block('simulink/Sources/In1', [mdl '/Salida y']);
    add_block('simulink/Sources/In1', [mdl '/Perturbacion d']);
    add_block('simulink/Math Operations/Sum', [mdl '/Error'], 'Inputs', '+-');
    add_block('simulink/Discrete/Unit Delay', [mdl '/Retardo error'], 'SampleTime', num2str(P.Ts_servo));
    add_block('simulink/Math Operations/Sum', [mdl '/Delta error'], 'Inputs', '+-');
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Controlador inteligente directo']);
    poblar_control_directo([mdl '/Controlador inteligente directo']);
    add_block('simulink/Discontinuities/Saturation', [mdl '/Saturacion'], ...
        'LowerLimit', num2str(P.u_min), 'UpperLimit', num2str(P.u_max));
    add_block('simulink/Ports & Subsystems/Out1', [mdl '/u(k)']);

    conectar(mdl, {
        'Referencia r/1', 'Error/1'
        'Salida y/1', 'Error/2'
        'Error/1', 'Retardo error/1'
        'Error/1', 'Delta error/1'
        'Retardo error/1', 'Delta error/2'
        'Error/1', 'Controlador inteligente directo/1'
        'Delta error/1', 'Controlador inteligente directo/2'
        'Perturbacion d/1', 'Controlador inteligente directo/3'
        'Controlador inteligente directo/1', 'Saturacion/1'
        'Saturacion/1', 'u(k)/1'
    });

    nota(mdl, 'Control directo 4.2.c: bloque intercambiable Fuzzy, red profunda o Q-Learning.');
    exportar_diagrama(mdl, outDir, 'sl_control_directo.png');
    abrir_y_exportar([mdl '/Controlador inteligente directo'], outDir, 'sl_controlador_directo.png');
end

function poblar_control_directo(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/e']);
    add_block('simulink/Sources/In1', [sub '/Delta e']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Signal Routing/Mux', [sub '/x servo'], 'Inputs', '3');
    crear_bloque_funcion_matlab([sub '/Fuzzy Toolbox act08_servo_fuzzy'], 'act08_servo_fuzzy', 1);
    crear_bloque_funcion_matlab([sub '/Deep Learning act08_servo_dl'], 'act08_servo_dl', 1);
    crear_bloque_funcion_matlab([sub '/Q-Learning act08_servo_ql'], 'act08_servo_ql', 1);
    add_block('simulink/Sources/Constant', [sub '/modo 1 Fuzzy 2 DL 3 QL'], 'Value', '1');
    add_block('simulink/Signal Routing/Multiport Switch', [sub '/Selector metodo'], 'Inputs', '3');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u estrella']);
    conectar(sub, {
        'e/1', 'x servo/1'
        'Delta e/1', 'x servo/2'
        'd/1', 'x servo/3'
        'x servo/1', 'Fuzzy Toolbox act08_servo_fuzzy/1'
        'x servo/1', 'Deep Learning act08_servo_dl/1'
        'x servo/1', 'Q-Learning act08_servo_ql/1'
        'modo 1 Fuzzy 2 DL 3 QL/1', 'Selector metodo/1'
        'Fuzzy Toolbox act08_servo_fuzzy/1', 'Selector metodo/2'
        'Deep Learning act08_servo_dl/1', 'Selector metodo/3'
        'Q-Learning act08_servo_ql/1', 'Selector metodo/4'
        'Selector metodo/1', 'u estrella/1'
    });
    nota(sub, 'El selector conmuta entre funciones MATLAB: Fuzzy Toolbox, Deep Learning y Q-Learning.');
end

function poblar_fuzzificacion(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/e']);
    add_block('simulink/Sources/In1', [sub '/Delta e']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Signal Routing/Mux', [sub '/vector membresias'], 'Inputs', '3');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/mu']);
    conectar(sub, {
        'e/1', 'vector membresias/1'
        'Delta e/1', 'vector membresias/2'
        'd/1', 'vector membresias/3'
        'vector membresias/1', 'mu/1'
    });
    nota(sub, 'Funciones NB, NS, ZE, PS, PB para e y Delta e; NO/YES para perturbacion.');
end

function poblar_inferencia_minimo(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/mu']);
    add_block('simulink/Math Operations/Gain', [sub '/AND minimo y reglas'], 'Gain', '1');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/activaciones']);
    conectar(sub, {
        'mu/1', 'AND minimo y reglas/1'
        'AND minimo y reglas/1', 'activaciones/1'
    });
    nota(sub, '35 reglas linguisticas; agregacion por AND-minimo.');
end

function poblar_defuzzificacion_wa(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/activaciones']);
    add_block('simulink/Math Operations/Gain', [sub '/promedio ponderado singleton'], 'Gain', '1');
    add_block('simulink/Discontinuities/Saturation', [sub '/limite u'], ...
        'LowerLimit', '0.05', 'UpperLimit', '1.0');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u*']);
    conectar(sub, {
        'activaciones/1', 'promedio ponderado singleton/1'
        'promedio ponderado singleton/1', 'limite u/1'
        'limite u/1', 'u*/1'
    });
    nota(sub, 'Defuzzificacion por promedio ponderado de consecuentes singleton.');
end

function generar_modelo_coordinacion(P, outDir)
    mdl = nuevo_modelo('act08_fase2_coordinacion_r2et');

    add_block('simulink/Sources/Constant', [mdl '/n*'], 'Value', num2str(P.n_ref));
    add_block('simulink/Sources/Constant', [mdl '/lambda(k)'], 'Value', num2str(P.lambda0));
    add_block('simulink/Sources/Step', [mdl '/d(k)'], 'Time', '95', 'Before', '0', 'After', '1');
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Controlador primario']);
    poblar_controlador_r2et([mdl '/Controlador primario']);
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Supervisor capacidad']);
    poblar_supervisor_capacidad([mdl '/Supervisor capacidad']);
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Esteras R2ET']);
    poblar_esteras([mdl '/Esteras R2ET']);
    add_block('simulink/Sinks/Scope', [mdl '/Scope n1 n2'], 'NumInputPorts', '2');

    conectar(mdl, {
        'n*/1', 'Controlador primario/1'
        'Esteras R2ET/1', 'Controlador primario/2'
        'Esteras R2ET/2', 'Controlador primario/3'
        'd(k)/1', 'Controlador primario/4'
        'Controlador primario/1', 'Supervisor capacidad/1'
        'Controlador primario/2', 'Supervisor capacidad/2'
        'Controlador primario/3', 'Supervisor capacidad/3'
        'Esteras R2ET/1', 'Supervisor capacidad/4'
        'Esteras R2ET/2', 'Supervisor capacidad/5'
        'Supervisor capacidad/1', 'Esteras R2ET/1'
        'Supervisor capacidad/2', 'Esteras R2ET/2'
        'Supervisor capacidad/3', 'Esteras R2ET/3'
        'Supervisor capacidad/4', 'Esteras R2ET/4'
        'lambda(k)/1', 'Esteras R2ET/5'
        'd(k)/1', 'Esteras R2ET/6'
        'Esteras R2ET/1', 'Scope n1 n2/1'
        'Esteras R2ET/2', 'Scope n1 n2/2'
    });

    nota(mdl, 'Fase 2: coordinacion de dos esteras, reparto del robot, admision y barrera de capacidad.');
    exportar_diagrama(mdl, outDir, 'sl_r2et_coordinacion.png');
    abrir_y_exportar([mdl '/Controlador primario'], outDir, 'sl_r2et_controlador_primario.png');
    abrir_y_exportar([mdl '/Supervisor capacidad'], outDir, 'sl_r2et_supervisor_capacidad.png');
    abrir_y_exportar([mdl '/Esteras R2ET'], outDir, 'sl_r2et_esteras.png');
end

function poblar_controlador_r2et(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/n*']);
    add_block('simulink/Sources/In1', [sub '/n1']);
    add_block('simulink/Sources/In1', [sub '/n2']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Math Operations/Sum', [sub '/e1'], 'Inputs', '+-');
    add_block('simulink/Math Operations/Sum', [sub '/e2'], 'Inputs', '+-');
    add_block('simulink/Ports & Subsystems/Subsystem', [sub '/Politica PID Fuzzy DL QL']);
    poblar_politica_r2et([sub '/Politica PID Fuzzy DL QL']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u1*']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u2*']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1*']);
    conectar(sub, {
        'n1/1', 'e1/1'
        'n*/1', 'e1/2'
        'n2/1', 'e2/1'
        'n*/1', 'e2/2'
        'e1/1', 'Politica PID Fuzzy DL QL/1'
        'e2/1', 'Politica PID Fuzzy DL QL/2'
        'd/1', 'Politica PID Fuzzy DL QL/3'
        'Politica PID Fuzzy DL QL/1', 'u1*/1'
        'Politica PID Fuzzy DL QL/2', 'u2*/1'
        'Politica PID Fuzzy DL QL/3', 'r1*/1'
    });
    nota(sub, 'Controlador primario: genera velocidades u1,u2 y reparto r1 a partir del estado completo.');
end

function poblar_politica_r2et(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/e1']);
    add_block('simulink/Sources/In1', [sub '/e2']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Signal Routing/Mux', [sub '/x r2et'], 'Inputs', '3');
    crear_bloque_funcion_matlab([sub '/Fuzzy Toolbox act08_r2et_fuzzy'], 'act08_r2et_fuzzy', 3);
    crear_bloque_funcion_matlab([sub '/Deep Learning act08_r2et_dl'], 'act08_r2et_dl', 3);
    crear_bloque_funcion_matlab([sub '/Q-Learning act08_r2et_ql'], 'act08_r2et_ql', 3);
    add_block('simulink/Sources/Constant', [sub '/modo 1 Fuzzy 2 DL 3 QL'], 'Value', '1');
    add_block('simulink/Signal Routing/Multiport Switch', [sub '/Selector metodo'], 'Inputs', '3');
    add_block('simulink/Signal Routing/Demux', [sub '/u1 u2 r1'], 'Outputs', '3');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u1*']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u2*']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1*']);
    conectar(sub, {
        'e1/1', 'x r2et/1'
        'e2/1', 'x r2et/2'
        'd/1', 'x r2et/3'
        'x r2et/1', 'Fuzzy Toolbox act08_r2et_fuzzy/1'
        'x r2et/1', 'Deep Learning act08_r2et_dl/1'
        'x r2et/1', 'Q-Learning act08_r2et_ql/1'
        'modo 1 Fuzzy 2 DL 3 QL/1', 'Selector metodo/1'
        'Fuzzy Toolbox act08_r2et_fuzzy/1', 'Selector metodo/2'
        'Deep Learning act08_r2et_dl/1', 'Selector metodo/3'
        'Q-Learning act08_r2et_ql/1', 'Selector metodo/4'
        'Selector metodo/1', 'u1 u2 r1/1'
        'u1 u2 r1/1', 'u1*/1'
        'u1 u2 r1/2', 'u2*/1'
        'u1 u2 r1/3', 'r1*/1'
    });
    nota(sub, 'Politicas R2ET en funciones MATLAB externas: act08_r2et_fuzzy, act08_r2et_dl y act08_r2et_ql.');
end

function poblar_supervisor_capacidad(sub)
    limpiar_subsistema(sub);
    nombres = {'u1*','u2*','r1*','n1','n2'};
    for i = 1:numel(nombres)
        add_block('simulink/Sources/In1', [sub '/' nombres{i}]);
    end
    add_block('simulink/Ports & Subsystems/Subsystem', [sub '/Barrera de capacidad']);
    poblar_barrera_capacidad([sub '/Barrera de capacidad']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u2']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/admision']);
    conectar(sub, {
        'u1*/1', 'Barrera de capacidad/1'
        'u2*/1', 'Barrera de capacidad/2'
        'r1*/1', 'Barrera de capacidad/3'
        'n1/1', 'Barrera de capacidad/4'
        'n2/1', 'Barrera de capacidad/5'
        'Barrera de capacidad/1', 'u1/1'
        'Barrera de capacidad/2', 'u2/1'
        'Barrera de capacidad/3', 'r1/1'
        'Barrera de capacidad/4', 'admision/1'
    });
    nota(sub, 'Reglas: redirigir robot si una estera se acerca a capacidad; reducir admision si ambas estan llenas.');
end

function poblar_barrera_capacidad(sub)
    limpiar_subsistema(sub);
    nombres = {'u1*','u2*','r1*','n1','n2'};
    for i = 1:numel(nombres)
        add_block('simulink/Sources/In1', [sub '/' nombres{i}]);
    end
    add_block('simulink/Math Operations/MinMax', [sub '/nmax'], 'Function', 'max', 'Inputs', '2');
    add_block('simulink/Ports & Subsystems/Subsystem', [sub '/Reglas riesgo alarma']);
    poblar_reglas_barrera([sub '/Reglas riesgo alarma']);
    add_block('simulink/Math Operations/Sum', [sub '/u1 corregido'], 'Inputs', '++');
    add_block('simulink/Math Operations/Sum', [sub '/u2 corregido'], 'Inputs', '++');
    add_block('simulink/Discontinuities/Saturation', [sub '/limite u1'], ...
        'LowerLimit', '0.05', 'UpperLimit', '1.0');
    add_block('simulink/Discontinuities/Saturation', [sub '/limite u2'], ...
        'LowerLimit', '0.05', 'UpperLimit', '1.0');
    add_block('simulink/Discontinuities/Saturation', [sub '/limite r1'], ...
        'LowerLimit', '0.2', 'UpperLimit', '0.8');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u2']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/admision']);
    conectar(sub, {
        'n1/1', 'nmax/1'
        'n2/1', 'nmax/2'
        'nmax/1', 'Reglas riesgo alarma/1'
        'n1/1', 'Reglas riesgo alarma/2'
        'n2/1', 'Reglas riesgo alarma/3'
        'r1*/1', 'Reglas riesgo alarma/4'
        'u1*/1', 'u1 corregido/1'
        'Reglas riesgo alarma/1', 'u1 corregido/2'
        'u2*/1', 'u2 corregido/1'
        'Reglas riesgo alarma/1', 'u2 corregido/2'
        'u1 corregido/1', 'limite u1/1'
        'u2 corregido/1', 'limite u2/1'
        'Reglas riesgo alarma/2', 'limite r1/1'
        'limite u1/1', 'u1/1'
        'limite u2/1', 'u2/1'
        'limite r1/1', 'r1/1'
        'Reglas riesgo alarma/3', 'admision/1'
    });
    nota(sub, 'Barrera: aplica refuerzo de velocidad, limita reparto y reduce admision si hay riesgo.');
end

function poblar_reglas_barrera(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/nmax']);
    add_block('simulink/Sources/In1', [sub '/n1']);
    add_block('simulink/Sources/In1', [sub '/n2']);
    add_block('simulink/Sources/In1', [sub '/r1*']);
    add_block('simulink/Sources/Constant', [sub '/du normal riesgo alarma'], 'Value', '0.07');
    add_block('simulink/Sources/Constant', [sub '/admision segura'], 'Value', '0.90');
    add_block('simulink/Discontinuities/Saturation', [sub '/r1 seguro'], ...
        'LowerLimit', '0.2', 'UpperLimit', '0.8');
    add_block('simulink/Sinks/Terminator', [sub '/nmax usado por reglas']);
    add_block('simulink/Sinks/Terminator', [sub '/n1 usado por reglas']);
    add_block('simulink/Sinks/Terminator', [sub '/n2 usado por reglas']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/delta u']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1 corregido']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/admision']);
    conectar(sub, {
        'du normal riesgo alarma/1', 'delta u/1'
        'r1*/1', 'r1 seguro/1'
        'r1 seguro/1', 'r1 corregido/1'
        'admision segura/1', 'admision/1'
        'nmax/1', 'nmax usado por reglas/1'
        'n1/1', 'n1 usado por reglas/1'
        'n2/1', 'n2 usado por reglas/1'
    });
    nota(sub, 'La implementacion MATLAB aplica umbrales 2.4/2.7, redireccion y admision 0.90/0.72 segun riesgo.');
end

function poblar_esteras(sub)
    limpiar_subsistema(sub);
    nombres = {'u1','u2','r1','admision','lambda','d'};
    for i = 1:numel(nombres)
        add_block('simulink/Sources/In1', [sub '/' nombres{i}]);
    end
    add_block('simulink/Discrete/Unit Delay', [sub '/n1(k)'], 'InitialCondition', '1.25');
    add_block('simulink/Discrete/Unit Delay', [sub '/n2(k)'], 'InitialCondition', '1.25');
    add_block('simulink/Sources/Constant', [sub '/uno'], 'Value', '1');
    add_block('simulink/Math Operations/Sum', [sub '/1-r1'], 'Inputs', '+-');
    add_block('simulink/Math Operations/Product', [sub '/entrada E1'], 'Inputs', '***');
    add_block('simulink/Math Operations/Product', [sub '/entrada E2'], 'Inputs', '***');
    add_block('simulink/Math Operations/Gain', [sub '/eta E1'], 'Gain', '0.46');
    add_block('simulink/Math Operations/Gain', [sub '/eta E2'], 'Gain', '0.46');
    add_block('simulink/Math Operations/Sum', [sub '/balance E1'], 'Inputs', '++-');
    add_block('simulink/Math Operations/Sum', [sub '/balance E2'], 'Inputs', '++-');
    add_block('simulink/Discontinuities/Saturation', [sub '/no negativo E1'], 'LowerLimit', '0', 'UpperLimit', 'inf');
    add_block('simulink/Discontinuities/Saturation', [sub '/no negativo E2'], 'LowerLimit', '0', 'UpperLimit', 'inf');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/n1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/n2']);
    conectar(sub, {
        'lambda/1', 'entrada E1/1'
        'admision/1', 'entrada E1/2'
        'r1/1', 'entrada E1/3'
        'uno/1', '1-r1/1'
        'r1/1', '1-r1/2'
        'u1/1', 'eta E1/1'
        'n1(k)/1', 'balance E1/1'
        'entrada E1/1', 'balance E1/2'
        'eta E1/1', 'balance E1/3'
        'balance E1/1', 'no negativo E1/1'
        'no negativo E1/1', 'n1(k)/1'
        'n1(k)/1', 'n1/1'
        'lambda/1', 'entrada E2/1'
        'admision/1', 'entrada E2/2'
        '1-r1/1', 'entrada E2/3'
        'u2/1', 'eta E2/1'
        'n2(k)/1', 'balance E2/1'
        'entrada E2/1', 'balance E2/2'
        'eta E2/1', 'balance E2/3'
        'balance E2/1', 'no negativo E2/1'
        'no negativo E2/1', 'n2(k)/1'
        'n2(k)/1', 'n2/1'
    });
    nota(sub, 'Balance de masa sin recorte superior: la capacidad de 3 piezas se evalua como restriccion de seguridad.');
end

function generar_modelo_supervisor(P, outDir)
    mdl = nuevo_modelo('act08_supervisor_pid_ocupacion');

    add_block('simulink/Sources/Constant', [mdl '/n*'], 'Value', num2str(P.n_ref));
    add_block('simulink/Sources/Constant', [mdl '/lambda(k)'], 'Value', num2str(P.lambda0));
    add_block('simulink/Sources/Step', [mdl '/d(k)'], 'Time', '95', 'Before', '0', 'After', '1');
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Supervisor inteligente']);
    poblar_supervisor_inteligente([mdl '/Supervisor inteligente']);
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/PID ocupacion']);
    poblar_pid_ocupacion([mdl '/PID ocupacion']);
    add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Esteras supervisadas']);
    poblar_esteras([mdl '/Esteras supervisadas']);
    add_block('simulink/Sinks/Scope', [mdl '/Scope ocupacion y control'], 'NumInputPorts', '2');

    conectar(mdl, {
        'Esteras supervisadas/1', 'Supervisor inteligente/1'
        'Esteras supervisadas/2', 'Supervisor inteligente/2'
        'd(k)/1', 'Supervisor inteligente/3'
        'n*/1', 'PID ocupacion/1'
        'Esteras supervisadas/1', 'PID ocupacion/2'
        'Esteras supervisadas/2', 'PID ocupacion/3'
        'Supervisor inteligente/1', 'PID ocupacion/4'
        'PID ocupacion/1', 'Esteras supervisadas/1'
        'PID ocupacion/2', 'Esteras supervisadas/2'
        'Supervisor inteligente/2', 'Esteras supervisadas/3'
        'Supervisor inteligente/3', 'Esteras supervisadas/4'
        'lambda(k)/1', 'Esteras supervisadas/5'
        'd(k)/1', 'Esteras supervisadas/6'
        'Esteras supervisadas/1', 'Scope ocupacion y control/1'
        'Esteras supervisadas/2', 'Scope ocupacion y control/2'
    });

    nota(mdl, 'Supervisor 4.2.d: ajuste de ganancias PID, admision y reparto con barrera de capacidad.');
    exportar_diagrama(mdl, outDir, 'sl_supervisor_pid.png');
    abrir_y_exportar([mdl '/Supervisor inteligente'], outDir, 'sl_supervisor_inteligente.png');
    abrir_y_exportar([mdl '/PID ocupacion'], outDir, 'sl_supervisor_pid_ocupacion.png');
    abrir_y_exportar([mdl '/Esteras supervisadas'], outDir, 'sl_supervisor_esteras.png');
end

function poblar_supervisor_inteligente(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/n1']);
    add_block('simulink/Sources/In1', [sub '/n2']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Signal Routing/Mux', [sub '/x supervisor'], 'Inputs', '3');
    crear_bloque_funcion_matlab([sub '/Fuzzy supervisor act08_supervisor_fuzzy'], 'act08_supervisor_fuzzy', 3);
    add_block('simulink/Signal Routing/Demux', [sub '/ganancia r1 admision'], 'Outputs', '3');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/ganancias']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/admision']);
    conectar(sub, {
        'n1/1', 'x supervisor/1'
        'n2/1', 'x supervisor/2'
        'd/1', 'x supervisor/3'
        'x supervisor/1', 'Fuzzy supervisor act08_supervisor_fuzzy/1'
        'Fuzzy supervisor act08_supervisor_fuzzy/1', 'ganancia r1 admision/1'
        'ganancia r1 admision/1', 'ganancias/1'
        'ganancia r1 admision/2', 'r1/1'
        'ganancia r1 admision/3', 'admision/1'
    });
    nota(sub, 'Supervisor difuso llamado desde MATLAB: act08_supervisor_fuzzy([n1 n2 d]).');
end

function poblar_reglas_supervisor(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/nmax']);
    add_block('simulink/Sources/In1', [sub '/d']);
    add_block('simulink/Sources/Constant', [sub '/ganancia normal riesgo'], 'Value', '1');
    add_block('simulink/Sources/Constant', [sub '/admision nominal'], 'Value', '1');
    add_block('simulink/Sinks/Terminator', [sub '/nmax comparado con 2.4 y 2.7']);
    add_block('simulink/Sinks/Terminator', [sub '/d activa refuerzo']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/ganancias']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/admision']);
    conectar(sub, {
        'ganancia normal riesgo/1', 'ganancias/1'
        'admision nominal/1', 'admision/1'
        'nmax/1', 'nmax comparado con 2.4 y 2.7/1'
        'd/1', 'd activa refuerzo/1'
    });
    nota(sub, 'En ejecucion: normal, riesgo y alarma ajustan Kp/Ki, refuerzo de velocidad y admision.');
end

function poblar_reparto_robot(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/n1']);
    add_block('simulink/Sources/In1', [sub '/n2']);
    add_block('simulink/Math Operations/Sum', [sub '/n1-n2'], 'Inputs', '+-');
    add_block('simulink/Math Operations/Gain', [sub '/ganancia reparto'], 'Gain', '-0.4');
    add_block('simulink/Sources/Constant', [sub '/reparto base'], 'Value', '0.5');
    add_block('simulink/Math Operations/Sum', [sub '/r1 crudo'], 'Inputs', '++');
    add_block('simulink/Discontinuities/Saturation', [sub '/limite r1'], ...
        'LowerLimit', '0.2', 'UpperLimit', '0.8');
    add_block('simulink/Ports & Subsystems/Out1', [sub '/r1']);
    conectar(sub, {
        'n1/1', 'n1-n2/1'
        'n2/1', 'n1-n2/2'
        'n1-n2/1', 'ganancia reparto/1'
        'reparto base/1', 'r1 crudo/1'
        'ganancia reparto/1', 'r1 crudo/2'
        'r1 crudo/1', 'limite r1/1'
        'limite r1/1', 'r1/1'
    });
    nota(sub, 'Si E1 esta mas cargada, se reduce r1; si E2 esta mas cargada, se aumenta r1.');
end

function poblar_pid_ocupacion(sub)
    limpiar_subsistema(sub);
    add_block('simulink/Sources/In1', [sub '/n*']);
    add_block('simulink/Sources/In1', [sub '/n1']);
    add_block('simulink/Sources/In1', [sub '/n2']);
    add_block('simulink/Sources/In1', [sub '/ganancias']);
    add_block('simulink/Discrete/Discrete PID Controller', [sub '/PID E1']);
    add_block('simulink/Discrete/Discrete PID Controller', [sub '/PID E2']);
    add_block('simulink/Math Operations/Sum', [sub '/e1'], 'Inputs', '+-');
    add_block('simulink/Math Operations/Sum', [sub '/e2'], 'Inputs', '+-');
    add_block('simulink/Sinks/Terminator', [sub '/ganancias aplicadas al PID']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u1']);
    add_block('simulink/Ports & Subsystems/Out1', [sub '/u2']);
    conectar(sub, {
        'n1/1', 'e1/1'
        'n*/1', 'e1/2'
        'n2/1', 'e2/1'
        'n*/1', 'e2/2'
        'e1/1', 'PID E1/1'
        'e2/1', 'PID E2/1'
        'ganancias/1', 'ganancias aplicadas al PID/1'
        'PID E1/1', 'u1/1'
        'PID E2/1', 'u2/1'
    });
    nota(sub, 'PID local por estera con ganancias ajustadas por el supervisor.');
end

function generar_graficas_de_validacion(P, outDir)
    [t, y, u] = simular_servo_pid(P);
    f = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(2,1);
    nexttile; plot(t, y, 'LineWidth', 1.4); hold on; yline(P.ref_servo, '--');
    grid on; ylabel('y(k)'); title('Respuesta servo PID');
    nexttile; stairs(t, u, 'LineWidth', 1.4); grid on; ylabel('u(k)'); xlabel('muestra k');
    guardar_figura(f, fullfile(outDir, 'sl_respuesta_servo_pid.png'));
    close(f);

    [tf, yf, uf] = simular_servo_fuzzy(P);
    f = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(2,1);
    nexttile; plot(tf, yf, 'LineWidth', 1.4); hold on; yline(P.ref_servo, '--');
    grid on; ylabel('y(k)'); title('Respuesta servo Fuzzy');
    nexttile; stairs(tf, uf, 'LineWidth', 1.4); grid on; ylabel('u(k)'); xlabel('muestra k');
    guardar_figura(f, fullfile(outDir, 'sl_respuesta_fuzzy_servo.png'));
    close(f);

    [t2, n, uc] = simular_r2et_supervisado(P);
    f = figure('Visible', 'off', 'Color', 'w');
    plot(t2, n(:,1), 'LineWidth', 1.4); hold on;
    plot(t2, n(:,2), 'LineWidth', 1.4);
    yline(P.n_ref, '--'); yline(P.n_max, ':');
    grid on; xlabel('muestra k'); ylabel('ocupacion'); legend('n1','n2','n*','capacidad', 'Location', 'best');
    title('Ocupacion R2ET supervisada');
    guardar_figura(f, fullfile(outDir, 'sl_r2et_ocupacion_demo.png'));
    guardar_figura(f, fullfile(outDir, 'sl_supervisor_scope_ocupacion.png'));
    close(f);

    f = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(3,1);
    nexttile; stairs(t2, uc(:,1), 'LineWidth', 1.2); grid on; ylabel('u1');
    nexttile; stairs(t2, uc(:,2), 'LineWidth', 1.2); grid on; ylabel('u2');
    nexttile; stairs(t2, uc(:,3), 'LineWidth', 1.2); grid on; ylabel('r1'); xlabel('muestra k');
    guardar_figura(f, fullfile(outDir, 'sl_supervisor_scope_control.png'));
    close(f);

end

function guardar_figura(figHandle, outFile)
    try
        exportgraphics(figHandle, outFile, 'Resolution', 180);
    catch
        print(figHandle, outFile, '-dpng', '-r180');
    end
end

function [t, y, u] = simular_servo_pid(P)
    N = 150;
    t = (0:N-1)';
    y = zeros(N,1);
    u = zeros(N,1);
    I = 0; e_ant = 0;
    Kp = 4.0; Ki = 0.30; Kd = 1.0; uff = 0.60;
    for k = 2:N
        r = P.ref_servo * double(k >= 10);
        d = double(k >= 85 && k <= 110);
        e = r - y(k-1);
        de = e - e_ant;
        raw = uff + Kp*e + Ki*I + Kd*de;
        u(k) = min(max(raw, P.u_min), P.u_max);
        if ~((raw > P.u_max && e > 0) || (raw < P.u_min && e < 0))
            I = min(max(I + e, -6), 6);
        end
        y(k) = P.a*y(k-1) + P.b*u(k) - P.bd*d;
        e_ant = e;
    end
end

function [t, y, u] = simular_servo_fuzzy(P)
    N = 150;
    t = (0:N-1)';
    y = zeros(N,1);
    u = zeros(N,1);
    e_ant = 0;
    for k = 2:N
        r = P.ref_servo * double(k >= 10);
        d = double(k >= 85 && k <= 110);
        e = r - y(k-1);
        de = e - e_ant;
        u(k) = act08_servo_fuzzy([e; de; d]);
        y(k) = P.a*y(k-1) + P.b*u(k) - P.bd*d;
        e_ant = e;
    end
end

function [t, n, uc] = simular_r2et_supervisado(P)
    N = 160;
    t = (0:N-1)';
    n = zeros(N,2);
    uc = zeros(N,3);
    n(1,:) = [1.25 1.25];
    for k = 2:N
        lambda = P.lambda0 + P.lambda_pulso * double(k >= 55 && k <= 82);
        d = double(k >= 95 && k <= 118);
        e = n(k-1,:) - P.n_ref;
        primary = act08_r2et_fuzzy([e(1); e(2); d]);
        sup = act08_supervisor_fuzzy([n(k-1,1); n(k-1,2); d]);
        u = min(max(primary(1:2)' * sup(1), P.u_min), P.u_max);
        r1 = sup(2);
        adm = sup(3);
        loss = [P.loss1, P.loss2] * d;
        feed = lambda * adm * [r1, 1-r1];
        out = P.eta * u .* (1-loss);
        n(k,:) = max(0, n(k-1,:) + feed - out);
        uc(k,:) = [u(1), u(2), r1];
    end
end

function r1 = reparto_difuso(theta, P)
    r1 = 0.5;
    if theta > 0.15
        r1 = 0.30;
    elseif theta < -0.15
        r1 = 0.70;
    end
    r1 = min(max(r1, P.r_min), P.r_max);
end

function [u, r1, adm] = supervisor_capacidad_matlab(u, r1, n, d, P)
    adm = 1.0;
    nmax = max(n);
    if nmax > P.alarm
        u = u + 0.14; adm = 0.72;
    elseif d > 0
        u = u + 0.14; adm = 0.86;
    elseif nmax > P.risk
        u = u + 0.07; adm = 0.90;
    end
    if n(1) >= P.n_max - 0.05 && n(2) < P.n_max - 0.05
        r1 = 0.20;
    elseif n(2) >= P.n_max - 0.05 && n(1) < P.n_max - 0.05
        r1 = 0.80;
    elseif all(n >= P.n_max - 0.05)
        r1 = 0.50; adm = 0.72;
    end
    u = min(max(u, P.u_min), P.u_max);
end

function mdl = nuevo_modelo(nombre)
    if bdIsLoaded(nombre)
        close_system(nombre, 0);
    end
    if exist([nombre '.slx'], 'file')
        delete([nombre '.slx']);
    end
    new_system(nombre);
    open_system(nombre);
    mdl = nombre;
end

function conectar(sys, pares)
    for i = 1:size(pares, 1)
        try
            add_line(sys, pares{i,1}, pares{i,2}, 'autorouting', 'on');
        catch ME
            error('No se pudo conectar %s -> %s en %s: %s', ...
                pares{i,1}, pares{i,2}, sys, ME.message);
        end
    end
end

function limpiar_subsistema(sub)
    try
        lineas = find_system(sub, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
        for i = 1:numel(lineas)
            delete_line(lineas(i));
        end
    catch
    end
    bloques = get_param(sub, 'Blocks');
    for i = 1:numel(bloques)
        delete_block([sub '/' bloques{i}]);
    end
end

function nota(sys, texto)
    try
        Simulink.Annotation(sys, texto);
    catch
    end
end

function crear_bloque_funcion_matlab(blockPath, functionName, outputWidth)
    try
        add_block('simulink/User-Defined Functions/MATLAB Function', blockPath);
        rt = sfroot;
        chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', blockPath);
        if isempty(chart)
            error('No se pudo encontrar el objeto Stateflow.EMChart.');
        end

        if outputWidth == 1
            initLine = 'y = 0.0;';
        else
            initLine = sprintf('y = zeros(%d,1);', outputWidth);
        end

        chart.Script = sprintf([ ...
            'function y = fcn(x)\n' ...
            '%% Llamada a funcion MATLAB externa usada por el modelo Simulink.\n' ...
            '%s\n' ...
            'y = %s(x);\n' ...
            'end\n'], initLine, functionName);
    catch ME
        try
            delete_block(blockPath);
        catch
        end
        warning('act08:simulink:fallback', ...
            'No se pudo crear MATLAB Function para %s (%s). Se crea subsistema visual conectado.', ...
            functionName, ME.message);
        crear_bloque_funcion_respaldo(blockPath, functionName, outputWidth);
    end
end

function crear_bloque_funcion_respaldo(blockPath, functionName, outputWidth)
    add_block('simulink/Ports & Subsystems/Subsystem', blockPath);
    limpiar_subsistema(blockPath);

    add_block('simulink/Sources/In1', [blockPath '/x']);
    add_block('simulink/Sinks/Terminator', [blockPath '/entrada x']);
    if outputWidth == 1
        value = '0';
    else
        value = sprintf('zeros(%d,1)', outputWidth);
    end
    add_block('simulink/Sources/Constant', [blockPath '/salida visual'], 'Value', value);
    add_block('simulink/Ports & Subsystems/Out1', [blockPath '/y']);
    conectar(blockPath, {
        'x/1', 'entrada x/1'
        'salida visual/1', 'y/1'
    });
    nota(blockPath, ['Fallback visual conectado. Funcion prevista: ' functionName '.']);
end

function exportar_diagrama(mdl, outDir, archivo)
    try
        Simulink.BlockDiagram.arrangeSystem(mdl);
    catch
    end
    set_param(mdl, 'ZoomFactor', 'FitSystem');
    print(['-s' mdl], '-dpng', '-r180', fullfile(outDir, archivo));
    rootDir = fileparts(fileparts(outDir));
    modelDir = fullfile(rootDir, 'simulations', 'generated_models');
    if ~exist(modelDir, 'dir')
        mkdir(modelDir);
    end
    save_system(mdl, fullfile(modelDir, [mdl '.slx']));
end

function abrir_y_exportar(sys, outDir, archivo)
    try
        open_system(sys);
        Simulink.BlockDiagram.arrangeSystem(sys);
    catch
    end
    print(['-s' sys], '-dpng', '-r180', fullfile(outDir, archivo));
end
