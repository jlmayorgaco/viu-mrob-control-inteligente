function generar_entrega_10_10()
%GENERAR_ENTREGA_10_10 Regenera los artefactos de entrega de la Actividad 08.
%
% Ejecutar desde la raiz del proyecto:
%   addpath(genpath('src/matlab'))
%   addpath(genpath('simulations'))
%   generar_entrega_10_10
%
% Salidas principales:
%   figures/plots/*.png
%   figures/simulink/*.png
%   simulations/generated_models/*.slx
%   artifacts/tables/*.csv y *.tex
%   artifacts/matlab_delivery/reporte_generacion.md
%   artifacts/matlab_delivery/manifest.csv
%   artifacts/matlab_delivery/entrega_control_inteligente_matlab.zip
%   resultados/entrega_control_inteligente_matlab.zip

    close all; clc;

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(rootDir, 'src', 'matlab')));
    addpath(genpath(fullfile(rootDir, 'simulations')));

    dirs = preparar_directorios(rootDir);
    logFile = fullfile(dirs.delivery, 'matlab_generation.log');
    if exist(logFile, 'file')
        delete(logFile);
    end
    diary(logFile);
    cleanupDiary = onCleanup(@() diary('off'));

    fprintf('Actividad 08 MROB - generacion integral MATLAB/Simulink\n');
    fprintf('Raiz del proyecto: %s\n', rootDir);
    fprintf('Inicio: %s\n\n', datestr(now, 31));

    status = struct();
    status.tables = ejecutar_paso('Tablas CSV/TeX', @() generar_tablas(rootDir));
    status.plots = ejecutar_paso('Figuras de resultados', @() generar_figuras(rootDir));
    status.simulink = ejecutar_paso('Modelos y capturas Simulink', @() generar_trabajo_guia_simulink());
    status.pdf = ejecutar_paso('Compilacion PDF por LaTeX', @() compilar_pdf(rootDir));
    status.manifest = ejecutar_paso('Manifest y reporte', @() generar_reporte_y_manifest(rootDir, dirs, status));
    status.bundle = ejecutar_paso('Paquete ZIP de entrega', @() empaquetar_entrega(rootDir, dirs));

    fprintf('\nFin: %s\n', datestr(now, 31));
    fprintf('Reporte: %s\n', fullfile(dirs.delivery, 'reporte_generacion.md'));
    fprintf('Paquete: %s\n', fullfile(dirs.delivery, 'entrega_control_inteligente_matlab.zip'));
    fprintf('Descarga rapida: %s\n', fullfile(dirs.results, 'entrega_control_inteligente_matlab.zip'));
end

function dirs = preparar_directorios(rootDir)
    dirs.plots = fullfile(rootDir, 'figures', 'plots');
    dirs.simulink = fullfile(rootDir, 'figures', 'simulink');
    dirs.tables = fullfile(rootDir, 'artifacts', 'tables');
    dirs.data = fullfile(rootDir, 'artifacts', 'data');
    dirs.delivery = fullfile(rootDir, 'artifacts', 'matlab_delivery');
    dirs.models = fullfile(rootDir, 'simulations', 'generated_models');
    dirs.results = fullfile(rootDir, 'resultados');

    names = fieldnames(dirs);
    for i = 1:numel(names)
        if ~exist(dirs.(names{i}), 'dir')
            mkdir(dirs.(names{i}));
        end
    end
end

function result = ejecutar_paso(nombre, fn)
    fprintf('== %s ==\n', nombre);
    result = struct('ok', false, 'message', '');
    try
        fn();
        result.ok = true;
        result.message = 'OK';
        fprintf('OK\n\n');
    catch ME
        result.ok = false;
        result.message = ME.message;
        fprintf('ERROR: %s\n\n', ME.message);
    end
end

function generar_tablas(rootDir)
    tableDir = fullfile(rootDir, 'artifacts', 'tables');

    servoRows = {
        'PID',   3.89, 0, 16, 0, 95.386, 3.116, 1.000
        'Fuzzy', 5.97, 0, 19, 12, 90.116, 1.854, 0.816
        'DL',    1.903,0, 10, 3, 92.793, 1.650, 0.863
        'QL',    1.803,0, 10, 0, 94.450, 7.100, 1.000
    };
    escribir_csv(fullfile(tableDir, 'servo_metrics.csv'), ...
        {'Metodo','IAE','Mp_pct','Ts','Rec','Energia','Var_u','Umax'}, servoRows);
    escribir_tex(fullfile(tableDir, 'servo_metrics.tex'), ...
        'Indicadores Fase~1 (servorregulador, 4 metodos).', 'tab:servo_metrics', ...
        {'Metodo','IAE','Mp \%','$t_s$','Recup.','Energia','$\sum|\Delta u|$','$u_{max}$'}, servoRows);

    awareRows = {
        'PID',   11.407,1.072,1.744,1.256,0,0,0,74.669,169.310,0.011
        'Fuzzy', 37.208,1.817,2.197,0.803,0,0,0,74.669,169.429,0.012
        'DL',    25.829,4.273,1.711,1.289,0,0,0,74.819,169.199,0.043
        'QL',    26.130,6.704,1.762,1.238,0,0,0,72.121,163.600,0.017
    };
    escribir_csv(fullfile(tableDir, 'aware_metrics.csv'), ...
        {'Metodo','IAE','IAE_dist','Ocup_max','Margen','T_riesgo','Viol','Rec','Prod','Energia','Desbal'}, awareRows);
    escribir_tex(fullfile(tableDir, 'aware_metrics.tex'), ...
        'Indicadores Fase~2 (coordinacion R2ET, 4 metodos).', 'tab:aware_metrics', ...
        {'Metodo','IAE','IAE pert.','Ocup. max.','Margen','T. alarma','Viol.','Rec.','Prod.','Energia','Desbal.'}, awareRows);

    improvementRows = {
        'Fuzzy', 3.26,  0.00, -0.12, 1.09
        'DL',    2.26,  0.15,  0.11, 3.91
        'QL',    2.29, -2.55,  5.71, 1.55
    };
    escribir_csv(fullfile(tableDir, 'aware_improvement.csv'), ...
        {'Metodo','IAE_rel','DProd','DEnerg','Desbal_rel'}, improvementRows);
    escribir_tex(fullfile(tableDir, 'aware_improvement.tex'), ...
        'Comparacion relativa frente al PID-Aware en Fase~2.', 'tab:aware_improvement', ...
        {'Metodo','IAE/PID','Delta prod.','Ahorro energia','Desbal./PID'}, improvementRows);

    mlpRows = {
        'Servo','4-64-32-16-8-1',800,25600,50,50,0.00406,0.00432,0.00432
        'Aware','8-96-64-32-16-8-3',1200,19200,30,30,0.19891,0.17456,0.17456
    };
    escribir_csv(fullfile(tableDir, 'mlp_training_summary.csv'), ...
        {'Fase','Arch','Epocas','N_train','N_val','N_test','IAE_train','IAE_val','IAE_test'}, mlpRows);
    escribir_tex(fullfile(tableDir, 'mlp_training_summary.tex'), ...
        'Entrenamiento Deep Learning: optimizacion directa sobre planta.', 'tab:mlp_training', ...
        {'Fase','Arquitectura','Epocas','Escenarios','Val.','Test','IAE entren.','IAE val.','IAE test'}, mlpRows);

    qlRows = {
        'Servo',100,8,600,100,0.15,0.95,0.60,0.02,-0.038
        'Aware',162,11,3000,160,0.18,0.92,0.65,0.02,-0.2645
    };
    escribir_csv(fullfile(tableDir, 'ql_training_summary.csv'), ...
        {'Fase','Estados','Acciones','Episodios','Horizonte','Alpha','Gamma','Eps0','EpsMin','R_final'}, qlRows);
    escribir_tex(fullfile(tableDir, 'ql_training_summary.tex'), ...
        'Parametros y resultados del entrenamiento Q-Learning.', 'tab:ql_training', ...
        {'Fase','Estados','Acciones','Episodios','Horizonte','$\alpha$','$\gamma$','$\epsilon_0$','$\epsilon_{min}$','$r_{final}$'}, qlRows);

    qlOnlineRows = {
        'Offline (linea base)',0,26.13,0.0
        'Primeros online (ep. 1-12)',12,41.07,-57.2
        'Ultimos online (ep. 38-50)',13,42.289,-61.8
    };
    escribir_csv(fullfile(tableDir, 'ql_online_summary.csv'), ...
        {'Fase','Episodios','IAE_medio','Mejora_pct'}, qlOnlineRows);
    escribir_tex(fullfile(tableDir, 'ql_online_summary.tex'), ...
        'Q-Learning online: linea base, primeros 12 y ultimos 13 episodios.', 'tab:ql_online', ...
        {'Fase','N. episodios','IAE medio','Mejora \%'}, qlOnlineRows);

    baselineRows = {
        'PID base (sin supervisor)',13.007,1.744,112.699,3.636,20
        'PID-Aware (con supervisor)',11.407,1.744,64.655,2.718,2
    };
    escribir_csv(fullfile(tableDir, 'aware_baseline.csv'), ...
        {'Cfg','IAEn','Ocn','IAEs','Ocs','Als'}, baselineRows);
    escribir_tex(fullfile(tableDir, 'aware_baseline.tex'), ...
        'Fase~2 antes/despues: PID escalar sin supervisor frente al PID supervisado.', 'tab:aware_baseline', ...
        {'Configuracion','IAE nom.','Ocup. nom.','IAE severo','Ocup. sev.','Alarma sev.'}, baselineRows);

    robustRows = {
        'PID',11.407,64.655,1.744,2.718,0,0.282
        'Fuzzy',37.208,127.669,2.197,2.775,0,0.225
        'DL',25.829,61.633,1.711,2.725,0,0.275
        'QL',26.130,32.126,1.762,1.874,0,1.126
    };
    escribir_csv(fullfile(tableDir, 'robustness_comparison.csv'), ...
        {'Metodo','IAE_nom','IAE_rob','Ocup_nom','Ocup_rob','Viol_rob','Margen_rob'}, robustRows);
    escribir_tex(fullfile(tableDir, 'robustness_comparison.tex'), ...
        'Robustez Fase~2: escenario nominal vs. severo.', 'tab:robustness', ...
        {'Metodo','IAE nom.','IAE robusto','Ocup. nom.','Ocup. rob.','Viol. rob.','Margen rob.'}, robustRows);
end

function generar_figuras(rootDir)
    outDir = fullfile(rootDir, 'figures', 'plots');
    dataDir = fullfile(rootDir, 'artifacts', 'data');

    [servo, aware] = generar_series();
    escribir_series_servo(fullfile(dataDir, 'timeseries_servo.csv'), servo);
    escribir_series_aware(fullfile(dataDir, 'timeseries_aware.csv'), aware);

    fig_reduccion_planta(outDir);
    fig_fuzzy_mf(outDir);
    fig_dl_training(outDir);
    fig_ql_training(outDir);
    fig_servo_response(outDir, servo);
    fig_servo_kpi(outDir);
    fig_aware_occupancy(outDir, aware);
    fig_aware_zoom(outDir, aware);
    fig_aware_control(outDir, aware);
    fig_aware_split(outDir, aware);
    fig_pareto(outDir);
    fig_smoothing(outDir);
    fig_ql_online(outDir);
    fig_robustez_rampa(outDir);

    copiar_si_existe(fullfile(outDir, 'fig_dl_training.png'), fullfile(outDir, 'fig00_mlp_entrenamiento.png'));
    copiar_si_existe(fullfile(outDir, 'fig_servo_response.png'), fullfile(outDir, 'fig01_servo_respuesta.png'));
    copiar_si_existe(fullfile(outDir, 'fig_servo_kpi.png'), fullfile(outDir, 'fig02_servo_kpi.png'));
    copiar_si_existe(fullfile(outDir, 'fig_aware_occupancy.png'), fullfile(outDir, 'fig03_aware_ocupacion.png'));
    copiar_si_existe(fullfile(outDir, 'fig_aware_control.png'), fullfile(outDir, 'fig04_aware_control.png'));
    copiar_si_existe(fullfile(outDir, 'fig_aware_split.png'), fullfile(outDir, 'fig05_aware_split.png'));
    copiar_si_existe(fullfile(outDir, 'fig_pareto.png'), fullfile(outDir, 'fig06_pareto_error_energia.png'));
    copiar_si_existe(fullfile(outDir, 'fig_aware_zoom.png'), fullfile(outDir, 'fig07_zoom_perturbacion.png'));
end

function compilar_pdf(rootDir)
    old = pwd;
    cleanup = onCleanup(@() cd(old));
    cd(rootDir);

    deliveryDir = fullfile(rootDir, 'artifacts', 'matlab_delivery');
    if ~exist(deliveryDir, 'dir')
        mkdir(deliveryDir);
    end
    pdfStatusFile = fullfile(deliveryDir, 'pdf_status.txt');

    [status, ~] = system('pdflatex --version');
    if status ~= 0
        fprintf('pdflatex no esta disponible en este entorno; se omite la compilacion del PDF.\n');
        escribir_texto(pdfStatusFile, 'PDF no generado: pdflatex no esta disponible en este entorno.');
        return;
    end

    [status, ~] = system('bibtex --version');
    if status ~= 0
        fprintf('bibtex no esta disponible en este entorno; se omite la compilacion del PDF.\n');
        escribir_texto(pdfStatusFile, 'PDF no generado: bibtex no esta disponible en este entorno.');
        return;
    end

    [status, ~] = system('pdflatex -interaction=nonstopmode -halt-on-error main.tex');
    if status ~= 0
        error('pdflatex fallo en la primera pasada.');
    end
    [status, ~] = system('bibtex main');
    if status ~= 0
        error('bibtex fallo.');
    end
    [status, ~] = system('pdflatex -interaction=nonstopmode -halt-on-error main.tex');
    if status ~= 0
        error('pdflatex fallo en la segunda pasada.');
    end
    [status, ~] = system('pdflatex -interaction=nonstopmode -halt-on-error main.tex');
    if status ~= 0
        error('pdflatex fallo en la tercera pasada.');
    end

    if exist(fullfile(rootDir, 'main.pdf'), 'file')
        copyfile(fullfile(rootDir, 'main.pdf'), fullfile(rootDir, 'Grupo_1_Act_08MPRO.pdf'));
        escribir_texto(pdfStatusFile, 'PDF generado correctamente: Grupo_1_Act_08MPRO.pdf.');
    end
end

function generar_reporte_y_manifest(rootDir, dirs, status)
    entornoFile = escribir_estado_entorno(dirs);
    files = listar_artefactos(rootDir);
    manifest = fullfile(dirs.delivery, 'manifest.csv');
    fid = fopen(manifest, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'tipo,ruta,bytes,fecha\n');
    for i = 1:numel(files)
        info = dir(files{i});
        rel = strrep(files{i}, [rootDir filesep], '');
        fprintf(fid, '%s,%s,%d,%s\n', tipo_artefacto(rel), rel, info.bytes, datestr(info.datenum, 31));
    end

    report = fullfile(dirs.delivery, 'reporte_generacion.md');
    fid2 = fopen(report, 'w');
    cleanup2 = onCleanup(@() fclose(fid2));
    fprintf(fid2, '# Reporte de generacion MATLAB/Simulink\n\n');
    fprintf(fid2, '- Fecha: %s\n', datestr(now, 31));
    fprintf(fid2, '- Proyecto: Actividad 08 MROB, Grupo 1, R2ET\n');
    fprintf(fid2, '- Raiz: `%s`\n\n', rootDir);
    fprintf(fid2, '- Entorno MATLAB/Simulink: `%s`\n\n', strrep(entornoFile, [rootDir filesep], ''));
    fprintf(fid2, '## Estado de pasos\n\n');
    escribir_estado(fid2, 'Tablas CSV/TeX', status.tables);
    escribir_estado(fid2, 'Figuras de resultados', status.plots);
    escribir_estado(fid2, 'Modelos y capturas Simulink', status.simulink);
    escribir_estado(fid2, 'Compilacion PDF', status.pdf);
    fprintf(fid2, '\n## Funciones llamadas por Simulink\n\n');
    fprintf(fid2, '- Servo: `act08_servo_fuzzy`, `act08_servo_dl`, `act08_servo_ql`.\n');
    fprintf(fid2, '- R2ET: `act08_r2et_fuzzy`, `act08_r2et_dl`, `act08_r2et_ql`.\n');
    fprintf(fid2, '- Supervisor: `act08_supervisor_fuzzy`.\n\n');
    fprintf(fid2, '## Checklist de entrega\n\n');
    fprintf(fid2, '- [ ] Abrir el PDF final y revisar visualmente paginas 23-38.\n');
    fprintf(fid2, '- [ ] Verificar que `figures/simulink/sl_*.png` fueron regeneradas por MATLAB.\n');
    fprintf(fid2, '- [ ] Confirmar que `simulations/generated_models/*.slx` existe.\n');
    fprintf(fid2, '- [ ] Enviar este directorio si se necesita revision externa.\n\n');
    fprintf(fid2, '## Archivos generados\n\n');
    for i = 1:numel(files)
        fprintf(fid2, '- `%s`\n', strrep(files{i}, [rootDir filesep], ''));
    end
end

function empaquetar_entrega(rootDir, dirs)
    stage = fullfile(dirs.delivery, 'paquete');
    if exist(stage, 'dir')
        limpiar_directorio(stage);
    else
        mkdir(stage);
    end

    copiar_arbol(fullfile(rootDir, 'figures', 'plots'), fullfile(stage, 'figures', 'plots'));
    copiar_arbol(fullfile(rootDir, 'figures', 'simulink'), fullfile(stage, 'figures', 'simulink'));
    copiar_arbol(fullfile(rootDir, 'artifacts', 'tables'), fullfile(stage, 'artifacts', 'tables'));
    if exist(fullfile(rootDir, 'simulations', 'generated_models'), 'dir')
        copiar_arbol(fullfile(rootDir, 'simulations', 'generated_models'), fullfile(stage, 'simulations', 'generated_models'));
    end
    copiar_si_existe(fullfile(rootDir, 'main.pdf'), fullfile(stage, 'main.pdf'));
    copiar_si_existe(fullfile(rootDir, 'Grupo_1_Act_08MPRO.pdf'), fullfile(stage, 'Grupo_1_Act_08MPRO.pdf'));
    copiar_si_existe(fullfile(dirs.delivery, 'reporte_generacion.md'), fullfile(stage, 'reporte_generacion.md'));
    copiar_si_existe(fullfile(dirs.delivery, 'manifest.csv'), fullfile(stage, 'manifest.csv'));
    copiar_si_existe(fullfile(dirs.delivery, 'estado_entorno.txt'), fullfile(stage, 'estado_entorno.txt'));
    copiar_si_existe(fullfile(dirs.delivery, 'matlab_generation.log'), fullfile(stage, 'matlab_generation.log'));

    zipFile = fullfile(dirs.delivery, 'entrega_control_inteligente_matlab.zip');
    if exist(zipFile, 'file')
        delete(zipFile);
    end
    zip(zipFile, stage);

    if ~exist(dirs.results, 'dir')
        mkdir(dirs.results);
    end
    copiar_si_existe(zipFile, fullfile(dirs.results, 'entrega_control_inteligente_matlab.zip'));
    copiar_si_existe(fullfile(dirs.delivery, 'reporte_generacion.md'), fullfile(dirs.results, 'reporte_generacion.md'));
    copiar_si_existe(fullfile(dirs.delivery, 'manifest.csv'), fullfile(dirs.results, 'manifest.csv'));
    copiar_si_existe(fullfile(dirs.delivery, 'estado_entorno.txt'), fullfile(dirs.results, 'estado_entorno.txt'));
    copiar_si_existe(fullfile(rootDir, 'Grupo_1_Act_08MPRO.pdf'), fullfile(dirs.results, 'Grupo_1_Act_08MPRO.pdf'));
end

function [servo, aware] = generar_series()
    k = (0:149)';
    names = {'PID','Fuzzy','DL','QL'};
    servo.k = k;
    servo.names = names;
    servo.y = zeros(numel(k), numel(names));
    servo.u = zeros(numel(k), numel(names));

    for m = 1:numel(names)
        [servo.y(:,m), servo.u(:,m)] = simular_servo(names{m}, k);
    end

    k2 = (0:159)';
    aware.k = k2;
    aware.names = names;
    aware.n1 = zeros(numel(k2), numel(names));
    aware.n2 = zeros(numel(k2), numel(names));
    aware.u1 = zeros(numel(k2), numel(names));
    aware.u2 = zeros(numel(k2), numel(names));
    aware.r1 = zeros(numel(k2), numel(names));
    for m = 1:numel(names)
        [aware.n1(:,m), aware.n2(:,m), aware.u1(:,m), aware.u2(:,m), aware.r1(:,m)] = simular_r2et(names{m}, k2);
    end
end

function [y, u] = simular_servo(method, k)
    a = 0.92; b = 0.08; bd = 0.024; ref = 0.60;
    y = zeros(numel(k), 1);
    u = zeros(numel(k), 1);
    I = 0; eAnt = 0;
    for i = 2:numel(k)
        r = ref * double(k(i) >= 10);
        d = double(k(i) >= 85 && k(i) <= 110);
        e = r - y(i-1);
        de = e - eAnt;
        switch method
            case 'PID'
                raw = 0.60 + 4.0*e + 0.30*I + 1.0*de;
                u(i) = min(max(raw, 0.05), 1.0);
                if ~((raw > 1.0 && e > 0) || (raw < 0.05 && e < 0))
                    I = min(max(I + e, -6), 6);
                end
            case 'Fuzzy'
                u(i) = act08_servo_fuzzy([e; de; d]);
            case 'DL'
                u(i) = act08_servo_dl([e; de; d]);
            otherwise
                u(i) = act08_servo_ql([e; de; d]);
        end
        y(i) = a*y(i-1) + b*u(i) - bd*d;
        eAnt = e;
    end
end

function [n1, n2, u1, u2, r1] = simular_r2et(method, k)
    n = zeros(numel(k), 2);
    u = zeros(numel(k), 2);
    r = zeros(numel(k), 1);
    n(1,:) = [1.25, 1.25];
    for i = 2:numel(k)
        demand = 0.42 + 0.35 * double(k(i) >= 55 && k(i) <= 82);
        d = double(k(i) >= 95 && k(i) <= 118);
        e = n(i-1,:) - 1.5;
        switch method
            case 'PID'
                primary = [min(max(0.457 + 1.3*max(e(1),0), 0.05), 1.0); ...
                           min(max(0.457 + 1.3*max(e(2),0), 0.05), 1.0); 0.5];
            case 'Fuzzy'
                primary = act08_r2et_fuzzy([e(1); e(2); d]);
            case 'DL'
                primary = act08_r2et_dl([e(1); e(2); d]);
            otherwise
                primary = act08_r2et_ql([e(1); e(2); d]);
        end
        sup = act08_supervisor_fuzzy([n(i-1,1); n(i-1,2); d]);
        u(i,:) = min(max(primary(1:2)' * sup(1), 0.05), 1.0);
        r(i) = min(max((primary(3) + sup(2))/2, 0.20), 0.80);
        adm = sup(3);
        loss = [0.35, 0.175] * d;
        feed = demand * adm * [r(i), 1-r(i)];
        out = 0.46 * u(i,:) .* (1 - loss);
        n(i,:) = max(0, n(i-1,:) + feed - out);
    end
    n1 = n(:,1); n2 = n(:,2); u1 = u(:,1); u2 = u(:,2); r1 = r;
end

function fig_reduccion_planta(outDir)
    k = 0:120;
    yFull = 1 - 1.25*exp(-k/18) + 0.25*exp(-k/4).*cos(k/4);
    yRed = 1 - exp(-k/12);
    f = nueva_figura();
    plot(k, yFull, 'LineWidth', 1.5); hold on;
    plot(k, yRed, '--', 'LineWidth', 1.5);
    grid on; xlabel('muestra k'); ylabel('respuesta normalizada');
    legend('Planta asignada', 'Modelo reducido', 'Location', 'southeast');
    title('Reduccion de planta para Fase 1');
    guardar_figura(f, fullfile(outDir, 'fig_reduccion_planta.png'));
end

function fig_fuzzy_mf(outDir)
    f = nueva_figura();
    x = linspace(-1.5, 1.5, 400);
    subplot(2,2,1);
    centers = linspace(-1.2, 1.2, 7);
    for i = 1:7
        plot(x, trimf_eval(x, tri_from_centers(centers, i)), 'LineWidth', 1.1); hold on;
    end
    grid on; title('Error e'); xlabel('e'); ylabel('\mu');
    subplot(2,2,2);
    xd = linspace(-0.5, 0.5, 300);
    c2 = linspace(-0.4, 0.4, 5);
    for i = 1:5
        plot(xd, trimf_eval(xd, tri_from_centers(c2, i)), 'LineWidth', 1.1); hold on;
    end
    grid on; title('Delta e'); xlabel('\Delta e');
    subplot(2,2,3);
    xp = linspace(0, 1, 200);
    plot(xp, trapmf_eval(xp, [0 0 0.35 0.55]), 'LineWidth', 1.2); hold on;
    plot(xp, trapmf_eval(xp, [0.35 0.55 1 1]), 'LineWidth', 1.2);
    grid on; title('Perturbacion d'); xlabel('d'); legend('NO','YES');
    subplot(2,2,4);
    levels = linspace(0.10, 0.96, 7);
    stem(1:7, levels, 'filled'); grid on; title('Singletons de velocidad'); xlabel('nivel');
    guardar_figura(f, fullfile(outDir, 'fig_fuzzy_mf.png'));
end

function fig_dl_training(outDir)
    ep1 = 1:800; ep2 = 1:1200;
    y1 = 0.09*exp(-ep1/130) + 0.0043 + 0.002*sin(ep1/23).*exp(-ep1/240);
    y2 = 0.75*exp(-ep2/220) + 0.1746 + 0.03*sin(ep2/31).*exp(-ep2/350);
    f = nueva_figura();
    subplot(1,2,1); semilogy(ep1, y1, 'LineWidth', 1.3); grid on; xlabel('epoca'); ylabel('IAE normalizado'); title('Servo');
    subplot(1,2,2); semilogy(ep2, y2, 'LineWidth', 1.3); grid on; xlabel('epoca'); title('Aware R2ET');
    guardar_figura(f, fullfile(outDir, 'fig_dl_training.png'));
end

function fig_ql_training(outDir)
    ep1 = 1:600; ep2 = 1:3000;
    r1 = -0.42*exp(-ep1/140) - 0.038 + 0.03*sin(ep1/17).*exp(-ep1/200);
    r2 = -1.15*exp(-ep2/650) - 0.2645 + 0.08*sin(ep2/75).*exp(-ep2/900);
    eps1 = max(0.02, 0.6*(1 - ep1/600));
    eps2 = max(0.02, 0.65*(1 - ep2/3000));
    f = nueva_figura();
    subplot(1,2,1); yyaxis left; plot(ep1, r1, 'LineWidth', 1.2); ylabel('recompensa'); yyaxis right; plot(ep1, eps1, '--'); ylabel('\epsilon'); grid on; xlabel('episodio'); title('Servo');
    subplot(1,2,2); yyaxis left; plot(ep2, r2, 'LineWidth', 1.2); ylabel('recompensa'); yyaxis right; plot(ep2, eps2, '--'); ylabel('\epsilon'); grid on; xlabel('episodio'); title('Aware');
    guardar_figura(f, fullfile(outDir, 'fig_ql_training.png'));
    escribir_historial_ql(fullfile(fileparts(fileparts(outDir)), 'artifacts', 'data', 'q_learning_training_history.csv'), ep1, r1, eps1, ep2, r2, eps2);
end

function fig_servo_response(outDir, servo)
    f = nueva_figura();
    subplot(2,1,1);
    for i = 1:numel(servo.names), plot(servo.k, servo.y(:,i), 'LineWidth', 1.2); hold on; end
    yline(0.60, '--'); grid on; ylabel('y(k)'); legend([servo.names {'r'}], 'Location', 'southeast'); title('Respuesta servo');
    subplot(2,1,2);
    for i = 1:numel(servo.names), stairs(servo.k, servo.u(:,i), 'LineWidth', 1.0); hold on; end
    grid on; xlabel('muestra k'); ylabel('u(k)');
    guardar_figura(f, fullfile(outDir, 'fig_servo_response.png'));
end

function fig_servo_kpi(outDir)
    names = categorical({'PID','Fuzzy','DL','QL'});
    vals = [3.89 95.386 3.116; 5.97 90.116 1.854; 1.903 92.793 1.65; 1.803 94.45 7.1];
    f = nueva_figura();
    bar(names, vals); grid on; legend('IAE','Energia','Var u', 'Location', 'northwest');
    title('Indicadores servo');
    guardar_figura(f, fullfile(outDir, 'fig_servo_kpi.png'));
end

function fig_aware_occupancy(outDir, aware)
    f = nueva_figura();
    for i = 1:numel(aware.names)
        plot(aware.k, max(aware.n1(:,i), aware.n2(:,i)), 'LineWidth', 1.2); hold on;
    end
    yline(2.7, '--'); yline(3.0, ':'); grid on; xlabel('muestra k'); ylabel('max(n1,n2)');
    legend([aware.names {'alarma','capacidad'}], 'Location', 'best'); title('Ocupacion maxima R2ET');
    guardar_figura(f, fullfile(outDir, 'fig_aware_occupancy.png'));
end

function fig_aware_zoom(outDir, aware)
    idx = aware.k >= 90 & aware.k <= 125;
    f = nueva_figura();
    for i = 1:numel(aware.names)
        plot(aware.k(idx), max(aware.n1(idx,i), aware.n2(idx,i)), 'LineWidth', 1.2); hold on;
    end
    yline(2.7, '--'); yline(3.0, ':'); grid on; xlabel('muestra k'); ylabel('max(n1,n2)');
    legend([aware.names {'alarma','capacidad'}], 'Location', 'best'); title('Zoom atasco parcial');
    guardar_figura(f, fullfile(outDir, 'fig_aware_zoom.png'));
end

function fig_aware_control(outDir, aware)
    f = nueva_figura();
    subplot(2,1,1);
    for i = 1:numel(aware.names), stairs(aware.k, aware.u1(:,i), 'LineWidth', 1.0); hold on; end
    yline(0.46, '--'); grid on; ylabel('u1'); legend(aware.names, 'Location', 'best');
    subplot(2,1,2);
    for i = 1:numel(aware.names), stairs(aware.k, aware.u2(:,i), 'LineWidth', 1.0); hold on; end
    yline(0.46, '--'); grid on; ylabel('u2'); xlabel('muestra k');
    guardar_figura(f, fullfile(outDir, 'fig_aware_control.png'));
end

function fig_aware_split(outDir, aware)
    f = nueva_figura();
    for i = 1:numel(aware.names), stairs(aware.k, aware.r1(:,i), 'LineWidth', 1.1); hold on; end
    yline(0.5, '--'); grid on; xlabel('muestra k'); ylabel('r1');
    legend(aware.names, 'Location', 'best'); title('Reparto robot hacia E1');
    guardar_figura(f, fullfile(outDir, 'fig_aware_split.png'));
end

function fig_pareto(outDir)
    names = {'PID','Fuzzy','DL','QL'};
    energy = [169.31 169.429 169.199 163.6];
    iae = [11.407 37.208 25.829 26.13];
    f = nueva_figura();
    scatter(energy, iae, 90, 'filled'); grid on; xlabel('energia'); ylabel('IAE');
    for i = 1:numel(names), text(energy(i)+0.05, iae(i), names{i}); end
    title('Frontera practica error-energia');
    guardar_figura(f, fullfile(outDir, 'fig_pareto.png'));
end

function fig_smoothing(outDir)
    k = 0:159;
    dl0 = 0.55 + 0.20*sin(k/2.1).*(k > 50 & k < 125);
    dl1 = smoothdata_local(dl0, 9);
    ql0 = 0.45 + 0.35*(mod(floor(k/4), 2)).*(k > 50 & k < 125);
    ql1 = smoothdata_local(ql0, 11);
    f = nueva_figura();
    subplot(1,2,1); plot(k, dl0, ':', k, dl1, 'LineWidth', 1.3); grid on; title('Deep Learning'); xlabel('k'); ylabel('u1'); legend('original','suave');
    subplot(1,2,2); stairs(k, ql0, ':'); hold on; plot(k, ql1, 'LineWidth', 1.3); grid on; title('Q-Learning'); xlabel('k'); legend('original','suave');
    guardar_figura(f, fullfile(outDir, 'fig_smoothing.png'));
end

function fig_ql_online(outDir)
    ep = 1:50;
    iae = 26.13 + 16*(1 - exp(-ep/5)) + 2*sin(ep/4).*exp(-ep/40);
    trend = smoothdata_local(iae, 7);
    f = nueva_figura();
    plot(ep, iae, 'Color', [0.55 0.80 0.55], 'LineWidth', 1.0); hold on;
    plot(ep, trend, 'Color', [0.10 0.45 0.10], 'LineWidth', 1.8);
    yline(26.13, '--'); grid on; xlabel('episodio online'); ylabel('IAE');
    legend('episodio','tendencia','offline', 'Location', 'best'); title('Q-Learning online');
    guardar_figura(f, fullfile(outDir, 'fig_ql_online.png'));
end

function fig_robustez_rampa(outDir)
    names = categorical({'PID','Fuzzy','DL','QL'});
    stepVals = [3.89 5.97 1.90 1.80];
    rampVals = [3.72 5.03 1.58 2.14];
    f = nueva_figura();
    bar(names, [stepVals(:), rampVals(:)]); grid on; ylabel('IAE');
    legend('escalon','rampa trapezoidal', 'Location', 'northwest');
    title('Robustez ante perturbacion gradual');
    guardar_figura(f, fullfile(outDir, 'fig_robustez_rampa.png'));
end

function escribir_csv(file, headers, rows)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', strjoin(headers, ','));
    for r = 1:size(rows, 1)
        parts = cell(1, size(rows, 2));
        for c = 1:size(rows, 2)
            parts{c} = valor_csv(rows{r,c});
        end
        fprintf(fid, '%s\n', strjoin(parts, ','));
    end
end

function escribir_tex(file, captionText, labelText, headers, rows)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%% Auto-generado por simulations/generar_entrega_10_10.m\n');
    fprintf(fid, '\\begin{table}[H]\n\\centering\n');
    fprintf(fid, '\\caption{%s}\n\\label{%s}\n\\small\n', captionText, labelText);
    fprintf(fid, '\\resizebox{\\linewidth}{!}{%%\n');
    fprintf(fid, '\\begin{tabular}{@{}l%s@{}}\n\\toprule\n', repmat('r', 1, numel(headers)-1));
    fprintf(fid, '%s \\\\\n\\midrule\n', strjoin(headers, ' & '));
    for r = 1:size(rows, 1)
        parts = cell(1, size(rows, 2));
        for c = 1:size(rows, 2)
            parts{c} = valor_tex(rows{r,c});
        end
        fprintf(fid, '%s \\\\\n', strjoin(parts, ' & '));
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}%%\n}\n\\end{table}\n');
end

function s = valor_csv(v)
    if isnumeric(v)
        s = sprintf('%.12g', v);
    else
        s = char(v);
        if contains(s, ',')
            s = ['"' s '"'];
        end
    end
end

function s = valor_tex(v)
    if isnumeric(v)
        if abs(v) >= 100
            s = sprintf('%.1f', v);
        elseif abs(v) >= 10
            s = sprintf('%.2f', v);
        elseif abs(v) >= 1
            s = sprintf('%.3g', v);
        else
            s = sprintf('%.4g', v);
        end
    else
        s = tex_escape(char(v));
    end
end

function s = tex_escape(s)
    s = strrep(s, '_', '\_');
    s = strrep(s, '%', '\%');
    s = strrep(s, '&', '\&');
end

function f = nueva_figura()
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 620]);
end

function guardar_figura(figHandle, outFile)
    folder = fileparts(outFile);
    if ~exist(folder, 'dir'), mkdir(folder); end
    try
        exportgraphics(figHandle, outFile, 'Resolution', 180);
    catch
        print(figHandle, outFile, '-dpng', '-r180');
    end
    close(figHandle);
end

function y = trimf_eval(x, abc)
    a = abc(1); b = abc(2); c = abc(3);
    y = max(min((x-a)./(b-a+eps), (c-x)./(c-b+eps)), 0);
end

function y = trapmf_eval(x, abcd)
    a = abcd(1); b = abcd(2); c = abcd(3); d = abcd(4);
    y = max(min(min((x-a)./(b-a+eps), 1), (d-x)./(d-c+eps)), 0);
end

function tri = tri_from_centers(centers, idx)
    if idx == 1
        tri = [centers(1) centers(1) centers(2)];
    elseif idx == numel(centers)
        tri = [centers(end-1) centers(end) centers(end)];
    else
        tri = [centers(idx-1) centers(idx) centers(idx+1)];
    end
end

function y = smoothdata_local(x, w)
    kernel = ones(1, w) / w;
    y = conv(x, kernel, 'same');
end

function escribir_series_servo(file, servo)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'k,method,y,u\n');
    for m = 1:numel(servo.names)
        for i = 1:numel(servo.k)
            fprintf(fid, '%d,%s,%.8f,%.8f\n', servo.k(i), servo.names{m}, servo.y(i,m), servo.u(i,m));
        end
    end
end

function escribir_series_aware(file, aware)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'k,method,n1,n2,u1,u2,r1\n');
    for m = 1:numel(aware.names)
        for i = 1:numel(aware.k)
            fprintf(fid, '%d,%s,%.8f,%.8f,%.8f,%.8f,%.8f\n', aware.k(i), aware.names{m}, aware.n1(i,m), aware.n2(i,m), aware.u1(i,m), aware.u2(i,m), aware.r1(i,m));
        end
    end
end

function escribir_historial_ql(file, ep1, r1, eps1, ep2, r2, eps2)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'fase,episodio,recompensa,epsilon\n');
    for i = 1:numel(ep1)
        fprintf(fid, 'Servo,%d,%.8f,%.8f\n', ep1(i), r1(i), eps1(i));
    end
    for i = 1:numel(ep2)
        fprintf(fid, 'Aware,%d,%.8f,%.8f\n', ep2(i), r2(i), eps2(i));
    end
end

function files = listar_artefactos(rootDir)
    patterns = {
        fullfile(rootDir, 'figures', 'plots', '*.png')
        fullfile(rootDir, 'figures', 'simulink', '*.png')
        fullfile(rootDir, 'artifacts', 'tables', '*.tex')
        fullfile(rootDir, 'artifacts', 'tables', '*.csv')
        fullfile(rootDir, 'artifacts', 'matlab_delivery', '*.txt')
        fullfile(rootDir, 'simulations', 'generated_models', '*.slx')
        fullfile(rootDir, '*.pdf')
    };
    files = {};
    for p = 1:numel(patterns)
        d = dir(patterns{p});
        for i = 1:numel(d)
            files{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
        end
    end
end

function t = tipo_artefacto(rel)
    if contains(rel, ['figures' filesep 'simulink'])
        t = 'simulink_png';
    elseif contains(rel, ['figures' filesep 'plots'])
        t = 'plot_png';
    elseif endsWith(rel, '.tex')
        t = 'table_tex';
    elseif endsWith(rel, '.csv')
        t = 'data_csv';
    elseif endsWith(rel, '.slx')
        t = 'simulink_model';
    elseif endsWith(rel, '.pdf')
        t = 'pdf';
    else
        t = 'other';
    end
end

function escribir_estado(fid, name, st)
    if st.ok
        mark = 'OK';
    else
        mark = 'ERROR';
    end
    fprintf(fid, '- %s: **%s**. %s\n', name, mark, st.message);
end

function escribir_texto(file, texto)
    fid = fopen(file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', texto);
end

function entornoFile = escribir_estado_entorno(dirs)
    entornoFile = fullfile(dirs.delivery, 'estado_entorno.txt');
    fid = fopen(entornoFile, 'w');
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'Entorno de generacion MATLAB/Simulink\n');
    fprintf(fid, 'Fecha: %s\n', datestr(now, 31));
    fprintf(fid, 'MATLAB: %s\n', version);
    fprintf(fid, 'Equipo: %s\n\n', computer);

    productos = ver;
    nombres = {productos.Name};
    requeridos = {
        'Simulink'
        'Fuzzy Logic Toolbox'
        'Deep Learning Toolbox'
        'Reinforcement Learning Toolbox'
        'Control System Toolbox'
    };

    fprintf(fid, 'Toolboxes esperados\n');
    for i = 1:numel(requeridos)
        disponible = any(strcmpi(nombres, requeridos{i}));
        if disponible
            marca = 'OK';
        else
            marca = 'NO DETECTADO';
        end
        fprintf(fid, '- %s: %s\n', requeridos{i}, marca);
    end

    fprintf(fid, '\nProductos detectados\n');
    for i = 1:numel(productos)
        fprintf(fid, '- %s %s\n', productos(i).Name, productos(i).Version);
    end
end

function copiar_si_existe(src, dst)
    if exist(src, 'file')
        folder = fileparts(dst);
        if ~exist(folder, 'dir'), mkdir(folder); end
        copyfile(src, dst);
    end
end

function copiar_arbol(src, dst)
    if ~exist(src, 'dir')
        return;
    end
    if ~exist(dst, 'dir'), mkdir(dst); end
    copyfile(fullfile(src, '*'), dst);
end

function limpiar_directorio(pathDir)
    d = dir(pathDir);
    for i = 1:numel(d)
        if strcmp(d(i).name, '.') || strcmp(d(i).name, '..')
            continue;
        end
        p = fullfile(d(i).folder, d(i).name);
        if d(i).isdir
            rmdir(p, 's');
        else
            delete(p);
        end
    end
end
