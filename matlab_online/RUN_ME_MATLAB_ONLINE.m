function RUN_ME_MATLAB_ONLINE()
%RUN_ME_MATLAB_ONLINE Punto de entrada unico para MATLAB Online.
%
% Uso:
%   1. Subir esta carpeta completa a MATLAB Online.
%   2. Abrir RUN_ME_MATLAB_ONLINE.m.
%   3. Pulsar Run.
%
% El proceso regenera tablas, figuras, modelos/capturas Simulink, reporte,
% manifest y paquete ZIP de resultados. Si el entorno no tiene LaTeX, la
% compilacion del PDF queda marcada en el reporte, pero los artefactos
% MATLAB/Simulink se generan igualmente.

    rootDir = fileparts(mfilename('fullpath'));
    cd(rootDir);

    addpath(genpath(fullfile(rootDir, 'src', 'matlab')));
    addpath(genpath(fullfile(rootDir, 'simulations')));

    fprintf('\nActividad 08 MROB - paquete autocontenido MATLAB Online\n');
    fprintf('Carpeta de trabajo: %s\n\n', rootDir);

    generar_entrega_10_10();

    fprintf('\nProceso terminado.\n');
    fprintf('Revise artifacts/matlab_delivery/reporte_generacion.md\n');
    fprintf('Descargue resultados/entrega_control_inteligente_matlab.zip\n');
end
