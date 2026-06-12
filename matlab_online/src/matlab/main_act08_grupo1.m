% main_act08_grupo1.m
% Actividad 08 MROB - Grupo 1 - Caso R2ET
% Punto de entrada MATLAB para regenerar modelos Simulink y capturas.

clear; clc; close all;

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(repoRoot, 'src', 'matlab')));
addpath(genpath(fullfile(repoRoot, 'simulations')));

disp("Actividad 08 - Grupo 1 - R2ET");
disp("Generando entrega completa MATLAB/Simulink...");

generar_entrega_10_10();

disp("Proceso terminado.");
