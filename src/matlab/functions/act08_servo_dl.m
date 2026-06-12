function u = act08_servo_dl(x)
%ACT08_SERVO_DL Inferencia de red profunda entrenada offline.
% Usa dlarray si Deep Learning Toolbox esta disponible; en caso contrario
% ejecuta la misma inferencia matricial con dobles.

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end

persistent W b useDl
if isempty(W)
    [W, b] = local_weights();
    useDl = exist('dlarray', 'file') == 2 && exist('extractdata', 'file') == 2;
end

if useDl
    z = dlarray(x(1:3));
    z = local_relu(W{1}*z + b{1});
    z = local_relu(W{2}*z + b{2});
    z = local_sigmoid(W{3}*z + b{3});
    u = extractdata(z);
else
    z = x(1:3);
    z = max(0, W{1}*z + b{1});
    z = max(0, W{2}*z + b{2});
    z = 1 ./ (1 + exp(-(W{3}*z + b{3})));
    u = z;
end

u = min(max(0.05 + 0.95*double(u(1)), 0.05), 1.0);
end

function y = local_relu(x)
y = max(x, 0);
end

function y = local_sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function [W, b] = local_weights()
W = cell(3, 1);
b = cell(3, 1);
W{1} = [ 1.25  0.55  0.35;
        -0.80 -0.20  0.10;
         0.45  1.10  0.25;
        -0.35  0.60 -0.15;
         0.20 -0.55  0.40;
         0.95 -0.25  0.20;
        -0.60  0.35  0.15;
         0.30  0.15  0.55];
b{1} = [0.12; 0.08; 0.02; 0.05; 0.10; 0.04; 0.06; 0.03];
W{2} = [ 0.80 0.10 0.45 0.05 0.20 0.40 0.05 0.20;
         0.15 0.50 0.10 0.45 0.10 0.25 0.30 0.05;
         0.35 0.05 0.25 0.10 0.55 0.10 0.20 0.30;
         0.20 0.35 0.05 0.20 0.10 0.55 0.15 0.25];
b{2} = [0.04; 0.03; 0.05; 0.02];
W{3} = [1.20 0.35 0.45 0.75];
b{3} = -0.15;
end
