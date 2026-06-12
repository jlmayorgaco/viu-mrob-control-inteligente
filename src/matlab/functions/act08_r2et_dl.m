function y = act08_r2et_dl(x)
%ACT08_R2ET_DL Inferencia de red profunda para coordinacion R2ET.
% Entrada: x = [e1; e2; d]. Salida: y = [u1; u2; r1].

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
    z = extractdata(z);
else
    z = x(1:3);
    z = max(0, W{1}*z + b{1});
    z = max(0, W{2}*z + b{2});
    z = 1 ./ (1 + exp(-(W{3}*z + b{3})));
end

y = [0.05 + 0.95*z(1);
     0.05 + 0.95*z(2);
     0.20 + 0.60*z(3)];
y = [local_clip(y(1), 0.05, 1.0);
     local_clip(y(2), 0.05, 1.0);
     local_clip(y(3), 0.20, 0.80)];
end

function y = local_relu(x)
y = max(x, 0);
end

function y = local_sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function y = local_clip(x, lo, hi)
y = min(max(x, lo), hi);
end

function [W, b] = local_weights()
W = cell(3, 1);
b = cell(3, 1);
W{1} = [ 1.10  0.10  0.35;
         0.10  1.10  0.35;
        -0.60  0.60  0.10;
         0.60 -0.60  0.10;
         0.30  0.30  0.45;
         0.80  0.20  0.20;
         0.20  0.80  0.20;
        -0.25 -0.25  0.30];
b{1} = [0.08; 0.08; 0.04; 0.04; 0.05; 0.02; 0.02; 0.10];
W{2} = [0.60 0.10 0.20 0.05 0.40 0.25 0.05 0.10;
        0.10 0.60 0.05 0.20 0.40 0.05 0.25 0.10;
        0.10 0.10 0.45 0.65 0.15 0.20 0.20 0.05;
        0.35 0.35 0.10 0.10 0.30 0.15 0.15 0.20;
        0.20 0.20 0.30 0.30 0.25 0.20 0.20 0.25];
b{2} = [0.03; 0.03; 0.02; 0.04; 0.05];
W{3} = [0.95 0.15 0.10 0.45 0.25;
        0.15 0.95 0.10 0.45 0.25;
       -0.10 0.10 1.10 -1.10 0.20];
b{3} = [-0.08; -0.08; 0.00];
end
