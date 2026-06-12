function y = act08_supervisor_fuzzy(x)
%ACT08_SUPERVISOR_FUZZY Supervisor difuso de capacidad para Simulink.
% Entrada: x = [n1; n2; d]. Salida: y = [gainScale; r1; admission].

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end
n1 = x(1);
n2 = x(2);
d = x(3);
nmax = max(n1, n2);

gainScale = 1.00;
admission = 1.00;
if nmax > 2.70
    gainScale = 1.35;
    admission = 0.72;
elseif d > 0.5
    gainScale = 1.22;
    admission = 0.86;
elseif nmax > 2.40
    gainScale = 1.12;
    admission = 0.90;
end

theta = n1 - n2;
r1 = 0.5 - 0.35*tanh(1.5*theta);
if n1 >= 2.95 && n2 < 2.95
    r1 = 0.20;
elseif n2 >= 2.95 && n1 < 2.95
    r1 = 0.80;
elseif n1 >= 2.95 && n2 >= 2.95
    r1 = 0.50;
    admission = min(admission, 0.72);
end

y = [local_clip(gainScale, 0.80, 1.50);
     local_clip(r1, 0.20, 0.80);
     local_clip(admission, 0.70, 1.00)];
end

function y = local_clip(x, lo, hi)
y = min(max(x, lo), hi);
end
