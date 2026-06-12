function y = act08_r2et_fuzzy(x)
%ACT08_R2ET_FUZZY Politica difusa para coordinacion de dos esteras.
% Entrada: x = [e1; e2; d]. Salida: y = [u1; u2; r1].

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end
e1 = x(1);
e2 = x(2);
d = x(3);

u1 = act08_servo_fuzzy([e1; 0; d]);
u2 = act08_servo_fuzzy([e2; 0; d]);
r1 = 0.5 - 0.35*tanh(1.8*(e1 - e2));

y = [local_clip(u1, 0.05, 1.0);
     local_clip(u2, 0.05, 1.0);
     local_clip(r1, 0.20, 0.80)];
end

function y = local_clip(x, lo, hi)
y = min(max(x, lo), hi);
end
