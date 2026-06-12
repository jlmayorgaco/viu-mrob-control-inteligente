function u = act08_servo_fuzzy(x)
%ACT08_SERVO_FUZZY Control servo con Fuzzy Logic Toolbox y respaldo numerico.
% Entrada: x = [e; de; d]. Salida: u en [0.05, 1.0].

x = double(x(:));
if numel(x) < 3
    x(3) = 0;
end
e = local_clip(x(1), -0.75, 0.75);
de = local_clip(x(2), -0.35, 0.35);
d = local_clip(x(3), 0, 1);

persistent fis useFis
if isempty(useFis)
    useFis = exist('sugfis', 'file') == 2 && exist('evalfis', 'file') == 2;
    if useFis
        try
            fis = local_build_servo_fis();
        catch
            useFis = false;
        end
    end
end

if useFis
    try
        u = evalfis(fis, [e de d]);
    catch
        u = local_fuzzy_fallback(e, de, d);
    end
else
    u = local_fuzzy_fallback(e, de, d);
end

u = local_clip(double(u(1)), 0.05, 1.0);
end

function fis = local_build_servo_fis()
fis = sugfis('Name', 'act08_servo_fuzzy');
fis = addInput(fis, [-0.75 0.75], 'Name', 'e');
fis = addInput(fis, [-0.35 0.35], 'Name', 'de');
fis = addInput(fis, [0 1], 'Name', 'd');

eNames = {'NB','NM','NS','ZE','PS','PM','PB'};
eCenters = linspace(-0.75, 0.75, 7);
for i = 1:numel(eNames)
    fis = addMF(fis, 'e', 'trimf', local_tri(eCenters, i), 'Name', eNames{i});
end

deNames = {'NB','NS','ZE','PS','PB'};
deCenters = linspace(-0.35, 0.35, 5);
for i = 1:numel(deNames)
    fis = addMF(fis, 'de', 'trimf', local_tri(deCenters, i), 'Name', deNames{i});
end

fis = addMF(fis, 'd', 'trapmf', [0 0 0.25 0.55], 'Name', 'NO');
fis = addMF(fis, 'd', 'trapmf', [0.35 0.65 1 1], 'Name', 'YES');
fis = addOutput(fis, [0.05 1.0], 'Name', 'u');

uValues = [0.10 0.22 0.34 0.48 0.64 0.82 0.96];
for i = 1:numel(uValues)
    fis = addMF(fis, 'u', 'constant', uValues(i), 'Name', sprintf('U%d', i));
end

rules = zeros(35, 6);
r = 1;
for ie = 1:7
    for ide = 1:5
        outIdx = local_clip(round(ie + 0.65*(ide - 3)), 1, 7);
        rules(r,:) = [ie ide 0 outIdx 1 1];
        r = r + 1;
    end
end
fis = addRule(fis, rules);
end

function y = local_fuzzy_fallback(e, de, d)
% Aproximacion Sugeno equivalente cuando no esta disponible el toolbox.
base = 0.48 + 0.72*e + 0.22*de + 0.12*double(d > 0.2);
y = local_clip(base, 0.05, 1.0);
end

function tri = local_tri(centers, idx)
if idx == 1
    tri = [centers(1) centers(1) centers(2)];
elseif idx == numel(centers)
    tri = [centers(end-1) centers(end) centers(end)];
else
    tri = [centers(idx-1) centers(idx) centers(idx+1)];
end
end

function y = local_clip(x, lo, hi)
y = min(max(x, lo), hi);
end
