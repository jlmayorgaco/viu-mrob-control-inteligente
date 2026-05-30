"""Genera fig_smoothing.png: control original (chattering) vs versión suavizada
(DL: penalización de suavidad reforzada + limitador de pendiente; QL: recompensa
que penaliza |Delta u|). Experimento adicional; no altera los resultados base."""
import importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('v2', ROOT / 'src/python/v2_simulate.py')
m = importlib.util.module_from_spec(spec); sys.modules['v2'] = m; spec.loader.exec_module(m)
plt = m.plt


def varu(res):
    u = res['u']
    return float(np.sum(np.abs(np.diff(u[:, 0]))) + np.sum(np.abs(np.diff(u[:, 1]))))


df = pd.read_csv(ROOT / 'artifacts/data/timeseries_aware.csv')
t = df['t'].to_numpy()
DLo, QLo = df['DL_u1'].to_numpy(), df['QL_u1'].to_numpy()
dlo_v = float(np.sum(np.abs(np.diff(df['DL_u1']))) + np.sum(np.abs(np.diff(df['DL_u2']))))
qlo_v = float(np.sum(np.abs(np.diff(df['QL_u1']))) + np.sum(np.abs(np.diff(df['QL_u2']))))

# DL mejorado: lambda_u reforzado + limitador de pendiente en despliegue
dl, _ = m.train_dl_aware_direct(layer_sizes=[8, 96, 64, 32, 16, 8, 3],
                                n_epochs=600, batch_size=16, lambda_u=0.4, seed=m.SEED + 1)
rds = m.simulate_aware_dl(dl, du_max=0.06); mds = m.aware_metrics(rds)
# QL mejorado: recompensa con penalizacion de |Delta spd|
Qs, _ = m.train_q_aware_smooth(w_slew=4.0); rqs = m.simulate_aware_ql(Qs); mqs = m.aware_metrics(rqs)
DLs, QLs = rds['u'][:, 0], rqs['u'][:, 0]

print('METRICS')
print('DL  original Var_u=%.1f | mejorado IAE=%.1f Var_u=%.1f Viol=%s' % (dlo_v, mds['IAE'], varu(rds), mds['Viol']))
print('QL  original Var_u=%.1f | mejorado IAE=%.1f Var_u=%.1f Viol=%s' % (qlo_v, mqs['IAE'], varu(rqs), mqs['Viol']))

plt.rcParams.update(m.PLT_RC)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ax[0].plot(t, DLo, color='#f3a3a3', lw=1.0, label='Original (Var$_u$=%.0f)' % dlo_v)
ax[0].plot(t, DLs, color='#dc2626', lw=1.7, label='Suavizado (Var$_u$=%.0f)' % varu(rds))
ax[0].set_title('Deep Learning — velocidad E1'); ax[0].set_xlabel('Muestra $k$')
ax[0].set_ylabel('Velocidad E1 (norm.)'); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.05)
ax[1].plot(t, QLo, color='#9ad7bd', lw=1.0, label='Original (Var$_u$=%.0f)' % qlo_v)
ax[1].plot(t, QLs, color='#059669', lw=1.7, label='Suavizado (Var$_u$=%.0f)' % varu(rqs))
ax[1].set_title('Q-Learning — velocidad E1'); ax[1].set_xlabel('Muestra $k$')
ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.05)
fig.suptitle(r'Suavizado del mando: penalización de $|\Delta u|$ (DL: pérdida + limitador; QL: recompensa)',
             fontsize=10, fontweight='bold')
fig.tight_layout(); m._sv(fig, 'fig_smoothing.png')
print('OK fig_smoothing.png')
