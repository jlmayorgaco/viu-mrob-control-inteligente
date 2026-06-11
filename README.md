# Actividad 08 MROB — Grupo 1 (Caso R2ET)

Control inteligente y supervisión avanzada para una Celda de Manufactura Flexible:
un robot que alimenta dos esteras transportadoras (R2ET), con capacidad de 3 piezas por estera.

## Métodos comparados

| Método | Descripción |
|--------|-------------|
| PID | PID discreto con anti-windup y feedforward; el supervisor ajusta sus ganancias por régimen |
| Fuzzy | Mamdani, 7 conjuntos de error × 5 de derivada (35 reglas), promedio ponderado de salida |
| Deep Learning | Red servo 4-64-32-16-8-1 (800 ép.) y red aware 8-96-64-32-16-8-3 (1200 ép.), entrenadas offline sobre el gemelo digital, sin frameworks de aprendizaje automático |
| Q-Learning | Tabla Q TD(0): servo 100 estados/8 acciones, aware 162 estados/11 acciones |

Planta de Fase 1: modelo reducido de primer orden obtenido de la dinámica dominante de la función de transferencia asignada,
`Gp(s) = (0.174 s + 0.3744)/(s^2 + 0.785 s + 0.3744)`, y discretizado por ZOH. Fase 2: balance de masa de las dos esteras, sin recorte superior de capacidad para medir violaciones.

## Estructura

- `src/python/v2_simulate.py` — simulación completa (NumPy): entrena, evalúa y genera figuras y tablas.
- `sections/` — capítulos del informe LaTeX; `main.tex` — documento principal.
- `figures/`, `artifacts/tables/`, `artifacts/data/` — salidas generadas por el script.
- `simulations/generar_simulink_act08_grupo1.m` — genera los modelos Simulink y exporta capturas.

## Reproducción

```bash
pip install -r requirements.txt
python src/python/v2_simulate.py          # genera figuras y tablas
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Resultados reproducibles con semilla fija (SEED=8).
