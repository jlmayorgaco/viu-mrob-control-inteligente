# Actividad 08 MROB - Grupo 1 (Caso R2ET)

Control inteligente y supervision avanzada para una Celda de Manufactura Flexible:
un robot alimenta dos esteras transportadoras R2ET, con capacidad de 3 piezas por estera.

## Metodos comparados

| Metodo | Descripcion |
|--------|-------------|
| PID | PID discreto con anti-windup y feedforward; el supervisor ajusta sus ganancias por regimen |
| Fuzzy | Mamdani, 7 conjuntos de error x 5 de derivada (35 reglas), promedio ponderado de salida |
| Deep Learning | Red servo 4-64-32-16-8-1 (800 ep.) y red aware 8-96-64-32-16-8-3 (1200 ep.), entrenadas offline sobre el gemelo digital, sin frameworks de aprendizaje automatico |
| Q-Learning | Tabla Q TD(0): servo 100 estados/8 acciones, aware 162 estados/11 acciones |

Planta de Fase 1: modelo reducido de primer orden obtenido de la dinamica dominante
de la funcion de transferencia asignada, `Gp(s) = (0.174 s + 0.3744)/(s^2 + 0.785 s + 0.3744)`,
y discretizado por ZOH. Fase 2: balance de masa de las dos esteras, sin recorte
superior de capacidad para medir violaciones.

## Estructura

- `sections/`: capitulos del informe LaTeX; `main.tex`: documento principal.
- `figures/`, `artifacts/tables/`, `artifacts/data/`: figuras, tablas y series usadas por el informe.
- `simulations/generar_entrega_10_10.m`: genera tablas, figuras, modelos Simulink, capturas, PDF y paquete de entrega.
- `simulations/generar_trabajo_guia_simulink.m`: genera modelos Simulink y capturas.
- `simulations/generated_models/`: modelos `.slx` reconstruidos por el generador.
- `src/matlab/`: punto de entrada y funciones MATLAB llamadas desde Simulink.
- `matlab_online/`: paquete autocontenido para subir a MATLAB Online y descargar resultados.

Funciones llamadas por los bloques Simulink:

- Servo: `act08_servo_fuzzy`, `act08_servo_dl`, `act08_servo_ql`.
- R2ET: `act08_r2et_fuzzy`, `act08_r2et_dl`, `act08_r2et_ql`.
- Supervisor: `act08_supervisor_fuzzy`.

## Reproduccion

En MATLAB, desde la raiz del proyecto:

```matlab
addpath(genpath('src/matlab'))
addpath(genpath('simulations'))
generar_entrega_10_10
```

Ejecucion por lote, si `matlab` esta en el PATH:

```powershell
matlab -batch "addpath(genpath('src/matlab')); addpath(genpath('simulations')); generar_entrega_10_10"
```

El generador compila el informe y actualiza `Grupo_1_Act_08MPRO.pdf`. Si se
quiere repetir solo la compilacion LaTeX de forma manual:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Resultados reproducibles con parametros fijos y semilla de simulacion constante.

Para MATLAB Online, comprimir y subir la carpeta `matlab_online/`, abrir
`RUN_ME_MATLAB_ONLINE.m` y descargar `resultados/entrega_control_inteligente_matlab.zip`.
