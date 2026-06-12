# Simulaciones - Actividad 08 MROB, Grupo 1 (R2ET)

Cuatro metodos de control: PID, Fuzzy, Deep Learning y Q-Learning.

El generador integral `generar_entrega_10_10` reconstruye tablas, figuras,
modelos Simulink, capturas, PDF y paquete de entrega. El generador
`generar_trabajo_guia_simulink` queda como subrutina para crear los modelos
Simulink y sus capturas.

Los bloques de politica del modelo llaman funciones MATLAB externas:

- `act08_servo_fuzzy`, `act08_servo_dl`, `act08_servo_ql`.
- `act08_r2et_fuzzy`, `act08_r2et_dl`, `act08_r2et_ql`.
- `act08_supervisor_fuzzy`.

Uso en MATLAB desde la raiz del proyecto:

```matlab
addpath(genpath('src/matlab'))
addpath(genpath('simulations'))
generar_entrega_10_10
```

Ejecucion por lote, si `matlab` esta en el PATH:

```powershell
matlab -batch "addpath(genpath('src/matlab')); addpath(genpath('simulations')); generar_entrega_10_10"
```

Salidas generadas:

- `figures/simulink/sl_*.png`: capturas insertadas en el PDF.
- `simulations/generated_models/*.slx`: modelos Simulink reconstruidos por el script.
- `artifacts/matlab_delivery/reporte_generacion.md`: reporte de ejecucion.
- `artifacts/matlab_delivery/entrega_control_inteligente_matlab.zip`: paquete para revision.
- `resultados/entrega_control_inteligente_matlab.zip`: copia directa para descargar desde MATLAB Online.
