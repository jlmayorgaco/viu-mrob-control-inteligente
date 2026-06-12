# Paquete autocontenido MATLAB Online

Este directorio contiene todo lo necesario para regenerar los artefactos del
trabajo en MATLAB Online sin depender de rutas externas del computador local.

## Uso

1. Comprimir y subir esta carpeta completa a MATLAB Online.
2. Descomprimirla en MATLAB Drive.
3. Abrir `RUN_ME_MATLAB_ONLINE.m`.
4. Pulsar **Run**.
5. Descargar `resultados/entrega_control_inteligente_matlab.zip`.

## Salidas principales

- `figures/plots/*.png`: graficas de resultados.
- `figures/simulink/sl_*.png`: capturas de diagramas Simulink.
- `simulations/generated_models/*.slx`: modelos Simulink reconstruidos.
- `artifacts/tables/*.csv` y `artifacts/tables/*.tex`: tablas del informe.
- `artifacts/matlab_delivery/reporte_generacion.md`: reporte de ejecucion.
- `artifacts/matlab_delivery/estado_entorno.txt`: version de MATLAB y toolboxes detectados.
- `artifacts/matlab_delivery/entrega_control_inteligente_matlab.zip`: paquete para descargar.
- `resultados/`: copia directa de los archivos que conviene descargar desde MATLAB Online.

## Nota sobre PDF

El generador intenta compilar `main.tex` si el entorno tiene `pdflatex` y
`bibtex`. MATLAB Online normalmente no trae una distribucion LaTeX completa;
si ese paso falla, el reporte lo indicara y el resto de artefactos queda
generado para integrarlo en el documento local.
