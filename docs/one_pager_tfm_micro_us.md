# TFM Microelectrónica (Universidad de Sevilla)
## Título propuesto
Co‑diseño HW/SW y aceleración en SoC FPGA de un receptor pasivo cognitivo para monitorización espectral basado en inferencia profunda cuantizada.

## 1. Problema
La percepción RF basada en ML en edge requiere ejecutar preprocesado y/o inferencia con restricciones de latencia, memoria y potencia. Un prototipo defendible debe medir prestaciones reales en SoC y justificar trade‑offs de microarquitectura.

## 2. Objetivo general
Diseñar, implementar e integrar un pipeline RX‑only donde al menos un bloque crítico se acelere en FPGA (DSP y/o inferencia), cuantificando: latencia E2E, throughput, utilización de recursos y frecuencia alcanzada.

## 3. Contribuciones (propias de este TFM)
M1. Arquitectura HW/SW del prototipo en SoC (bloques, interfaces, presupuesto de latencia, buffers).  
M2. Diseño microarquitectónico de un acelerador (HLS o RTL) para:
- opción preferida: núcleo de inferencia cuantizada (INT8/INT4) de un modelo pequeño, y/o
- opción alternativa: canalizador DSP (FFT/STFT/PFB) para generación de features.
M3. Integración en la plataforma objetivo (SoC) + verificación contra golden model.  
M4. Caracterización: LUT/DSP/BRAM, Fmax, latencias por bloque, throughput sostenido, y (si posible) potencia/temperatura.

## 4. Alcance / No‑alcance
Alcance: recepción pasiva, preprocesado baseband, inferencia cuantizada, integración y medición HW.  
No‑alcance: emisión activa, optimización a ASIC final, dataset clasificado.

## 5. Plataforma objetivo (prototipo)
SoC FPGA de bajo coste (p. ej., Zynq‑7020 o Zynq UltraScale+), ejecutando Linux/usuariospace y un datapath HW acelerado.

## 6. Validación y criterios de éxito (DoD)
- Baseline CPU‑only vs versión acelerada.
- Report de recursos + Fmax.
- Latencia E2E por ventana medida (y por bloque si posible).
- Evidencia de correctitud: comparación contra golden vectors (tolerancia definida).

## 7. Riesgos y mitigación
Si no hay cierre de timing → simplificar arquitectura y cuantización.  
Si streaming live complica → modo offline con IQ grabado para validación determinista.
