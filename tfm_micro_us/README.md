# TFM Microelectrónica - Universidad de Sevilla

## Co-diseño HW/SW y aceleración en SoC FPGA

Este módulo contiene el trabajo del TFM de Microelectrónica enfocado en:

- Arquitectura HW/SW del prototipo en SoC
- Diseño microarquitectónico de aceleradores (HLS o RTL)
- Integración en plataforma objetivo (Zynq)
- Caracterización de recursos y rendimiento

## Estructura

```
tfm_micro_us/
├── hw/          # Diseño hardware (HLS/RTL)
└── sw/          # Software de integración
```

## Contribuciones (M1-M4)

- **M1**: Arquitectura HW/SW (bloques, interfaces, buffers)
- **M2**: Diseño de acelerador (inferencia INT8/INT4 o DSP)
- **M3**: Integración SoC + verificación contra golden
- **M4**: Caracterización (LUT/DSP/BRAM, Fmax, latencias)

## Plataforma objetivo

SoC FPGA de bajo coste (Zynq-7020 o Zynq UltraScale+)
