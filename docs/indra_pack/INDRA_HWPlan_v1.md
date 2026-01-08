# INDRA Hardware Plan v1

**Proyecto**: Cognitive Trusted RF Receiver  
**Versión**: 1.0  
**Fecha**: 2024-12-15  
**Objetivo**: Definir particionado HW/SW y opciones de aceleración

---

## Resumen Ejecutivo

Este documento presenta dos opciones de aceleración hardware para el Cognitive RF Receiver:
- **Opción A**: Aceleración STFT en FPGA + CNN en CPU/NPU
- **Opción B**: Aceleración CNN completa en acelerador dedicado

Ambas opciones mantienen compatibilidad con el golden model Python existente.

---

## Arquitectura Actual (SW-only)

```
┌─────────────────────────────────────────────────────────┐
│                     HOST CPU                            │
├─────────────────────────────────────────────────────────┤
│  IQ Input → STFT → Normalize → CNN → Policy → Output   │
│    (4096)   (256,15)  (f32)    (5)    (label)          │
└─────────────────────────────────────────────────────────┘

Latencia actual (CPU): ~5-10 ms total
```

---

## Opción A: STFT en FPGA + CNN en CPU

### Diagrama de Datos

```
┌──────────────┐    AXI-Stream    ┌─────────────┐    DMA    ┌──────────┐
│   ADC/SDR    │ ───────────────► │ FPGA STFT   │ ────────► │   CPU    │
│  (IQ @ 1MS/s)│                  │ + Normalize │           │  (CNN)   │
└──────────────┘                  └─────────────┘           └──────────┘
                                        │
                                  BRAM: 8 KB
```

### Latency Budget

| Bloque | Descripción | Latencia (μs) | % Total |
|--------|-------------|---------------|---------|
| ADC → FPGA | Streaming IQ | 4 | 2% |
| STFT (256-pt FFT) | Radix-4, pipelined | 3 | 1.5% |
| Mag + Log | dB conversion | 2 | 1% |
| Normalize | Per-sample mean/std | 1 | 0.5% |
| DMA → CPU | 256×15×4 bytes | 10 | 5% |
| CNN Inference | ONNX Runtime | 150 | 75% |
| Policy | Threshold check | 1 | 0.5% |
| **TOTAL** | | **~170 μs** | |

### Interfaces

#### IQ Input (AXI-Stream)
```
tdata[31:0]   : {Q[15:0], I[15:0]} fixed-point
tvalid        : Sample valid
tlast         : End of frame (each 4096 samples)
```

#### Feature Output (AXI-Stream to DMA)
```
tdata[31:0]   : float32 spectrogram bin
tvalid        : Valid bin
tlast         : End of spectrogram
```

### Ventajas/Desventajas

| | Opción A |
|--|----------|
| ✅ | STFT determinista, bajo jitter |
| ✅ | CPU libre para policy/logging |
| ✅ | Fácil debug (STFT aislada) |
| ⚠️ | CNN sigue en CPU (~75% latencia) |

---

## Opción B: CNN Completo en Acelerador

### Diagrama de Datos

```
┌──────────────┐    AXI-Stream    ┌─────────────────────┐    IRQ    ┌──────────┐
│   ADC/SDR    │ ───────────────► │   CNN ACCELERATOR   │ ────────► │   CPU    │
│  (IQ @ 1MS/s)│                  │ STFT + Conv + FC    │           │ (Policy) │
└──────────────┘                  └─────────────────────┘           └──────────┘
                                        │
                                  BRAM: 64 KB + Weights
```

### Latency Budget

| Bloque | Descripción | Latencia (μs) | % Total |
|--------|-------------|---------------|---------|
| ADC → Accel | Streaming IQ | 4 | 10% |
| STFT | Pipelined | 3 | 7.5% |
| Conv Layer 1 | 32 filters, 3×3 | 10 | 25% |
| Conv Layer 2 | 64 filters, 3×3 | 15 | 37.5% |
| FC + Softmax | 5 classes | 5 | 12.5% |
| IRQ → CPU | Result ready | 2 | 5% |
| Policy | Threshold check | 1 | 2.5% |
| **TOTAL** | | **~40 μs** | |

### Ventajas/Desventajas

| | Opción B |
|--|----------|
| ✅ | Latencia 4× menor |
| ✅ | CPU casi libre |
| ⚠️ | Más complejidad HW |
| ⚠️ | Pesos en BRAM/DDR |

---

## Plan B (Fallback)

Si no hay recursos FPGA disponibles:

1. **CPU-only con SIMD**: Usar NEON/AVX para STFT y CNN (~5 ms)
2. **GPU embebida**: Jetson Nano puede lograr ~2 ms con TensorRT
3. **NPU comercial**: Google Coral, Intel NCS2 (~1 ms inference)

---

## Interfaces AXI Conceptuales

### Control de Bloque STFT (AXI-Lite)

| Offset | Nombre | Acceso | Descripción |
|--------|--------|--------|-------------|
| 0x00 | CTRL | R/W | [0]=Enable, [1]=Soft reset |
| 0x04 | STATUS | R | [0]=Busy, [1]=Done, [2]=Error |
| 0x08 | N_FFT | R/W | FFT size (default: 256) |
| 0x0C | HOP | R/W | Hop length (default: 64) |
| 0x10 | FRAME_CNT | R | Frames processed |

### Control CNN Accelerator

| Offset | Nombre | Acceso | Descripción |
|--------|--------|--------|-------------|
| 0x00 | CTRL | R/W | [0]=Start, [1]=Reset, [2]=IRQ_EN |
| 0x04 | STATUS | R | [0]=Busy, [1]=Done |
| 0x08 | INPUT_ADDR | R/W | DMA source address |
| 0x0C | OUTPUT_ADDR | R/W | DMA dest address |
| 0x10 | LOGITS[0-4] | R | Output logits (5× float32) |

---

## Buffer Sizes

| Buffer | Tamaño | Descripción |
|--------|--------|-------------|
| IQ Input FIFO | 8 KB | 4096 samples × 2 channels × 16 bit |
| FFT Twiddle | 1 KB | Pre-computed factors |
| Spectrogram | 15 KB | 256 × 15 × float32 |
| CNN Weights | 50 KB | Cuantizados INT8 |
| CNN Activations | 32 KB | Intermediate tensors |

---

## Verification Plan

### Golden Model

El software Python existente sirve como golden reference:

```python
# Golden model outputs
golden_logits = model.predict(features)
golden_probs = softmax(golden_logits / T)
golden_label = apply_policy(golden_probs, mode)
```

### Test Vectors

1. **Functional**: 100 muestras (20 por clase) del dataset v0
2. **Corner cases**: Señales con SNR extremo (-20 dB, +40 dB)
3. **OOD**: 50 muestras fuera de distribución

### Criterios de Aceptación

| Check | Criterio |
|-------|----------|
| Accuracy Match | |logits_hw - logits_sw| < 0.01 |
| Latency | < umbral de opción (40 o 170 μs) |
| Throughput | Sostenido @ 1 MS/s |
| Error Rate | Sin errores en 10⁶ frames |

### Flujo de Verificación

```
1. Generar test vectors desde Python
2. Ejecutar en simulación RTL (ModelSim/VCS)
3. Comparar outputs vs golden
4. Deploy en FPGA eval board
5. Medir latencia real con trigger scope
```

---

## Roadmap

| Fase | Duración | Entregable |
|------|----------|------------|
| **Fase 1** | 2 semanas | Spec congelada, testbench |
| **Fase 2** | 4 semanas | RTL STFT + verificación |
| **Fase 3** | 4 semanas | RTL CNN (si Opción B) |
| **Fase 4** | 2 semanas | Integración + demo FPGA |

---

*Documento preparado para demo INDRA — sin placa física disponible*
