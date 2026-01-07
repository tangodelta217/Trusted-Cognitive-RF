
# Dataset Card — rf_v0_synth (V0)

## Resumen
Dataset sintético narrowband IQ para percepción RF y evaluación de OOD/open-set.
Generado proceduralmente con semilla fija y parámetros en `common/data/configs/dataset_v0.yaml`.

## Tareas previstas
- Clasificación de familia/modulación (baseline).
- Detección de OOD / open-set (UNKNOWN).
- Calibración de incertidumbre.

## Formato
- **IQ**: complex64, shape (N, 2048), fs = 1 Msps
- **Splits**: train / val / test_id / test_ood_mod / test_ood_chan
- **Manifest**: CSV por ejemplo con metadatos (SNR, CFO, canal, seed)

## Señal
| Parámetro | Valor |
|-----------|-------|
| Sample rate (fs) | 1,000,000 Hz |
| Muestras por ejemplo | 2048 |
| Samples por símbolo (sps) | 8 |
| Símbolos por ejemplo | 256 |
| Pulse shaping | RRC (rolloff=0.35, span=8) |

## Clases

### ID (In-Distribution) - Entrenamiento
| Clase | Descripción |
|-------|-------------|
| BPSK | Binary PSK |
| QPSK | Quadrature PSK |
| QAM16 | 16-QAM |
| GFSK | Gaussian FSK |
| NOISE | Ruido puro (idle) |

### OOD-MOD (Out-of-Distribution - Semántico)
Modulaciones **no vistas** durante entrenamiento:
| Clase | Descripción |
|-------|-------------|
| PSK8 | 8-PSK |
| QAM64 | 64-QAM |
| CPFSK | Continuous Phase FSK |

### OOD-CHAN (Out-of-Distribution - Canal)
Clases ID pero con impairments fuera del rango de entrenamiento.

## Impairments

### Rangos ID (train/val/test_id)
| Impairment | Rango |
|------------|-------|
| SNR (dB) | [-2, 18] |
| CFO (Hz) | [-200, 200] |
| Ganancia | [0.7, 1.3] |
| Fase (rad) | [0, 2π) |
| Multipath taps | 1-2 |
| Max delay (samples) | 8 |

### Rangos OOD-CHAN (test_ood_chan)
| Impairment | Rango |
|------------|-------|
| SNR (dB) | [-10, 24] |
| CFO (Hz) | [-2000, 2000] |
| Ganancia | [0.5, 1.6] |
| Fase (rad) | [0, 2π) |
| Multipath taps | 3-7 |
| Max delay (samples) | 32 |

## Tamaños (Presets)

### TINY (validación rápida)
| Split | Por clase | Total |
|-------|-----------|-------|
| train | 300 | 1,500 |
| val | 60 | 300 |
| test_id | 100 | 500 |
| test_ood_mod | 100 | 300 |
| test_ood_chan | — | 200 |
| **Total** | — | **2,800** |

### STANDARD (entrenamiento completo)
| Split | Por clase | Total |
|-------|-----------|-------|
| train | 2,000 | 10,000 |
| val | 400 | 2,000 |
| test_id | 500 | 2,500 |
| test_ood_mod | 500 | 1,500 |
| test_ood_chan | — | 2,000 |
| **Total** | — | **18,000** |

## Reproducibilidad
- **Seed global**: 20251224
- Cada ejemplo tiene seed determinista: `base_seed + offset + index`
- Regenerar: mismo config + seed → mismos NPZ y manifest

## Archivos generados

```
data/datasets/v0/
├── train.npz          # Split de entrenamiento
├── val.npz            # Split de validación
├── test_id.npz        # Test in-distribution
├── test_ood_mod.npz   # Test OOD semántico
├── test_ood_chan.npz  # Test OOD canal
└── manifest.csv       # Metadatos por ejemplo
```

## Generación

```bash
cd "c:\Users\User\Desktop\TFM Indra"
python scripts/make_dataset_v0.py --config common/data/configs/dataset_v0.yaml --out data/datasets/v0 --verify
```

## Licencia y uso
Datos sintéticos generados por scripts del proyecto.
No incluye contenido decodificado ni señales capturadas de emisiones reales.
