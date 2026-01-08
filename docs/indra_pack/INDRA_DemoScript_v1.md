# INDRA Demo Script v1

## Preparación (5 min)

### Instalación
```bash
pip install -r requirements.txt
pip install -r requirements_ml.txt
```

### Verificación rápida
```bash
python -m pytest tests/ -q
# Esperado: 24+ passed
```

---

## Demo Offline (30 segundos)

### Comando principal
```bash
python -m tools.demo
```

### Lo que genera:
1. `reports/demo/waterfall.png` — Espectrograma
2. `reports/demo/events.json` — Log de eventos

### Salida esperada:
```
COGNITIVE RF RECEIVER — OFFLINE DEMO
====================================
Temperature: 1.7358
Thresholds: {'SURVEILLANCE': 0.48, 'TRUSTED': 0.56, 'CONSERVATIVE': 0.65}

Processing ID sample...
  SURVEILLANCE: BPSK (conf=0.963, τ=0.483)
  TRUSTED: BPSK (conf=0.963, τ=0.562)
  CONSERVATIVE: BPSK (conf=0.963, τ=0.646)

Processing OOD sample...
  SURVEILLANCE: QPSK (conf=0.xxx, τ=0.483)
  TRUSTED: ⚠ UNKNOWN (conf=0.xxx, τ=0.562)
  CONSERVATIVE: ⚠ UNKNOWN (conf=0.xxx, τ=0.646)

DEMO SUMMARY
============
Total events: 6
UNKNOWN detections: N
✓ UNKNOWN detected in CONSERVATIVE mode
```

---

## Comandos alternativos

### Un solo modo
```bash
python -m tools.demo --mode conservative
```

### WOW Check completo
```bash
python -m tools.wow_check --verbose
```

### Calibración de umbrales
```bash
python -m tools.calibrate_thresholds
```

### Evaluación OOD
```bash
python -m tools.ood_eval
```

---

## Artefactos para mostrar

| Archivo | Descripción |
|---------|-------------|
| `reports/demo/waterfall.png` | Visualización espectrograma |
| `reports/figures/reliability.png` | Diagrama calibración |
| `reports/figures/risk_coverage.png` | Curva risk-coverage |
| `reports/figures/ood_roc.png` | ROC detección OOD |

---

## Puntos clave de la demo

1. **Calibración**: ECE mejora 40+% con Temperature Scaling
2. **3 modos**: SURVEILLANCE (95% cov), TRUSTED (85%), CONSERVATIVE (75%)
3. **UNKNOWN**: Señales dudosas se abstienen → operador humano
4. **RX-only**: Sistema pasivo, sin transmisión
