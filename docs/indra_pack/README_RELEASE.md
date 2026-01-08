# Cognitive Trusted RF Receiver — Release Notes INDRA

**Versión**: 0.7-demo  
**Fecha**: 2024-12-15  
**Destino**: Demostración INDRA  

---

## 🎯 Resumen Ejecutivo

Sistema de receptor RF cognitivo pasivo (RX-only) con:
- **Clasificación de señales** con 5 modulaciones (BPSK, QPSK, 16QAM, GFSK, NOISE)
- **Estimación de confianza calibrada** (Temperature Scaling)
- **Detección de señales desconocidas** (OOD) con etiqueta UNKNOWN
- **3 modos operativos** configurables según criticidad de decisión

---

## 🚀 Quickstart (3 comandos)

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m tools.wow_check --verbose
```

---

## ✅ Verificación WOW

El comando `python -m tools.wow_check` verifica:

| Check | Criterio | Estado |
|-------|----------|--------|
| ECE Improvement | ≥20% reducción relativa | ✓ |
| AUROC OOD | ≥0.65 (entropy/energy) | ✓ |
| Operating Modes | 3 modos distintos | ✓ |

---

## 📊 Modos de Operación

| Modo | Coverage | Umbral τ | Descripción |
|------|----------|----------|-------------|
| SURVEILLANCE | 95% | 0.48 | Alta cobertura, pocas abstenciones |
| TRUSTED | 80% | 0.60 | Balance confianza/cobertura |
| CONSERVATIVE | 70% | 0.68 | Alta precisión, más UNKNOWN |

---

## 📁 Estructura Clave

```
├── tools/wow_check.py    # Verificador WOW
├── tests/                # Pytest tests
├── Makefile              # make setup/test/demo/wow
└── runs/wow_check/       # Artefactos generados
```

---

## 🔒 Cumplimiento de Restricciones

- ✅ RX-only: Sin transmisión ni interferencia
- ✅ Sin decodificación de contenido
- ✅ Datos sintéticos reproducibles
- ✅ Offline: Sin dependencias externas

---

## 📞 Contacto

Proyecto TFM — Universidad de Sevilla / UTAMED  
Para INDRA — Enero 2026
