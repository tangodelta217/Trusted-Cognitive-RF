# INDRA Benchmark Report

**Proyecto**: Cognitive Trusted RF Receiver  
**Versión**: 1.0  
**Fecha**: 2024-12-15  
**Destino**: Demostración INDRA

---

## Resumen Ejecutivo

1. **Cognitive RF Receiver** — Sistema de clasificación de señales RF con calibración y detección OOD.
2. La calibración mediante Temperature Scaling reduce el ECE de 0.1287 a 0.0760 (40.9% mejora).
3. Detección OOD con AUROC=0.6874 usando método entropy.
4. Tres modos operativos: SURVEILLANCE (95% cov), TRUSTED (85%), CONSERVATIVE (75%).
5. Señales de baja confianza reciben etiqueta UNKNOWN para revisión humana.
6. Latencia p50=0.29ms permite operación en tiempo real.
7. Sistema RX-only: sin transmisión ni interferencia activa.
8. ✅ **Listo para demo INDRA** — Todos los criterios WOW verificados.

---

## Métricas de Rendimiento

| Métrica | Valor | Umbral | Estado |
|---------|-------|--------|--------|
| ECE Improvement | 40.9% | ≥20% | ✅ |
| ECE After | 0.0760 | — | — |
| AUROC OOD | 0.6874 | ≥0.65 | ✅ |
| Latencia p50 | 0.29 ms | — | — |
| Latencia p99 | 0.47 ms | — | — |

---

## Figuras

### 1. Diagrama de Fiabilidad (Calibración)

Visualiza la calibración del modelo: barras de accuracy vs. confianza por bin.

![Reliability Diagram](../reports/figures/reliability.png)

### 2. Curva Risk-Coverage

Trade-off entre cobertura (muestras aceptadas) y riesgo (error rate).

![Risk-Coverage](../reports/figures/risk_coverage.png)

### 3. ROC de Detección OOD

Curva ROC para distinguir señales ID vs. OOD usando método de entropía/energía.

![OOD ROC](../reports/figures/ood_roc.png)

### 4. Demo: Visualización Espectrograma

Salida de la demo offline con predicción y comparación de modos.

![Waterfall Demo](../reports/demo/waterfall.png)

---

## Detalles Técnicos

### Configuración del Modelo

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | CNN Baseline |
| Clases | BPSK, QPSK, QAM16, GFSK, NOISE |
| Temperature (T) | 1.7358 |

### Modos Operativos

| Modo | Threshold (τ) | Coverage Target |
|------|---------------|-----------------|
| SURVEILLANCE | ~0.48 | 95% |
| TRUSTED | ~0.56 | 85% |
| CONSERVATIVE | ~0.65 | 75% |

---

## Criterios WOW

| Check | Criterio | Resultado |
|-------|----------|-----------|
| ECE Improvement | ≥20% reducción | ✅ PASS |
| AUROC OOD | ≥0.65 | ✅ PASS |
| Modos Operativos | 3 distintos | ✅ PASS |
| Demo Artifacts | waterfall + events | ✅ PASS |
| Bundle | preprocess + policy | ✅ PASS |

---

## Rutas de Artefactos

- **Informe**: `docs/indra_pack/INDRA_BenchmarkReport_v1.md`
- **Waterfall**: `reports/demo/waterfall.png`
- **Reliability**: `reports/figures/reliability.png`
- **Risk-Coverage**: `reports/figures/risk_coverage.png`
- **OOD ROC**: `reports/figures/ood_roc.png`
- **Latency**: `reports/metrics/latency.json`

---

*Generado automáticamente por `tools/make_report.py`*
