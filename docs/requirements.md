# Requisitos del proyecto (versión inicial)

| Req-ID | Descripción | Cómo se valida | Versión objetivo |
|---|---|---|---|
| R0 | Repo reproducible con estructura y documentación mínima | `tree` + README + docs | V0.1 |
| R1 | Dataset manifest + dataset card | `manifest.csv` + `dataset_card.md` | V0.2 |
| R2 | Pipeline de features determinista | pruebas de hash/tolerancia | V0.3 |
| R3 | Baseline ML cuantizable y exportable | training + export ONNX | V0.4 |
| R4 | Assurance mínimo (calibración + UNKNOWN) | ECE + riesgo-cobertura + OOD ROC | V0.5 |
| R5 | Demo reproducible end-to-end | `demo_v0.py` + vídeo | V0.7 |
| R6 | Ejecución en target CPU-only | logs en placa | V0.8 |
| R7 | Golden vectors para verificación HW | npz + referencia | V0.9 |
