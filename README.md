# Cognitive Trusted RF Receiver on Chip

> **Receptor RF cognitivo de confianza para monitorización espectral en edge.**

## Objetivo

Diseñar e implementar un prototipo end‑to‑end de receptor pasivo (RX‑only) que:

1. Ingiere datos IQ (offline y/o streaming)
2. Extrae representaciones (features/embeddings)
3. Produce eventos (clasificación/detección) con estimación de confianza
4. Marca **UNKNOWN** en señales fuera de distribución (OOD)
5. Permite despliegue y medida en edge (SoC FPGA)

## 🔒 Enfoque de seguridad y cumplimiento

Este proyecto opera **únicamente en modo recepción pasiva**:

- ❌ NO hay transmisión activa
- ❌ NO hay interferencia o jamming
- ❌ NO hay decodificación de contenido
- ✅ Solo capturas en bandas permitidas o pruebas cableadas/atenuadas

## Estructura del repositorio

```
cognitive-rf-receiver/
├── docs/                    # Documentación del proyecto
│   ├── one_pager_programa.md       # One-pager programa paraguas
│   ├── one_pager_tfm_micro_us.md   # One-pager TFM Microelectrónica US
│   ├── one_pager_tfm_ai_utamed.md  # One-pager TFM IA UTAMED
│   ├── dataset_card_v0.md          # Dataset card V0
│   ├── requirements.md             # Requisitos y validación
│   └── decisions.md                # Registro de decisiones
├── common/                  # Utilidades compartidas
│   ├── data/                # Generación y manejo de datos
│   │   ├── configs/         # Configuraciones YAML
│   │   ├── generators/      # Generadores de señales e impairments
│   │   ├── io.py            # Save/load NPZ
│   │   ├── manifest.py      # Generación de manifest CSV
│   │   └── make_dataset_v0.py  # Script principal de generación
│   ├── features/            # Extracción de features
│   │   ├── configs/         # Config features YAML
│   │   ├── golden/          # Golden examples
│   │   ├── tests/           # Unit tests
│   │   ├── preprocess.py    # DC removal, RMS normalize
│   │   ├── stft.py          # STFT
│   │   ├── spectrogram.py   # Power, log, normalize
│   │   └── extract.py       # API pública
│   └── __init__.py
├── tfm_micro_us/            # TFM Microelectrónica (Universidad de Sevilla)
│   ├── hw/                  # Diseño hardware (HLS/RTL)
│   └── sw/                  # Software de integración
├── tfm_ai_utamed/           # TFM Inteligencia Artificial (UTAMED)
│   ├── ssl/                 # Self-supervised learning
│   ├── assurance/           # Calibración, OOD, open-set
│   ├── federated/           # Federated learning robusto
│   └── rl_twin/             # Digital twin + RL
├── scripts/                 # Scripts de utilidad
│   ├── make_dataset_v0.py   # CLI para generar dataset
│   ├── make_golden_v0.py    # Generar golden features
│   └── verify_features_v0.py # Verificar pipeline features
└── data/                    # Datos generados (NO en git)
    └── datasets/v0/         # Dataset V0 sintético
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd cognitive-rf-receiver
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Generación del Dataset V0

El dataset sintético V0 contiene señales IQ moduladas con impairments de canal para entrenamiento y evaluación de OOD:

```bash
# Generar con preset TINY (rápido, para validación)
python scripts/make_dataset_v0.py --verify

# Generar con preset STANDARD (completo)
# Editar common/data/configs/dataset_v0.yaml: preset: STANDARD
python scripts/make_dataset_v0.py --verify
```

**Salida** (en `data/datasets/v0/`):
- `train.npz`, `val.npz`, `test_id.npz` — splits ID
- `test_ood_mod.npz` — modulaciones no vistas (8PSK, 64QAM, CPFSK)
- `test_ood_chan.npz` — canal más duro (SNR/CFO extremos)
- `manifest.csv` — metadatos por ejemplo

Ver [Dataset Card V0](docs/dataset_card_v0.md) para detalles completos.

## Extracción de Features (V0.3)

Pipeline determinista STFT para convertir IQ a tensor ML:

```bash
# Generar golden examples
python scripts/make_golden_v0.py

# Verificar pipeline
python scripts/verify_features_v0.py

# Ejecutar tests
python -m unittest common.features.tests.test_features_v0
```

**Pipeline**: IQ (2048) → DC removal → RMS norm → STFT → log power → normalize → (1,256,15) float32

Ver [Features V0](docs/features_v0.md) para detalles.

## Plan de versiones

| Versión | Descripción |
|---------|-------------|
| **V0.1** | Estructura del repo + documentación |
| **V0.2** | Dataset sintético reproducible |
| **V0.3** | Pipeline de features determinista (actual) |
| **V0.4** | Baseline ML cuantizable + ONNX |
| **V0.5** | Assurance mínimo (calibración + UNKNOWN) |
| **V1** | Aceleración FPGA de bloques críticos |
| **V2** | RF foundation model auto-supervisado |
| **V3** | Federated + digital twin + RL |

## Documentación

- [One-pager Programa Paraguas](docs/one_pager_programa.md)
- [One-pager TFM Microelectrónica US](docs/one_pager_tfm_micro_us.md)
- [One-pager TFM IA UTAMED](docs/one_pager_tfm_ai_utamed.md)
- [Dataset Card V0](docs/dataset_card_v0.md)
- [Features V0](docs/features_v0.md)
- [Requisitos del proyecto](docs/requirements.md)
- [Registro de decisiones](docs/decisions.md)

## Contribuciones

Este proyecto está organizado para evitar solape académico entre dos TFMs:

- **TFM Microelectrónica (US)**: Co-diseño HW/SW, aceleración FPGA, integración SoC
- **TFM IA (UTAMED)**: SSL, assurance, federated learning, RL/digital twin

## Licencia

*Por definir*

