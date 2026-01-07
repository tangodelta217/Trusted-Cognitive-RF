# Cognitive Trusted RF Receiver on Chip (Programa paraguas)

## 1. Título
Cognitive Trusted RF Receiver on Chip: percepción RF confiable en edge con co‑diseño HW/SW, assurance, aprendizaje federado y control cognitivo.

## 2. Contexto y motivación
En entornos dinámicos de espectro, un receptor pasivo necesita: (i) procesar datos IQ/tiempo‑frecuencia con baja latencia, (ii) inferir eventos con modelos ML en edge bajo restricciones de cómputo/energía, y (iii) operar de forma confiable: saber cuándo su decisión no es fiable (open‑set/OOD). Este proyecto construye un prototipo reproducible que integra percepción, assurance y despliegue en plataforma SoC.

**Enfoque de seguridad y cumplimiento**: únicamente recepción pasiva (RX‑only). No se aborda emisión activa, interferencia, ni decodificación de contenido. Capturas solo en bandas permitidas o mediante pruebas cableadas/atenuadas.

## 3. Objetivo general
Diseñar e implementar un prototipo end‑to‑end de receptor pasivo que:
1) ingiere IQ (offline y/o streaming),
2) extrae representaciones (features/embeddings),
3) produce eventos (clasificación/detección) con estimación de confianza,
4) marca UNKNOWN en señales fuera de distribución,
5) y permite despliegue y medida en edge (SoC FPGA) para cuantificar latencia/throughput/recursos.

## 4. Contribuciones del programa (artefactos comunes)
C1. Dataset manifest + dataset card (especificación reproducible: sampling, ventanas, clases, splits ID/OOD).  
C2. Pipeline de preprocesado y evaluación (scripts deterministas, métricas y plots).  
C3. Modelo baseline cuantizable + export ONNX (workload común).  
C4. Demo reproducible: IQ → waterfall → eventos (label/confianza/UNKNOWN) + latencias por bloque.  
C5. Golden vectors para verificación HW (entrada/salida de bloques a acelerar).  

## 5. Alcance y no‑alcance
**Alcance**: percepción RF (clasificación/detección), OOD/open‑set, despliegue edge, validación reproducible, simulación de FL y RL.  
**No‑alcance**: transmisión activa, técnicas ofensivas, datasets clasificados, interceptación de contenido.

## 6. Métricas objetivo (a fijar en V0.2/V0.3)
- Precisión/F1 en ID (tarea básica).
- OOD/open‑set: AUROC/AUPR y riesgo‑cobertura.
- Calibración: ECE y reliability diagram.
- Edge: latencia E2E por ventana, throughput sostenido, recursos (LUT/DSP/BRAM) y Fmax.

## 7. Plan por versiones
V0: pipeline end‑to‑end reproducible (offline), baseline + assurance mínimo + demo + ejecución CPU‑only (PC y, si posible, board).  
V1: aceleración FPGA de bloques críticos (DSP y/o inferencia) + medición HW completa.  
V2: RF foundation model auto‑supervisado (pequeño) + mejoras en open‑set.  
V3: federated robusto simulado + digital twin + RL para planificación de sensado.

## 8. Riesgos y mitigaciones
R1. Datasets insuficientes → usar combinación de datos públicos + sintético controlado + capturas en ISM.  
R2. Latencia/recursos en FPGA → cuantización, reducción de arquitectura, particionado HW/SW.  
R3. RL/gemelo poco realista → comparar contra heurísticas simples y fijar límites de validez.

## 9. Entregables finales del programa
- Repositorio con scripts reproducibles, modelos exportables y documentación.
- Memorias separadas (Micro US / IA UTAMED) con contribuciones propias.
- Demostrador (vídeo + ejecución) con logs y métricas.
