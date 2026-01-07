# TFM Inteligencia Artificial (UTAMED)
## Título propuesto
Percepción RF confiable con modelos auto‑supervisados, calibración de incertidumbre, aprendizaje federado robusto y planificación cognitiva mediante gemelo digital y RL.

## 1. Problema
Los modelos RFML suelen degradar con cambios de canal/hardware y ante señales no vistas. Además, en escenarios distribuidos no es viable centralizar IQ; se requiere aprendizaje federado y mecanismos de confianza (assurance) para operar de forma segura.

## 2. Objetivo general
Desarrollar un marco reproducible de percepción RF que:
1) aprenda representaciones (foundation model pequeño) mediante auto‑supervisión,
2) incorpore assurance (calibración + open‑set/OOD),
3) soporte aprendizaje federado robusto (simulado) en no‑IID,
4) y use un gemelo digital para entrenar una política RL que optimice la selección de recursos de sensado bajo presupuesto computacional.

## 3. Contribuciones (propias de este TFM)
A1. RF foundation model pequeño (SSL) + evaluación en tareas downstream.  
A2. Assurance: ECE + riesgo‑cobertura + OOD/open‑set con métricas estándar.  
A3. Federated learning robusto: FedAvg baseline + agregación robusta y evaluación con nodos no‑IID/defectuosos.  
A4. Digital twin + RL: formulación MDP, comparación RL vs heurísticas bajo presupuesto fijo.

## 4. Alcance / No‑alcance
Alcance: RX‑only, datos públicos/sintéticos/capturas legales, simulación de nodos, RL para planificación de sensado (no emisión).  
No‑alcance: interceptación/decodificación de contenido, transmisión activa, técnicas ofensivas.

## 5. Criterios de éxito (DoD)
- Mejora de sample‑efficiency por SSL vs supervisado puro.
- Calibración mejorada (reducción ECE) y capacidad de abstención útil (riesgo‑cobertura).
- FL robusto supera FedAvg en no‑IID con nodos defectuosos (métricas definidas).
- RL supera heurística (p. ej., round‑robin) en "detecciones por coste computacional".

## 6. Riesgos y mitigación
Gemelo poco realista → definir límites + validar contra escenarios OOD y heurísticas.  
Coste de SSL → usar arquitectura pequeña y dataset controlado.
