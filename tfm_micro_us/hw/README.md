# Hardware Accelerator Module

## Overview

This directory contains specifications for potential hardware acceleration of the Cognitive RF Receiver signal processing pipeline.

## Contents

| File | Description |
|------|-------------|
| `spec.md` | Detailed hardware specification |
| `README.md` | This file |

## Architecture Options

### Option A: STFT Acceleration Only
- FPGA-based STFT with streaming interface
- CNN remains on CPU/NPU
- Latency: ~170 μs total

### Option B: Full CNN Acceleration  
- Complete pipeline in dedicated accelerator
- CPU only handles policy decisions
- Latency: ~40 μs total

## Interfaces

All hardware blocks use AXI4 family interfaces:
- **AXI-Stream**: IQ data and feature tensors
- **AXI-Lite**: Control/status registers
- **AXI-Full**: DMA for weight loading

## Golden Model

The Python implementation in `tfm_ai_utamed/` serves as the golden reference for hardware verification.

## Status

📋 **Specification Phase** — No physical hardware available for demo.

See `docs/indra_pack/INDRA_HWPlan_v1.md` for full details.
