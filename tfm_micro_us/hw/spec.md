# Hardware Specification — Cognitive RF Receiver

## Version: 1.0

---

## 1. Overview

This document specifies the hardware interface and resource requirements for accelerating the Cognitive RF Receiver signal processing pipeline.

---

## 2. Register Map (AXI-Lite)

### 2.1 STFT Accelerator Block

**Base Address**: 0x4000_0000

| Offset | Name | Width | Access | Description |
|--------|------|-------|--------|-------------|
| 0x00 | CTRL | 32 | R/W | Control register |
| | | | | [0] = Enable (1=run, 0=stop) |
| | | | | [1] = Soft Reset |
| | | | | [2] = IRQ Enable |
| 0x04 | STATUS | 32 | R | Status register |
| | | | | [0] = Busy |
| | | | | [1] = Done |
| | | | | [2] = Error |
| | | | | [7:4] = Error code |
| 0x08 | CONFIG | 32 | R/W | Configuration |
| | | | | [7:0] = log2(N_FFT) - 1 |
| | | | | [15:8] = log2(HOP) - 1 |
| | | | | [16] = Window type (0=Hann) |
| 0x0C | FRAME_CNT | 32 | R | Frames processed |
| 0x10 | INPUT_PTR | 32 | R/W | DMA source (IQ buffer) |
| 0x14 | OUTPUT_PTR | 32 | R/W | DMA dest (spectrogram) |

### 2.2 CNN Accelerator Block

**Base Address**: 0x4001_0000

| Offset | Name | Width | Access | Description |
|--------|------|-------|--------|-------------|
| 0x00 | CTRL | 32 | R/W | Control register |
| | | | | [0] = Start inference |
| | | | | [1] = Abort |
| | | | | [2] = IRQ Enable |
| 0x04 | STATUS | 32 | R | Status register |
| | | | | [0] = Busy |
| | | | | [1] = Done |
| | | | | [7:4] = Layer index |
| 0x08 | INPUT_ADDR | 32 | R/W | Feature input address |
| 0x0C | OUTPUT_ADDR | 32 | R/W | Logits output address |
| 0x10 | WEIGHT_ADDR | 32 | R/W | Weights base address |
| 0x20 | LOGITS_0 | 32 | R | Output logit class 0 (float32) |
| 0x24 | LOGITS_1 | 32 | R | Output logit class 1 |
| 0x28 | LOGITS_2 | 32 | R | Output logit class 2 |
| 0x2C | LOGITS_3 | 32 | R | Output logit class 3 |
| 0x30 | LOGITS_4 | 32 | R | Output logit class 4 |

---

## 3. Buffer Sizes

### 3.1 STFT Block

| Buffer | Size | Type | Description |
|--------|------|------|-------------|
| IQ Input FIFO | 8 KB | BRAM | 4096 samples × I/Q × 16-bit |
| Twiddle Factors | 1 KB | ROM | FFT coefficients |
| Window LUT | 512 B | ROM | Hann window values |
| Output FIFO | 16 KB | BRAM | Spectrogram (256×15×f32) |

### 3.2 CNN Block

| Buffer | Size | Type | Description |
|--------|------|------|-------------|
| Feature Input | 16 KB | BRAM | 256×15×f32 |
| Weights (L1) | 4.6 KB | ROM/DDR | 32×1×3×3 INT8 + bias |
| Weights (L2) | 18.4 KB | ROM/DDR | 64×32×3×3 INT8 + bias |
| Weights (FC) | 20 KB | ROM/DDR | 64×8×5 INT8 + bias |
| Activations | 32 KB | BRAM | Intermediate tensors |

**Total BRAM**: ~96 KB

---

## 4. Interfaces

### 4.1 AXI-Stream (IQ Input)

```
Signal       Width    Direction    Description
-------------------------------------------------------
s_axis_tdata   32     Slave →      {Q[15:0], I[15:0]}
s_axis_tvalid   1     Slave →      Data valid
s_axis_tready   1     ← Master     Ready to accept
s_axis_tlast    1     Slave →      End of frame
```

### 4.2 AXI-Stream (Spectrogram Output)

```
Signal       Width    Direction    Description
-------------------------------------------------------
m_axis_tdata   32     ← Master     float32 bin value
m_axis_tvalid   1     ← Master     Data valid
m_axis_tready   1     Slave →      Downstream ready
m_axis_tlast    1     ← Master     End of spectrogram
m_axis_tuser    8     ← Master     [7:4]=row, [3:0]=col
```

---

## 5. Latency Budget

### Option A (STFT HW + CNN SW)

| Stage | Latency | Notes |
|-------|---------|-------|
| IQ → STFT | 4 μs | Streaming |
| FFT 256pt | 3 μs | Radix-4 pipelined |
| Mag + dB | 2 μs | LUT-based log |
| Norm | 1 μs | Per-sample |
| DMA out | 10 μs | 15 KB @ 1.5 GB/s |
| CNN (CPU) | 150 μs | ONNX Runtime |
| **Total** | **~170 μs** | |

### Option B (Full HW)

| Stage | Latency | Notes |
|-------|---------|-------|
| IQ → STFT | 4 μs | Streaming |
| FFT 256pt | 3 μs | Pipelined |
| Conv L1 | 10 μs | 32 filters |
| Conv L2 | 15 μs | 64 filters |
| FC + Softmax | 5 μs | 5 classes |
| IRQ | 2 μs | Result ready |
| **Total** | **~40 μs** | |

---

## 6. Verification Plan

### Golden Reference

Python implementation in `tfm_ai_utamed/` is the golden model.

### Test Vectors

```python
# Generate test vector
np.savez("hw_testvec.npz", iq=iq_samples, expected_logits=logits)
```

### Pass Criteria

| Check | Criterion |
|-------|-----------|
| Functional | max|logits_hw - logits_golden| < 0.01 |
| Latency | < target for chosen option |
| Throughput | 1 frame / 1 ms continuous |
| Stability | No errors in 10^6 frames |

---

## 7. Plan B (Fallback)

If FPGA resources unavailable:

1. **Pure SW**: SIMD-optimized (5-10 ms)
2. **Edge NPU**: Google Coral USB (~1 ms)
3. **GPU**: Jetson Nano TensorRT (~2 ms)

---

*Specification for INDRA demo — conceptual design without physical hardware*
