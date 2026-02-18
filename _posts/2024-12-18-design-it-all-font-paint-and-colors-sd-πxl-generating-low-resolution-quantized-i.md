---
title: "[Design it all: font, paint, and colors] SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation"
date: 2024-12-18 00:54:43
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/full/10.1145/3680528.3687570>

[SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/full/10.1145/3680528.3687570)

### **SD-πXL: 픽셀 이미지 생성의 새로운 시도**

이번 시그라프 아시아 2024에서 다룬 SD-πXL는 픽셀 이미지 생성에 집중한 논문으로, Diffusion 기반의 **Score Distillation Sampling (SDS)**과 **Discrete Quantization**이 결합된 점이 특징이었습니다.

---

### **핵심 아이디어: 픽셀 단위 양자화와 SDS**

1. **Condition 기반 생성**
   - 입력으로 **Canny Edge, Depth Map**과 같은 보조 정보를 활용해 구조적 특성을 유지합니다.
   - 이를 통해 **디노이징** 과정이 픽셀 팔레트의 양자화된 색상 정보를 조건으로 활용할 수 있습니다.
2. **Softmax → Gumbel-Softmax → Convex Sum**
   - **Softmax**: 연속적인 확률 분포를 예측합니다.
   - **Gumbel-Softmax**: 샘플링을 통해 연속 값을 **Discrete Representation**으로 변환합니다.
   - **Convex Sum**: 최종적으로 팔레트의 색상 중 가장 적합한 값을 결합해 픽셀 단위로 색을 결정합니다.
3. **개별 픽셀 최적화**
   - 각 픽셀 단위로 색상 정보를 **양자화**(Quantization)하는 구조이기 때문에 정교한 결과를 얻을 수 있습니다.

---
