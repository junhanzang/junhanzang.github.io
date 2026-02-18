---
title: "[Design it all: font, paint, and colors] Colorful Diffuse Intrinsic Image Decomposition in the Wild"
date: 2024-12-18 01:35:32
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2409.13690>

[Colorful Diffuse Intrinsic Image Decomposition in the Wild](https://arxiv.org/abs/2409.13690)

### **LVCD: 시간적 일관성을 잡아낸 라인아트 비디오 컬러화**

시그라프 아시아 2024에서 들었던 \*\*"LVCD: Reference-based Lineart Video Colorization with Diffusion Models"\*\*는 디퓨전 모델을 기반으로 **라인아트 비디오를 채색**하는 새로운 접근법을 제안했습니다.

---

### **핵심 아이디어**

1. **Reference Attention**
   - 기준 프레임의 색상 정보를 **Cross-Attention**을 통해 다음 프레임에 자연스럽게 전파합니다.
   - 이를 통해 프레임 간 색상 일관성을 유지하면서 고품질의 채색이 가능합니다.
2. **Sketch-Guided ControlNet**
   - 라인아트의 구조를 유지하며 세밀한 채색을 가능하게 합니다.
   - ControlNet 구조를 활용해 색상 정보와 스케치를 효과적으로 결합합니다.
3. **Overlapped Blending**
   - 프레임 경계를 **겹쳐서 생성**함으로써 에러 누적을 완화하고 부드러운 연결을 제공합니다.

---
