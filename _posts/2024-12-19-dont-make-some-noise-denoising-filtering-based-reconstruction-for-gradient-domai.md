---
title: "[(Don't) Make Some Noise: Denoising] Filtering-Based Reconstruction for Gradient-Domain Rendering"
date: 2024-12-19 22:26:58
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687568>

[Filtering-Based Reconstruction for Gradient-Domain Rendering | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687568)

### **Filtering-Based Reconstruction for Gradient-Domain Rendering**

Monte Carlo 렌더링은 사실적인 이미지를 생성하는 데 사용되지만, 노이즈 문제로 인해 고품질 재구성이 어렵습니다. 이번 SIGGRAPH Asia 2024에서 발표된 **"Filtering-Based Reconstruction for Gradient-Domain Rendering"** 논문은 Gradient-Domain 렌더링의 노이즈 문제를 해결하기 위해 고유의 필터링 기반 재구성 기법을 제안했습니다.

![](/assets/images/posts/441/img.jpg)

---

### **1. 논문의 주요 내용**

이 논문은 Gradient-Domain 정보를 활용하여 고품질 이미지를 재구성하기 위해 다음과 같은 혁신적인 방법을 제안했습니다:

- **Gradient-Guided Filtering**
  - 각 픽셀의 출력을 주변 픽셀의 가중 조합으로 모델링하고, 이를 최적화된 필터링 가중치를 계산하여 처리합니다.
  - 이를 통해 기존의 Poisson 방정식 기반 방법이 가지는 스파이크 아티팩트를 줄였습니다.
- **Coarse-to-Fine Strategy**
  - 다중 해상도를 활용한 단계별 접근법으로, 저해상도에서 시작해 고해상도로 점진적으로 업샘플링하며 세부 정보를 보존합니다.
- **Guided Linear Upsampling (GLU)**
  - 기존의 단순한 업샘플링 대신, GLU를 활용하여 다운샘플링과 업샘플링 과정에서 세부 정보를 최대한 유지합니다.
- **Per-Pixel Weighted Loss**
  - 픽셀별로 가중치를 적용한 손실 함수를 설계해, 부드러운 영역과 경계선을 더 효과적으로 처리했습니다.

---
