---
title: "[Make it Yours - Customizing Image Generation] Customizing Text-to-Image Diffusion with Object Viewpoint Control"
date: 2024-12-18 00:08:32
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687564>

[Customizing Text-to-Image Diffusion with Object Viewpoint Control | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687564)

이 논문은 \*\*3D 객체의 시점(Viewpoint)\*\*을 제어하면서도 Text-to-Image Diffusion 모델을 커스터마이징하는 방법을 제시합니다. 기존 2D 기반 모델이 **시점 제어**에 한계를 보인다는 점을 개선한 것이 핵심입니다.

---

### **핵심 기술**

1. **FeatureNeRF**:
   - 다수의 이미지 시점 데이터를 학습해 **3D Latent Feature**를 생성합니다.
   - 이를 통해 Diffusion 모델이 원하는 시점(Viewpoint)에서 객체를 렌더링할 수 있습니다.
2. **Pose-Conditioned Transformer**:
   - 객체의 시점 조건(예: 카메라 각도)을 Transformer 레이어에 추가합니다.
   - Text Prompt와 함께 시점 정보를 조건으로 활용해 **3D 시점 제어**를 가능하게 합니다.

---
