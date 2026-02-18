---
title: "[Text, Texturing, and Stylization] Text-Guided Texturing by Synchronized Multi-View Diffusion"
date: 2024-12-19 02:28:59
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2311.12891>

[Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/abs/2311.12891)

### **Text-Guided Texturing by Synchronized Multi-View Diffusion: 텍스처 생성의 새로운 접근**

3D 텍스처 생성에서 시점 간 일관성 문제는 오랜 과제였습니다. **"Text-Guided Texturing by Synchronized Multi-View Diffusion"** 논문은 이러한 문제를 해결하기 위해 **Stable Diffusion** 기반의 다중 시점 동기화(Synchronized Multi-View Diffusion) 접근을 제안합니다. 이 방법은 텍스처 품질과 시점 간 일관성을 크게 향상시키는 혁신적인 방법론으로 평가받고 있습니다.

![](/assets/images/posts/430/img.jpg)

---

### **핵심 내용 요약**

1. **동기화된 잠재 공간 활용**
   - 기존 방식은 각 시점에서 독립적으로 텍스처를 생성한 뒤 이를 사후 보정(Post-Processing)하는 과정에서 **Seam(이음새)** 문제와 비일관성이 발생했습니다.
   - 본 논문은 Stable Diffusion의 **잠재 공간(Latent Space)**에서 겹치는 영역의 데이터를 공유 및 융합해, 초기부터 시점 간 구조적 합의를 도출합니다.
2. **Self-Attention Reuse 기법**
   - Stable Diffusion에서 각 시점의 **Self-Attention 정보**를 재활용해 텍스처와 구조적 정보를 강화합니다.
   - 이로 인해 텍스처의 세부 디테일과 시점 간 일관성이 대폭 개선되었습니다.
3. **성능 결과**
   - **FID(Frechet Inception Distance)**, **PSNR(Peak Signal-to-Noise Ratio)** 등 주요 평가 지표에서 기존 방식(T2I, Text2Tex) 대비 우수한 성능을 입증했습니다.
   - 특히 복잡한 3D 장면에서도 Seamless한 텍스처와 고품질 결과를 보여줬습니다.

---
