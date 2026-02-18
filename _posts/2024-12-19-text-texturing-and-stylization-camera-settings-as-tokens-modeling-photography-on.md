---
title: "[Text, Texturing, and Stylization] Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models"
date: 2024-12-19 02:50:21
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687635>

[Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687635)

### **Camera Settings as Tokens: AI와 사진의 새로운 융합**

**텍스트-투-이미지 생성 모델**은 예술적 창작에서 혁신을 가져왔지만, 실제 사진 촬영의 물리적 요소를 반영하는 데는 한계가 있었습니다. **"Camera Settings as Tokens"**는 이러한 문제를 해결하기 위해 **카메라 설정(초점 거리, 조리개 값, ISO 등)을 텍스트 토큰으로 통합**하는 방식을 제안하며, AI를 통해 사진의 물리적 제어 가능성을 확장한 연구입니다.

![](/assets/images/posts/434/img.jpg)

---

### **핵심 기법 및 특징**

1. **카메라 설정의 텍스트 토큰화**
   - 카메라 설정을 텍스트 공간에 통합하여, **Latent Diffusion Models (LDMs)**이 사진 촬영의 물리적 원칙을 이해하고 반영할 수 있도록 설계되었습니다.
   - **LoRA(저랭크 어댑터)**를 활용하여, 텍스트 프롬프트와 물리적 설정이 조화를 이루도록 학습했습니다.
2. **CameraSettings20k 데이터셋 구축**
   - 20,000개 이상의 RAW 이미지를 기반으로 초점 거리, 조리개 값, ISO 등 표준화된 촬영 설정과 함께 학습.
   - 이를 통해 사진적 일관성과 품질을 유지할 수 있는 데이터셋을 제공.
3. **ControlNet과의 통합**
   - ControlNet을 결합하여 카메라 설정 기반으로 텍스처, 깊이, 구조 등을 정교하게 제어.
   - 이를 통해 텍스트 프롬프트와 물리적 설정의 융합으로 세밀한 이미지 제어 가능.
4. **다양한 사진적 제어 가능성**
   - 예를 들어, "Portrait with 85mm lens, f/1.4 aperture, soft background"와 같은 텍스트 입력으로 초점 심도(Depth of Field)와 빛의 효과를 반영한 이미지를 생성.

---
