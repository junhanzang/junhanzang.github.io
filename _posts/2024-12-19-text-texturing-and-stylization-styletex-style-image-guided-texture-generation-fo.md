---
title: "[Text, Texturing, and Stylization] StyleTex: Style Image-Guided Texture Generation for 3D Models"
date: 2024-12-19 02:34:29
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2411.00399>

[StyleTex: Style Image-Guided Texture Generation for 3D Models](https://arxiv.org/abs/2411.00399)

### **StyleTex: 단일 스타일 이미지로 3D 텍스처 생성하기**

3D 텍스처 생성은 게임, 영화, AR/VR 등 다양한 분야에서 중요한 과제입니다. 하지만 기존 방식은 **뷰 간 일관성**, **스타일-콘텐츠 분리**, 그리고 **세부 표현** 등에서 여전히 제한점이 있었습니다. **"StyleTex"**는 이러한 문제를 해결하기 위해 설계된 **Diffusion 모델 기반 텍스처 생성 파이프라인**으로, 단일 스타일 이미지와 텍스트 프롬프트만으로도 고품질 3D 텍스처를 생성할 수 있는 혁신적인 접근법을 제안합니다.

![](/assets/images/posts/431/img.jpg)

---

### **핵심 기법 및 특징**

1. **스타일-콘텐츠 분리**
   - StyleTex는 CLIP 임베딩 공간에서 **스타일과 콘텐츠를 분리**하는 독창적인 방법을 제안합니다.
   - 참조 이미지의 스타일 임베딩을 콘텐츠 임베딩과 정교하게 분리해, 텍스처 생성 과정에서 콘텐츠 누출을 방지합니다.
2. **Geometry-Aware ControlNet**
   - 생성 과정에서 기하학적 일관성을 유지하기 위해 ControlNet을 활용해 **깊이 정보**와 **법선 맵**을 통합합니다.
3. **Interval Score Matching (ISM)**
   - 기존의 Score Distillation Sampling(SDS)을 대체하여 **오버스무딩과 과도한 채도 문제**를 해결하며, 텍스처 품질을 향상시킵니다.
4. **높은 사용자 만족도**
   - 사용자 연구 결과, StyleTex는 전반적인 품질, 스타일 충실도, 콘텐츠 제거 측면에서 기존 기법보다 훨씬 높은 점수를 기록했습니다.

---
