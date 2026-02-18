---
title: "[Look at it Differently: Novel View Synthesis] Cafca: High-quality Novel View Synthesis of Expressive Faces from Casual Few-shot Captures"
date: 2024-12-19 01:08:09
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687580>

[Cafca: High-quality Novel View Synthesis of Expressive Faces from Casual Few-shot Captures | SIGGRAPH Asia 2024 Conference Paper](https://dl.acm.org/doi/10.1145/3680528.3687580)

### **CAFCA: Few-shot Capture로 표현력 있는 얼굴 생성하기**

**CAFCA (Casual Few-shot Capture for Expressive Faces)**는 단 세 장의 이미지로 고품질 3D 얼굴 모델을 생성하고, 다양한 각도에서의 새로운 시점을 합성할 수 있는 혁신적인 접근 방식을 제시합니다.

이 연구는 **데이터셋의 중요성**을 강조하며, 풍부한 다양성을 가진 **합성 데이터**를 활용해 표현력 있는 얼굴 생성의 한계를 극복했습니다.

![](/assets/images/posts/422/img.jpg)

---

### **핵심 내용 요약**

1. **합성 데이터 기반 학습**
   - 다양한 표현과 뷰포인트를 포함한 합성 데이터셋으로 **사람 얼굴의 3D 표현 사전 모델(Prior)**을 학습합니다.
   - 이 합성 데이터셋은 물리 기반의 렌더링 기법(Cycles Renderer)을 사용해 생성되어, 조명 및 텍스처의 세밀한 디테일까지 반영합니다.
2. **Few-shot Fine-tuning**
   - 합성 데이터로 학습된 사전 모델은 단 3장의 실제 이미지를 기반으로 **개인화된 3D 얼굴 모델**로 파인튜닝됩니다.
   - 파인튜닝 과정에서는 **NeRF(Neural Radiance Fields)**와 **Mip-NeRF 360** 백본을 활용해 높은 해상도의 결과를 생성합니다.
3. **고유한 정규화와 손실 설계**
   - **Implicit Regularization**과 **Explicit Regularization** 기법을 통해 미세한 디테일을 유지하면서도 과적합을 방지합니다.
   - 사용된 손실 항목에는 **Perceptual Loss**와 **Geometric Consistency Loss** 등이 포함됩니다.

---
