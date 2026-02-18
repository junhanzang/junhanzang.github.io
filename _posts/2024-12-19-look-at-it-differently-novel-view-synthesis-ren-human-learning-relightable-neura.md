---
title: "[Look at it Differently: Novel View Synthesis] ReN Human: Learning Relightable Neural Implicit Surfaces for Animatable Human Rendering"
date: 2024-12-19 01:35:47
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3678002>

[ReN Human: Learning Relightable Neural Implicit Surfaces for Animatable Human Rendering | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3678002)

### **ReN Human: Sparse Video로 3D 인간 모델을 생성하고 재조명까지 가능하게**

**ReN Human**은 Sparse 또는 단일 비디오 입력으로 고품질 3D 인간 모델을 생성하며, 이를 **새로운 뷰, 포즈, 조명**으로 렌더링할 수 있는 기술을 제안합니다. 하지만, 이 논문은 단순한 모델링을 넘어 **재질(Material), 기하학(Geometry), 조명(Illumination)**을 분리하는 점에서 더 깊은 기술적 접근을 시도하고 있습니다.

![](/assets/images/posts/425/img.jpg)

---

### **느낀 점: 솔직히 어려운 내용**

1. **구형 가우시안 혼합 모델(Spherical Gaussian Mixtures)**
   - 논문에서는 이 기법을 통해 공간적으로 변화하는 조명 환경을 학습하고, 인간 움직임으로 인해 발생하는 동적 차폐(Self-Occlusion)를 모델링한다고 합니다.
   - 하지만 발표와 Abstract만으로는 이 기법이 어떻게 적용되고, 실제로 어떤 이점을 가지는지 명확히 이해하지 못했습니다.
2. **물리 기반 렌더링**
   - **몬테카를로 중요도 샘플링(Monte Carlo Importance Sampling)**을 활용한다고 하지만, 렌더링 적분을 효율적으로 처리하는 과정이 구체적으로 어떻게 작동하는지 이해하기 어려웠습니다.
3. **Sparse Video에서 높은 품질**
   - 제한된 데이터로도 높은 품질을 구현했다고는 하지만, 얼마나 적은 데이터로 어떤 수준의 디테일을 유지할 수 있는지 사례를 명확히 이해하지 못했습니다.

---
