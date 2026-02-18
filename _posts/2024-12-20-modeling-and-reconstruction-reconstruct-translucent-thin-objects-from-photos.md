---
title: "[Modeling and Reconstruction] Reconstruct translucent thin objects from photos"
date: 2024-12-20 17:08:13
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - reconstruct translucent thin objects from photos
---

<https://dl.acm.org/doi/10.1145/3680528.3687572>

[Reconstructing translucent thin objects from photos | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687572)

### 1. 간단한 요약 및 소개

**"Reconstructing Translucent Thin Objects from Photos"**는 얇고 반투명한 객체(예: 잎, 종이 등)의 3D 형상과 광학적 특성을 정밀하게 복원하는 시스템을 제안합니다. 이 논문은 새로운 계층적 볼륨 모델과 차별 가능한 렌더링 기법을 결합하여 복잡한 광학 효과를 복제합니다.

![](/assets/images/posts/458/img.jpg)

---

### 2. 기존 문제점

- **반투명한 객체의 특성**: 기존 방식은 이러한 객체의 다층 구조 및 광학적 특성을 정확히 재현하지 못했습니다.
- **모델 복잡성**: 광학적 파라미터와 다중 산란 효과를 동시에 최적화하는 것은 계산 비용이 높고 불안정했습니다.
- **노이즈 문제**: 몬테카를로 렌더링의 노이즈는 역 렌더링 과정에서 민감한 영향을 미쳤습니다.

---
