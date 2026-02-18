---
title: "[Beauty Salon: Hair, Face, Lips, and Teeth] Hairmony: Fairness-aware hairstyle classification"
date: 2024-12-20 16:51:05
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - hairmony: fairness-aware hairstyle classification
---

<https://dl.acm.org/doi/10.1145/3680528.3687582>

[Hairmony: Fairness-aware hairstyle classification | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687582)

### **Hairmony: 공정성을 고려한 헤어스타일 분류**

#### **1. 간단한 요약 및 소개**

**Hairmony**는 단일 이미지로부터 다양한 헤어스타일을 정확히 분류하는 시스템입니다. 기존 모델들이 편향되거나 특정 스타일에 한정되었던 한계를 극복하기 위해, 공정성과 다양성을 강조한 데이터셋과 새로운 분류 체계를 도입했습니다. Hairmony는 합성 데이터를 활용해 다양한 스타일을 학습하고, DINOv2 백본을 활용해 분류 성능을 향상시켰습니다.

![](/assets/images/posts/455/img.jpg)

---

#### **2. 기존 문제점**

1. **데이터 편향**: 기존 데이터셋은 특정 스타일(예: 직모)에 편중되어 있어 다양한 헤어스타일을 공정하게 학습하지 못함.
2. **분류의 한계**: 꼬임 머리나 포니테일과 같은 복잡한 스타일을 구분하기 어려움.
3. **현실 데이터 레이블링의 어려움**: 현실 세계에서 모든 이미지를 정확히 라벨링하는 것은 비용과 시간이 많이 소요됨.
4. **조명 및 시야 제한**: 강한 조명 아래 또는 특정 각도의 이미지는 분류 정확도를 떨어뜨림.

---
