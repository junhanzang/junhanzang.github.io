---
title: "[Modeling and Reconstruction] FaçAID: A Transformer Model for Neuro-Symbolic Facade Reconstruction"
date: 2024-12-20 16:59:01
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - façaid: a transformer model for neuro-symbolic facade reconstruction
---

<https://dl.acm.org/doi/10.1145/3680528.3687657>

[FaçAID: A Transformer Model for Neuro-Symbolic Facade Reconstruction | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687657)

#### **1. 간단한 요약 및 소개**

FaçAID는 **건축 외관**의 디지털 재구성을 위해 **Neuro-Symbolic 접근**을 활용한 혁신적인 모델입니다. 이 논문은 Transformer를 기반으로 **split grammar**와 **tree-based procedural generation**을 결합해, 2D 이미지에서 복잡한 건축 외관을 재구성할 수 있는 기술을 제안합니다. 이 모델은 건축 디자인, 도시 계획, 게임 엔진 등 다양한 분야에서 응용 가능성을 보여줍니다.

![](/assets/images/posts/456/img.jpg)

---

#### **2. 기존 문제점**

1. **정확한 외관 재구성의 어려움**:
   - 기존의 건축 외관 모델링 기법은 이미지 세그먼트에서 복잡한 절차적 재구성을 수행하는 데 어려움을 겪음.
2. **노이즈 민감성**:
   - 입력 데이터(이미지)가 노이즈가 많은 경우, 기존 방법론은 잘못된 결과를 생성하기 쉬움.
3. **2D 이미지 중심의 제한**:
   - 기존 방법은 주로 2D 평면에서 작동하며, 3D 구조로 확장하기 어려움.
4. **자동화의 부족**:
   - 수작업으로 의존해야 하는 절차적 프로세스가 많아 효율성이 떨어짐.

---
