---
title: "[Modeling and Reconstruction] Large Scale Farm Scene Modeling from Remote Sensing Imagery"
date: 2024-12-20 17:04:00
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - large scale farm scene modeling from remote sensing imagery
---

<https://dl.acm.org/doi/10.1145/3687918>

[Large Scale Farm Scene Modeling from Remote Sensing Imagery | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687918)

#### **1. 간단한 요약 및 소개**

이 논문은 위성 이미지를 활용하여 대규모 농장 장면을 모델링하는 확장 가능한 프레임워크를 제안합니다. 이 시스템은 위성 데이터를 기반으로 필드, 나무, 도로, 초원을 포함한 4개의 주요 레이어를 생성하며, 사용자가 파라미터를 조정하여 다양한 농장 레이아웃과 성장 단계를 시뮬레이션할 수 있도록 설계되었습니다.

![](/assets/images/posts/457/img.jpg)

---

#### **2. 기존 문제점**

1. **복잡한 농장 구조 재구성의 어려움**:
   - 농장의 규모와 다양성을 반영하기 위해서는 고정밀 데이터와 모델이 필요하지만, 기존 방법론은 제한적.
2. **재현성과 확장성의 부족**:
   - 다양한 규모와 환경을 재현하는 데 적합하지 않음.
3. **직관적인 사용자 제어 부재**:
   - 사용자가 농장 모델의 레이아웃이나 패턴을 수정하기 어려움.

---
