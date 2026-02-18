---
title: "[Hand and Human] World-Grounded Human Motion Recovery via Gravity-View Coordinates"
date: 2024-12-20 18:08:32
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - world-grounded human motion recovery via gravity-view coordinates
---

<https://arxiv.org/abs/2409.06662>

[World-Grounded Human Motion Recovery via Gravity-View Coordinates](https://arxiv.org/abs/2409.06662)

### 1. 간단한 요약 및 소개

**Title:** World-Grounded Human Motion Recovery via Gravity-View Coordinates  
이 논문은 단일 영상으로부터 중력과 시점에 정렬된 세계 좌표계에서 3D 인간 동작을 재구성하는 새로운 방법을 제안합니다. 기존의 카메라 중심 좌표계 기반 방식의 한계를 극복하기 위해 중력-뷰(Gravity-View) 좌표계를 활용하며, 이는 인공지능과 로봇 학습 등 다양한 응용 분야에서의 동작 재구성에 혁신적인 접근법을 제공합니다.

![](/assets/images/posts/464/img.jpg)

### 2. 기존 문제점

- 기존 카메라 중심의 인간 동작 재구성 방식은 카메라 이동 시 물리적으로 부자연스러운 결과를 초래.
- 기존의 세계 좌표계 기반 방법은 누적 오차(accumulated error) 문제로 인해 장기적인 모션 재구성이 어려움.
- 세계 좌표계를 정의하는 데 있어 회전의 자유도 문제가 존재하며, 이는 데이터 학습 과정에서 불확실성을 증가시킴.

### 3. 해결법

- **중력-뷰(Gravity-View) 좌표계 도입:** 중력 방향과 카메라 뷰 방향을 기준으로 고유한 좌표계를 정의하여, 프레임별 동작을 세계 좌표계로 변환.
- **Transformer 기반 네트워크:** RoPE(Rotary Positional Embedding)를 활용하여 시간적 상관관계를 효율적으로 학습.
- **에러 누적 방지:** 각 프레임의 동작을 독립적으로 예측하고, 카메라 회전을 통해 세계 좌표계로 일관성 있게 변환.

### 4. 기여

1. **새로운 좌표계 설계:** 중력-뷰 좌표계를 통해 누적 오차를 줄이고 중력 방향에 자연스럽게 정렬된 동작 생성.
2. **효율적인 네트워크 설계:** Transformer와 RoPE를 조합하여 긴 시퀀스에서도 강력한 일반화 성능 발휘.
3. **종합적인 성능 향상:** 카메라 공간 및 세계 공간에서 기존 방식보다 정확하고 자연스러운 동작 재구성을 입증.
4. **응용 가능성:** 이 기술은 VR/AR, 로봇 학습, 동작 분석 등 다양한 분야에 활용 가능.

### 5. 한계 및 개인적 생각

- **계산 비용:** Transformer 기반 설계로 인해 학습 및 추론 과정에서 높은 GPU 자원 요구.
- **데이터셋 한계:** 정적 배경 데이터에 의존하여 동적 배경에서의 성능 저하 가능성.
- **세부 동작 모델링:** 작은 구조(예: 손, 발)의 세부 묘사에서 제한적인 표현력을 보임.
- **추가적 발전 방향:** 다양한 데이터셋 통합과 동적 배경을 포함한 확장 연구 필요.
