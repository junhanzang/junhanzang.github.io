---
title: "[Modeling and Reconstruction] DreamUDF: Generating Unsigned Distance Fields from A Single Image"
date: 2024-12-20 17:15:21
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - "dreamudf: generating unsigned distance fields from a single image"
  - dreamudf
  - generating unsigned distance fields from a single image
---

<https://dl.acm.org/doi/10.1145/3687769>

[DreamUDF: Generating Unsigned Distance Fields from A Single Image | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687769)

1. Summary & Introduction (요약 및 소개)   
  
DreamUDF는 단일 이미지로부터 3D 모델링을 위한 혁신적인 접근 방식을 소개하며, 기존의 닫힌 표면(closed-surface) 프레임워크의 한계를 극복합니다. 다중 뷰 데이터 사전 확률(priors)을 위한 확산 모델(diffusion models)과 개방적이고 임의적인 위상(arbitrary topologies)을 위한 부호 없는 거리장(UDFs: Unsigned Distance Fields)을 결합하여, 3D 형상 생성의 새로운 기준을 제시합니다.

**주요 기여:** 높은 충실도(fidelity)로 개방형 및 폐쇄형 표면을 생성할 수 있는 능력으로, 의류 디자인 및 가상 객체 재구성과 같은 실제 응용 분야에 다양하게 활용될 수 있습니다.

![](/assets/images/posts/459/img.jpg)

---

2. Existing Challenges (기존의 문제점)   
  
**제한적인 위상(Topology) 처리:** 대부분의 방법들은 닫힌 표면에 중점을 두어, 개방형 경계 모델링이 필요한 나뭇잎, 옷, 꽃과 같은 실제 객체들을 처리하는 데 어려움을 겪습니다.   
  
**기울기(Gradient) 민감도:** UDF와 함께 SDS 손실(loss)을 직접 사용하면 기울기 불안정성을 야기하여 훈련 실패로 이어집니다.   
  
**희소한(Sparse) 감독(Supervision):** 단일 이미지 입력으로는 다중 뷰 일관성을 달성하기 어려운 경우가 많아, 기하학적 구조의 견고함(robustness)이 제한됩니다.

---
