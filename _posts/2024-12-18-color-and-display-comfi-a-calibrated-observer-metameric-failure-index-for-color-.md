---
title: "[Color and Display] COMFI: A Calibrated Observer Metameric Failure Index for Color Critical Tasks"
date: 2024-12-18 01:43:07
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687701>

[COMFI: A Calibrated Observer Metameric Failure Index for Color Critical Tasks | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687701)

### **COMFI: 관찰자 메타메리즘 평가를 위한 새로운 지표**

이번 시그라프 아시아 2024에서 발표된 \*\*"COMFI: A Calibrated Observer Metameric Failure Index for Color Critical Tasks"\*\*는 색상 일치 평가에서 발생하는 \*\*관찰자 간의 시각 차이(OMF)\*\*를 평가하기 위한 메트릭입니다.

---

### **핵심 아이디어**

1. **OMF란 무엇인가?**
   - 같은 색상이 **다른 스펙트럼**을 가질 때, 특정 관찰자에게는 색상이 다르게 보이는 현상입니다.
   - 이는 특히 **HDR**과 **Wide Color Gamut (WCG)** 디스플레이에서 심각하게 발생할 수 있습니다.
2. **Monte Carlo 기반의 시뮬레이션**
   - **518명의 가상 관찰자**를 Monte Carlo 방법을 통해 시뮬레이션하고, 각 관찰자의 \*\*색상 민감도 함수(CMF)\*\*를 기반으로 색상 일치 정도를 평가합니다.
3. **COMFI 계산 과정**
   - 두 디스플레이의 스펙트럼을 입력으로 받아, 각각의 색상 차이를 **표준 편차**로 계산합니다.
   - 최종 결과는 **신뢰도 높은 OMF 예측 메트릭**으로 제공됩니다.
4. **실험 검증**
   - **37명의 영화 산업 전문가**를 대상으로 다양한 디스플레이를 비교 평가하는 실험을 통해 검증되었습니다.

---
