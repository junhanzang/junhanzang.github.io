---
title: "[Color and Display] AR-DAVID: Augmented Reality Display Artifact Video Dataset"
date: 2024-12-18 01:53:57
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3687969>

[AR-DAVID: Augmented Reality Display Artifact Video Dataset | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687969)

### **AR-DAVID: 증강현실 디스플레이의 왜곡 평가를 위한 데이터셋**

증강현실(AR) 환경은 **전통적인 디스플레이**와 달리 가상 콘텐츠와 실제 배경이 혼합되기 때문에 왜곡 평가 방식에도 차이가 있습니다. 이번에 발표된 **"AR-DAVID"** 논문은 이러한 차이점을 정량화하고 새로운 퀄리티 평가 방법의 필요성을 제시했습니다.

---

### **핵심 내용**

1. **432가지 조건**
   - **6가지 왜곡 유형**: Blur, Color Fringes, Contrast Loss 등
   - **3가지 배경 패턴**: 단색, 자연 패턴, 복잡한 텍스처
   - **2가지 밝기 수준**: 10 cd/m²와 100 cd/m²
2. **퀄리티 평가 방법**
   - **ASAP (쌍 비교)**: 참가자들이 왜곡의 가시성을 주관적으로 평가하도록 했습니다.
   - **PWCMP**: 쌍 비교 데이터를 기반으로 왜곡 결과를 정량화된 스케일로 변환했습니다.
3. **결과**
   - 배경의 밝기나 패턴은 왜곡의 가시성에 **생각보다 큰 영향을 주지 않았습니다.**
   - 기존 퀄리티 메트릭들은 AR의 **광학적 혼합 환경**을 제대로 반영하지 못해 새로운 메트릭이 필요함을 확인했습니다.

---
