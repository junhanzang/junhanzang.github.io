---
title: "[Color and Display] elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker"
date: 2024-12-18 01:47:31
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687586>

[elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker | SIGGRAPH As](https://dl.acm.org/doi/10.1145/3680528.3687586)

### **Flicer: VRR에서 깜빡임(flicker) 최적화를 위한 새로운 접근**

이번 SIGGRAPH Asia 2024에서 인상적이었던 논문 중 하나는 \*\*"elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker"\*\*입니다. 이 논문은 **깜빡임 감지와 모델링**을 개선하여 VR과 같은 가변 주사율 디스플레이 환경에서 **시각적 불편함**을 줄이기 위한 연구입니다.

---

### **핵심 내용**

1. **깜빡임(flicker)란?**
   - 인간의 시각은 특정 주파수(8-16Hz)에 **민감**하게 반응하며, 이러한 깜빡임은 시각적 피로와 불쾌감을 유발할 수 있습니다.
   - 특히 **Variable Refresh Rate (VRR)** 기술이 적용된 VR/AR 디스플레이에서 미세한 주사율 변화가 깜빡임을 유발합니다.
2. **새로운 모델: elaTCSF**
   - 기존 \*\*Temporal Contrast Sensitivity Function (TCSF)\*\*을 확장하여 **휘도(luminance)**, **시야각(eccentricity)**, \*\*영역(area)\*\*을 고려한 모델을 제안합니다.
   - 이를 통해 더 정확한 **깜빡임 감지**와 **최적화**가 가능해졌습니다.
3. **기술적 접근**
   - \*\*푸리에 변환(Fourier Transform)\*\*을 통해 깜빡임을 분석한 후, 이를 **변수화**해 새로운 감지 모델을 학습합니다.
   - VRR 환경에서 **깜빡임이 덜 보이는 안전한 프레임 범위**를 예측하는 데 사용됩니다.

---
