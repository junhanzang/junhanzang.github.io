---
title: "[Color and Display] Perspective-Aligned AR Mirror with Under-Display Camera"
date: 2024-12-18 01:50:38
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3687995>

[Perspective-Aligned AR Mirror with Under-Display Camera | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687995)

### **Perspective-Aligned AR Mirror: 투명 디스플레이와 카메라의 혁신적 결합**

이번에 다뤄볼 논문은 \*\*"Perspective-Aligned AR Mirror with Under-Display Camera"\*\*입니다. 기존 AR 미러 시스템의 **시점 왜곡** 문제를 해결하기 위해 **투명 디스플레이**와 카메라를 결합한 혁신적인 접근법을 제안했습니다.

---

### **핵심 아이디어**

1. **투명 디스플레이와 카메라 배치**
   - 기존 AR 미러는 카메라가 **화면 옆이나 위**에 있어 시점이 왜곡되는 단점이 있습니다.
   - 이 논문은 **투명 디스플레이 뒤에 카메라**를 배치해 **정확한 시점**에서 사용자 이미지를 획득합니다.
2. **이미지 복원**
   - 투명 디스플레이를 통해 카메라가 이미지를 캡처하면 **빛의 손실**, **블러**, **노이즈**와 같은 **저하된 품질**의 이미지가 발생합니다.
   - 이를 해결하기 위해 **물리 기반 시뮬레이션**과 **딥러닝** 기반의 **이미지 복원 알고리즘**을 설계했습니다.
3. **실시간 처리**
   - 최적화된 파이프라인을 통해 **Full HD 해상도**에서 **30fps**의 실시간 성능을 달성했습니다.
   - 이를 통해 미팅이나 AR 애플리케이션에서 **레이턴시 이슈**를 최소화했습니다.

---
