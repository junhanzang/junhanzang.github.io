---
title: "[Color and Display] Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation"
date: 2024-12-18 01:45:49
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

### **Large Étendue 3D Holographic Display: 더 넓어진 시야와 고화질 홀로그램**

이번 시그라프 아시아 2024에서 인상 깊었던 논문 중 하나는 \*\*"Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation"\*\*입니다. 이 논문은 기존 홀로그램 디스플레이의 한계를 뛰어넘어 **더 큰 아이박스와 고화질 이미지**를 제공하는 혁신적인 접근법을 제안했습니다.

---

### **핵심 아이디어**

1. **아이박스 확장**
   - 기존 **단일 소스** 기반 디스플레이는 시야를 조금만 벗어나도 화면이 사라지는 문제(아이박스 축소)가 있었습니다.
   - 이 논문은 **멀티 소스 조명**을 활용해 아이박스를 크게 확장하면서도 시야각을 유지했습니다.
2. **동적 푸리에 변조**
   - **푸리에 평면**에 **동적 진폭 SLM**을 배치해 콘텐츠에 맞게 **진폭과 위상**을 조정합니다.
   - 이 방법은 기존의 **고차 회절 문제**나 **고스트 이미지**를 줄이면서도, 고품질 이미지를 재구성합니다.
3. **확률적 최적화**
   - **확률적 최적화**를 활용해 목표로 하는 **라이트 필드**를 단계적으로 학습하며 \*\*시간-다중화(Time Multiplexing)\*\*를 통해 이미지 품질을 개선했습니다.

---
