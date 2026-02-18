---
title: "[Design it all: font, paint, and colors] Inverse Painting: Reconstructing The Painting Process"
date: 2024-12-18 01:39:22
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687574>

[Inverse Painting: Reconstructing The Painting Process | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687574)

### **Inverse Painting: 사람처럼 그리는 AI의 혁신적 접근**

이번 시그라프 아시아 2024에서 흥미로웠던 논문 중 하나는 \*\*"Inverse Painting: Reconstructing the Painting Process"\*\*입니다. 이 논문은 기존의 정적 이미지 생성이 아닌, **그림의 생성 과정을 시퀀스로 재구성**하여 사람의 화풍을 모방하는 접근법을 제시했습니다.

---

### **핵심 아이디어**

1. **단계적 그림 생성 (Step-by-Step)**
   - 빈 캔버스에서 시작해 한 단계씩 업데이트하며 **사람처럼 그리는 과정**을 재현합니다.
   - Diffusion 모델을 기반으로 하며, **각 단계에서 그림의 일부 영역**을 그려나가는 방식을 사용합니다.
2. **텍스트와 마스크를 활용한 두 단계 접근**
   - **텍스트 명령**: 다음에 무엇을 그릴지 예측 (예: "하늘", "산").
   - **마스크 명령**: 그림의 어느 영역에 그릴지를 지정.
   - 이 두 명령을 기반으로 **Diffusion Renderer**가 캔버스를 업데이트합니다.
3. **시간 정보(Time Interval)**
   - 프레임 간의 시간 간격을 설정해 그림의 **순차적 진행**을 제어합니다.
   - 이는 사람이 그림을 그릴 때 배경부터 그리는 **Layering 기법**과 비슷한 결과를 만듭니다.

---
