---
title: "[Design it all: font, paint, and colors] ProcessPainter: Learning to draw from sequence data"
date: 2024-12-18 01:22:01
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687596>

[ProcessPainter: Learning to draw from sequence data | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687596)

### **ProcessPainter: 그림을 그리는 AI의 새로운 접근**

시그라프 아시아 2024에서 흥미로웠던 논문 중 하나는 ProcessPainter: Learning to Draw from Sequence Data입니다. 이 논문은 단순히 이미지를 한 번에 생성하는 것이 아니라, AI가 **사람처럼 점진적으로 그림을 완성하는 과정**을 학습한 연구입니다.

---

### **핵심 아이디어: 단계적 그림 생성**

1. **Sequential Learning**
   - AI는 **시퀀스 데이터**를 통해 그림을 그리는 과정을 학습합니다.
   - 각 단계는 **텍스트 프롬프트**와 **Mask 데이터**를 조건으로 사용해 **어떤 부분을 그릴지** 결정합니다.
   - 이를 통해 마치 실제 화가처럼 **단계적으로** 새로운 부분을 추가하거나 세부를 채워 나갑니다.
2. **Inverse Painting**
   - **Inverse Painting**은 그림을 완성하기까지의 **순서**를 학습하는 방식입니다.
   - 기존 Diffusion 모델처럼 한 번에 결과물을 생성하는 대신, AI는 **학습된 순서**에 따라 **단계적 예측**을 수행합니다.
3. **Artwork Replication**
   - AI는 기존의 그림을 입력으로 받아들여, 사람이 그렸던 것과 유사한 순서로 그림을 다시 그리는 능력을 갖춥니다.
   - 이 과정은 **Stroke-Based Rendering**과 다르게 시퀀스 기반으로 이루어집니다.

---
