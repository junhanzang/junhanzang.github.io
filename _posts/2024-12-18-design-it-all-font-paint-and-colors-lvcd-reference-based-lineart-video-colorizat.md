---
title: "[Design it all: font, paint, and colors] LVCD: Reference-based Lineart Video Colorization with Diffusion Models"
date: 2024-12-18 01:30:05
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2409.12960>

[LVCD: Reference-based Lineart Video Colorization with Diffusion Models](https://arxiv.org/abs/2409.12960)

### **LVCD: 라인아트 비디오 컬러화에서 시간적 일관성을 잡다**

LVCD는 라인아트 비디오에 **시간적 일관성**을 갖춘 컬러화를 수행하는 연구입니다. Diffusion 모델과 Reference Attention을 활용해 프레임 간 컬러 전파의 자연스러움을 극대화한 점이 특징입니다.

---

### **핵심 아이디어**

1. **Reference Attention**
   - 기준 프레임(Reference Frame)에서 컬러를 추출하고 **Cross-Attention**을 통해 다음 프레임에 전달합니다.
   - 이를 통해 컬러 일관성을 유지하면서도 자연스러운 전환이 가능합니다.
2. **Overlapped Blending**
   - 비디오를 여러 세그먼트로 나누고, **겹쳐서 생성**함으로써 프레임 경계에서 발생하는 **에러 누적**을 방지합니다.
3. **Diffusion 기반 컬러화**
   - Keyframe의 컬러를 기반으로 디퓨전 모델이 나머지 영역을 채웁니다.
   - 컬러의 자연스러운 퍼짐과 세부 묘사가 가능하지만 연산량이 많아 속도가 느립니다.

---
