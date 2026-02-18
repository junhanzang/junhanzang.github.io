---
title: "[Domo Arigato, Mr. Roboto / Robots and Characters] MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting"
date: 2024-12-19 02:06:41
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2409.14393>

[MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting](https://arxiv.org/abs/2409.14393)

### **MaskedMimic: 물리 기반 캐릭터 제어를 위한 통합 프레임워크**

**MaskedMimic**는 캐릭터 애니메이션에서 다양한 제어 방식(텍스트, 키프레임, 객체 등)을 단일 모델로 통합하는 새로운 접근법을 제안합니다. 이 논문은 특히 복잡한 장면에서 캐릭터가 동작을 자연스럽게 수행하도록 돕는 물리 기반 제어 모델을 소개합니다.

![](/assets/images/posts/428/img.jpg)

---

### **주요 내용**

1. **모션 인페인팅(Motion Inpainting) 접근법**
   - MaskedMimic은 **랜덤 마스킹된 모션 시퀀스**를 기반으로 학습되며, 누락된 데이터를 채워 전체 모션을 생성합니다.
   - 이 접근법을 통해 텍스트, 키프레임, 객체 등 다양한 입력 방식에 따라 물리적으로 타당한 동작을 생성할 수 있습니다.
2. **유니파이드 모델의 장점**
   - 기존의 작업별 컨트롤러를 따로 학습해야 했던 방식과 달리, MaskedMimic은 **단일 통합 모델**로 모든 작업을 지원합니다.
   - 이를 통해 새로운 작업이나 장면에 대한 일반화 성능이 뛰어납니다.
3. **주요 응용 분야**
   - **VR 트래킹**: 머리와 손 위치만으로 전신 동작 생성
   - **객체 상호작용**: 가구와의 자연스러운 상호작용
   - **불규칙 지형 이동**: 다양한 지형에서의 로코모션
   - **텍스트 기반 스타일링**: 텍스트 명령에 따라 동작 스타일 조정

---
