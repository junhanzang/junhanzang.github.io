---
title: "[Domo Arigato, Mr. Roboto / Robots and Characters] Decoupling Contact for Fine-Grained Motion Style Transfer"
date: 2024-12-19 02:02:23
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687609>

[Decoupling Contact for Fine-Grained Motion Style Transfer | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687609)

### **Decoupling Contact for Fine-Grained Motion Style Transfer: SIGGRAPH Asia 2024 세션 리뷰**

**Decoupling Contact for Fine-Grained Motion Style Transfer**는 모션 스타일 전환에서 접촉(contact) 제어의 중요성을 강조한 논문입니다. 이 연구는 특히 게임과 애니메이션에서 자연스러운 모션 스타일을 구현하기 위해 접촉 타이밍과 궤적(trajectory)을 세밀하게 분리하고 조정하는 방법을 제안합니다.

![](/assets/images/posts/427/img.jpg)

---

### **주요 내용**

1. **모션 스타일과 접촉의 상관관계**
   - 기존의 많은 연구는 스타일과 콘텐츠의 분리를 시도했지만, 접촉(Contact)은 이 두 요소와 강하게 연결되어 있어 분리가 어려운 부분으로 남아 있었습니다.
   - 이 논문은 **힙 속도(hip velocity)**를 접촉을 제어하기 위한 대리 변수로 활용하여, 자연스러운 모션과 스타일 표현을 가능하게 합니다.
2. **주요 접근 방식**
   - **Motion Manifold**: 모션의 스타일, 접촉 타이밍, 궤적을 각각 별도로 인코딩하고, 이를 합성하여 세밀한 제어가 가능하도록 구성.
   - **Transformer 기반 모델**: 힙 속도를 중심으로 관계를 모델링하여 접촉 타이밍을 제어.
   - **새로운 메트릭(Contact Precision-Recall)**: 인간 지각에 기반한 자연스러움을 평가하기 위해 기존 메트릭보다 정교한 측정 도구 제안.
3. **실험 결과**
   - 기존 방식에 비해 더 자연스러운 모션 생성과 스타일 표현력을 입증.
   - 특히, 접촉 타이밍과 궤적의 제어 가능성이 대폭 개선되었습니다.

---
